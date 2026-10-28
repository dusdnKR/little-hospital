import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
from clinical_memory.models.contract import validate_backend_config


class TransformersBackend:
    def __init__(self, config, root, output):
        validate_backend_config(config)
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Inference requires exactly one visible GPU")
        name = torch.cuda.get_device_name(0)
        if config["allowed_gpu_name_contains"] not in name or "H100" in name:
            raise RuntimeError("Only the explicitly allocated A100 is allowed")
        self.torch, self.config, self.root, self.output = (
            torch,
            config,
            Path(root),
            Path(output),
        )
        self.calls, self.case_id = ([], None)
        self.generation_seed = config["seed"]
        call_path = self.output / "private_calls.jsonl"
        if config.get("resume_call_log") and call_path.exists():
            self.calls = [
                json.loads(s)
                for s in call_path.read_text(encoding="utf-8").splitlines()
            ]
        self.start = time.monotonic()
        torch.set_num_threads(4)
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        model_path = self.root / config["model_directory"]
        manifest = json.loads((model_path / "ksc_download_manifest.json").read_text())
        if (
            manifest["model_revision"] != config["model_revision"]
            or manifest["model_id"] != config["model_id"]
        ):
            raise RuntimeError("Model download revision mismatch")
        self.processor = AutoProcessor.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            device_map={"": 0},
            attn_implementation=config["attention"],
        )
        self.model.eval()
        self.metadata = {
            "gpu": name,
            "total_vram_bytes": torch.cuda.get_device_properties(0).total_memory,
            "model_load_seconds": round(time.monotonic() - self.start, 3),
            "model_id": config["model_id"],
            "model_revision": config["model_revision"],
            "backend": config["backend"],
            "dtype": config["dtype"],
            "attention": config["attention"],
            "cuda_runtime": torch.version.cuda,
            "enable_thinking": False,
            "versions": {
                p: importlib.metadata.version(p)
                for p in [
                    "torch",
                    "torchvision",
                    "transformers",
                    "accelerate",
                    "pillow",
                    "huggingface-hub",
                ]
            },
        }
        (self.output / "runtime.json").write_text(json.dumps(self.metadata, indent=2))

    def generate(self, role, system, payload, image_names):
        started = time.monotonic()
        try:
            return self._generate(role, system, payload, image_names)
        except Exception as error:
            failure = {
                "case_id": self.case_id,
                "role": role,
                "status": "failed",
                "error_type": type(error).__name__,
                "message": str(error)[:2000],
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "input_tokens": None,
                "output_tokens": None,
                "completed_calls_before_attempt": len(self.calls),
            }
            with (self.output / "failed_calls.jsonl").open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
            raise

    def set_seed(self, seed):
        if type(seed) is not int or seed < 0:
            raise ValueError("Invalid generation seed")
        self.generation_seed = seed
        self.torch.manual_seed(seed)
        self.torch.cuda.manual_seed_all(seed)

    def _generate(self, role, system, payload, image_names):
        from PIL import Image

        torch, cfg = (self.torch, self.config)
        if len(self.calls) >= cfg.get(
            "max_model_calls", 128
        ) or time.monotonic() - self.start > cfg.get("max_backend_seconds", 3000):
            raise RuntimeError("Model call or time budget reached")
        started = time.monotonic()
        images, image_audit = ([], [])
        base = (self.root / "data/images").resolve()
        for name in image_names:
            path = (base / name).resolve()
            if path.parent != base or not path.is_file():
                raise ValueError(
                    "Image must be a released local file in the image directory"
                )
            with Image.open(path) as original:
                im = original.convert("RGB")
                original_size = list(im.size)
            im.thumbnail(
                (cfg["image_long_side_px"], cfg["image_long_side_px"]),
                Image.Resampling.LANCZOS,
            )
            images.append(im)
            image_audit.append(
                {
                    "name": name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "original_size": original_size,
                    "input_size": list(im.size),
                }
            )
        text = json.dumps(payload, ensure_ascii=False)
        content = [{"type": "text", "text": text}]
        for name, im in zip(image_names, images):
            content.extend(
                [
                    {"type": "text", "text": "Released image: " + name},
                    {"type": "image", "image": im},
                ]
            )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]
        rendered = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        kwargs = {"text": [rendered], "return_tensors": "pt"}
        if images:
            kwargs.update(images=images, max_pixels=cfg["image_max_pixels"])
        inputs = self.processor(**kwargs)
        if images and ("pixel_values" not in inputs or "image_grid_thw" not in inputs):
            raise RuntimeError("Images were not encoded into model inputs")
        base_role = role.removesuffix("_repair")
        max_tokens = cfg["max_new_tokens"][base_role]
        input_tokens = int(inputs["input_ids"].shape[-1])
        if input_tokens + max_tokens > cfg["max_context_tokens"]:
            raise RuntimeError("Context budget exceeded; no silent truncation")
        input_shapes = {
            key: list(value.shape)
            for key, value in inputs.items()
            if hasattr(value, "shape")
        }
        grids = inputs["image_grid_thw"].tolist() if "image_grid_thw" in inputs else []
        inputs = inputs.to("cuda:0")
        torch.cuda.synchronize()
        ready = time.monotonic()
        generation = {"do_sample": cfg["do_sample"]}
        if cfg["do_sample"]:
            generation.update({k: cfg[k] for k in ("temperature", "top_p", "top_k")})
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                **generation,
                use_cache=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )
        torch.cuda.synchronize()
        suffix = generated[0, input_tokens:]
        response = self.processor.tokenizer.decode(
            suffix, skip_special_tokens=True
        ).strip()
        entry = {
            "call_index": len(self.calls) + 1,
            "case_id": self.case_id,
            "role": role,
            "generation_settings": {
                **generation,
                "enable_thinking": False,
                "max_new_tokens": max_tokens,
                "run_seed": self.generation_seed,
            },
            "rendered_prompt_sha256": hashlib.sha256(
                rendered.encode("utf-8")
            ).hexdigest(),
            "system_prompt": system,
            "payload": payload,
            "response": response,
            "input_tokens": input_tokens,
            "output_tokens": int(suffix.numel()),
            "generation_seconds": round(time.monotonic() - ready, 3),
            "total_seconds": round(time.monotonic() - started, 3),
            "images": image_audit,
            "input_shapes": input_shapes,
            "image_grid_thw": grids,
            "hit_token_cap": int(suffix.numel()) >= max_tokens,
            "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(0),
        }
        self.calls.append(entry)
        with (self.output / "private_calls.jsonl").open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return response
