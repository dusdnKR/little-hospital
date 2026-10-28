def validate_backend_config(config):
    if (
        type(config.get("do_sample")) is not bool
        or config.get("enable_thinking") is not False
    ):
        raise ValueError("Explicit sampling mode and thinking-off are required")
    if config.get("dtype") != "bfloat16":
        raise ValueError("This validated backend implements bfloat16 only")
    if config.get("backend") != "transformers_in_process_single_model":
        raise ValueError("Backend name does not match implementation")
    if any((k in config for k in ("generation", "role_generation"))):
        raise ValueError("Nested generation settings are not supported")
    if config["do_sample"]:
        if (
            not isinstance(config.get("temperature"), (int, float))
            or isinstance(config["temperature"], bool)
            or (not 0 < config["temperature"] <= 2)
            or (not isinstance(config.get("top_p"), (int, float)))
            or isinstance(config["top_p"], bool)
            or (not 0 < config["top_p"] <= 1)
            or (type(config.get("top_k")) is not int)
            or (config["top_k"] < 1)
        ):
            raise ValueError("Sampling requires valid temperature, top_p and top_k")
    elif any((k in config for k in ("temperature", "top_p", "top_k"))):
        raise ValueError("Greedy mode must not contain unused sampling parameters")
    for key in ("seed", "max_context_tokens", "image_long_side_px", "image_max_pixels"):
        value = config.get(key)
        if type(value) is not int or value < (0 if key == "seed" else 1):
            raise ValueError("Invalid integer runtime setting: " + key)
    limits = config.get("max_new_tokens")
    if (
        not isinstance(limits, dict)
        or not limits
        or any((type(v) is not int or v < 1 for v in limits.values()))
    ):
        raise ValueError("Invalid output token limits")
