"""Regression checks for playback timing and missing replay images."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image, UnidentifiedImageError

from clinical_memory.simulation.__main__ import main
from clinical_memory.simulation.replay import Attachment, Message, Replay
from clinical_memory.simulation.validate import validate
from tests.test_simulation import encounter


class SimulationCLITests(unittest.TestCase):
    def test_headless_advance_uses_selected_speed(self):
        replay = Replay({"2B": [encounter(exam=True)], "4B": [], "9B": []})
        observed = []

        def capture(screen, hospital, scale):
            observed.append(hospital.time)
            return Image.new("RGB", (20, 20))

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "preview.png"
            with (
                patch("sys.argv", ["simulate", "--speed", "4", "--advance", "2",
                                   "--screenshot", str(target)]),
                patch("clinical_memory.simulation.__main__.load_replay", return_value=replay),
                patch("clinical_memory.simulation.ui.Screen.render", autospec=True,
                      side_effect=capture),
                patch("builtins.print"),
            ):
                main()
            self.assertTrue(target.is_file())
        self.assertAlmostEqual(observed[0], 8.0)

    def test_validation_rejects_missing_image(self):
        episode = replace(encounter(), messages=(
            Message("system", "Examination", "exam", (Attachment("missing.png"),)),
        ))
        replay = Replay({"2B": [episode], "4B": [], "9B": []})
        with patch("clinical_memory.simulation.validate.load_replay", return_value=replay):
            with self.assertRaisesRegex(ValueError, "Replay image is missing"):
                validate()

    def test_validation_rejects_unreadable_image(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.png"
            path.write_bytes(b"not an image")
            episode = replace(encounter(), messages=(
                Message("system", "Examination", "exam",
                        (Attachment(path.name, str(path)),)),
            ))
            replay = Replay({"2B": [episode], "4B": [], "9B": []})
            with patch("clinical_memory.simulation.validate.load_replay", return_value=replay):
                with self.assertRaises(UnidentifiedImageError):
                    validate()


if __name__ == "__main__":
    unittest.main()
