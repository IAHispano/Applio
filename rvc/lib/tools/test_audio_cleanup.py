import hashlib
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from rvc.lib.tools import audio_cleanup as cleanup


class CleanupTests(unittest.TestCase):
    def test_off_is_exact_and_does_not_resolve_assets(self):
        audio = np.arange(100, dtype=np.int16)
        with patch.object(cleanup, "_runtime_assets") as assets, patch.object(
            cleanup, "_ffmpeg"
        ) as ffmpeg:
            self.assertIs(cleanup.cleanup_audio(audio, 48000), audio)
        assets.assert_not_called()
        ffmpeg.assert_not_called()

    def test_silence_and_empty_audio_do_not_download_models(self):
        with patch.object(cleanup, "_runtime_assets") as assets:
            for shape in [(0,), (20,), (20, 2)]:
                data = np.zeros(shape, np.float32)
                result = cleanup.cleanup_audio(data, 48000, "speech")
                self.assertEqual(result.shape, shape)
                self.assertFalse(result.any())
        assets.assert_not_called()

    def test_invalid_settings_fail_before_processing(self):
        audio = np.ones(10, np.float32)
        for mode, strength in [
            ("unknown", 12),
            ("speech", 0),
            ("light", 31),
            ("light", np.nan),
        ]:
            with self.subTest(mode=mode, strength=strength), self.assertRaises(
                ValueError
            ):
                cleanup.cleanup_audio(audio, 48000, mode, strength)
        with self.assertRaises(ValueError):
            cleanup.cleanup_audio(np.array([np.nan]), 48000, "speech")

    def test_failed_cleanup_keeps_the_original_file(self):
        import soundfile as sf

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "speech.wav"
            sf.write(path, np.ones(20, np.float32) * 0.1, 48000)
            original = path.read_bytes()
            with patch.object(
                cleanup, "cleanup_audio", side_effect=RuntimeError("processor failed")
            ):
                with self.assertRaisesRegex(RuntimeError, "processor failed"):
                    cleanup.cleanup_file(path, "speech")
            self.assertEqual(path.read_bytes(), original)

    def test_processor_error_is_not_silently_bypassed(self):
        with patch.object(cleanup.subprocess, "run") as run:
            run.return_value.returncode = 1
            run.return_value.stderr = "missing model"
            with self.assertRaisesRegex(RuntimeError, "missing model"):
                cleanup._run(["processor"], 1)

    def test_checksum_is_rechecked_and_corrupt_cache_is_repaired(self):
        runtime, model = b"runtime", b"model"
        runtime_hash = hashlib.sha256(runtime).hexdigest()
        model_hash = hashlib.sha256(model).hexdigest()
        downloads = []

        def download(url, destination, sha256):
            data = model if str(destination).endswith(".tar.gz") else runtime
            self.assertEqual(hashlib.sha256(data).hexdigest(), sha256)
            Path(destination).write_bytes(data)
            downloads.append(url)

        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"APPLIO_CLEANUP_CACHE": directory}
        ), patch.object(
            cleanup.platform, "system", return_value="Windows"
        ), patch.object(
            cleanup.platform, "machine", return_value="AMD64"
        ), patch.object(
            cleanup, "RUNTIMES", {("Windows", "x86_64"): ("runtime.exe", runtime_hash)}
        ), patch.object(
            cleanup, "MODEL_HASH", model_hash
        ), patch.object(
            cleanup, "download_http", side_effect=download
        ):
            executable, _ = cleanup._runtime_assets()
            self.assertEqual(len(downloads), 2)
            cleanup._runtime_assets()
            self.assertEqual(len(downloads), 2)
            executable.write_bytes(b"corrupt")
            cleanup._runtime_assets()
            self.assertEqual(len(downloads), 3)
            self.assertEqual(executable.read_bytes(), runtime)
            self.assertTrue((executable.parent / "LICENSE-MIT").is_file())


@unittest.skipUnless(
    os.environ.get("APPLIO_TEST_CLEANUP_RUNTIME"),
    "Set APPLIO_TEST_CLEANUP_RUNTIME to run native processor checks",
)
class NativeCleanupTests(unittest.TestCase):
    def test_sample_count_channels_rates_and_native_tail(self):
        rng = np.random.default_rng(42)
        for mode in ["light", "speech"]:
            for rate in [16000, 32000, 44100, 48000]:
                with self.subTest(mode=mode, rate=rate):
                    t = np.arange(rate // 2 + 17) / rate
                    mono = (
                        0.1 * np.sin(2 * np.pi * 300 * t) + rng.normal(0, 0.005, len(t))
                    ).astype(np.float32)
                    source = np.column_stack((mono, mono * 0.5))
                    result = cleanup.cleanup_audio(source, rate, mode, 12)
                    self.assertEqual(result.shape, source.shape)
                    self.assertTrue(np.isfinite(result).all())
                    self.assertGreater(float(np.max(abs(result[-rate // 100 :]))), 0)

    @unittest.skipUnless(
        os.environ.get("APPLIO_TEST_SPEECH_WAV"), "Provide a local speech fixture"
    )
    def test_spoken_tail_survives_flush(self):
        import soundfile as sf

        source, rate = sf.read(os.environ["APPLIO_TEST_SPEECH_WAV"], dtype="float32")
        # End on active speech rather than letting trailing fixture silence
        # hide truncation. Pure tones are not valid speech-preservation tests.
        active = np.flatnonzero(abs(source) > 0.03)
        source = source[max(0, active[-1] - rate * 2) : active[-1] + 1]
        result = cleanup.cleanup_audio(source, rate, "speech", 12)
        self.assertEqual(result.shape, source.shape)
        self.assertGreater(float(np.max(abs(result[-rate // 20 :]))), 0.005)

    def test_int16_level_is_normalized_not_clipped(self):
        t = np.arange(48000) / 48000
        mono = (np.sin(2 * np.pi * 300 * t) * 6000).astype(np.int16)
        result = cleanup.cleanup_audio(mono, 48000, "light", 6)
        self.assertLess(np.max(abs(result)), 0.3)
        self.assertGreater(np.max(abs(result)), 0.1)


if __name__ == "__main__":
    unittest.main()
