import sys
import types
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
librosa = types.ModuleType("librosa")
librosa.effects = types.ModuleType("librosa.effects")
sys.modules.setdefault("librosa", librosa)
sys.modules.setdefault("librosa.effects", librosa.effects)
from rvc.lib.tools.split_audio import merge_audio


class MergeAudioTests(unittest.TestCase):
    def test_longer_chunk_stays_at_its_start(self):
        original = [np.ones(16000, dtype=np.float32)]
        converted = [np.ones(20000, dtype=np.float32)]
        intervals = np.array([[16000, 32000]])
        merged = merge_audio(original, converted, intervals, 16000, 16000)
        self.assertEqual(int(np.argmax(merged > 0)), 16000)
        self.assertTrue(np.all(merged[16000:36000] == 1))
        self.assertTrue(np.all(merged[:16000] == 0))

    def test_shorter_chunk_still_fills_its_slot(self):
        original = [np.ones(16000, dtype=np.float32)]
        converted = [np.full(12000, 2, dtype=np.float32)]
        intervals = np.array([[16000, 32000]])
        merged = merge_audio(original, converted, intervals, 16000, 16000)
        self.assertEqual(len(merged), 32000)
        self.assertTrue(np.all(merged[16000:28000] == 2))
        self.assertTrue(np.all(merged[28000:] == 0))

    def test_next_chunk_stays_on_time(self):
        original = [np.ones(16000, dtype=np.float32), np.ones(16000, dtype=np.float32)]
        converted = [
            np.ones(20000, dtype=np.float32),
            np.full(16000, 3, dtype=np.float32),
        ]
        intervals = np.array([[0, 16000], [32000, 48000]])
        merged = merge_audio(original, converted, intervals, 16000, 16000)
        self.assertTrue(np.all(merged[:20000] == 1))
        self.assertTrue(np.all(merged[20000:32000] == 0))
        self.assertTrue(np.all(merged[32000:48000] == 3))


if __name__ == "__main__":
    unittest.main()
