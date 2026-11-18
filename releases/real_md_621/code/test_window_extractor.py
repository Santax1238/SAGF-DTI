import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location('window_extractor', Path(__file__).with_name('window_extractor.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class WindowTests(unittest.TestCase):
    def test_half_partition_odd_length(self):
        chunks = [np.arange(0, 4), np.arange(4, 8), np.arange(8, 11)]
        first = np.concatenate(list(m.slice_chunks(iter(chunks), 0, 5)))
        second = np.concatenate(list(m.slice_chunks(iter(chunks), 5, 11)))
        np.testing.assert_array_equal(first, np.arange(5))
        np.testing.assert_array_equal(second, np.arange(5, 11))
        np.testing.assert_array_equal(np.r_[first, second], np.arange(11))

    def test_full_window_is_identity(self):
        chunks = [np.arange(0, 2), np.arange(2, 3), np.arange(3, 10)]
        np.testing.assert_array_equal(np.concatenate(list(m.slice_chunks(iter(chunks), 0, 10))), np.arange(10))

    def test_short_source_fails(self):
        with self.assertRaises(ValueError):
            list(m.slice_chunks(iter([np.arange(3)]), 2, 5))

    def test_invalid_window_fails(self):
        for a, b in [(-1, 3), (2, 2), (4, 3)]:
            with self.assertRaises(ValueError):
                list(m.slice_chunks(iter([np.arange(5)]), a, b))

    def test_no_read_after_window(self):
        def source():
            yield np.arange(10)
            raise AssertionError('reader must stop at the requested frame')
        np.testing.assert_array_equal(np.concatenate(list(m.slice_chunks(source(), 1, 4))), np.arange(1, 4))


if __name__ == '__main__':
    unittest.main()
