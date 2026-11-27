import tempfile
import time
import unittest
from pathlib import Path

import contest_submission
import numpy as np

from gas_a.pair_review import PairReview


class PairReviewTests(unittest.TestCase):
    def test_only_targeted_disagreements_swap_scores(self):
        base = np.array([[.8, .1, .04, .02, .02, .02],
                         [.02, .1, .7, .02, .08, .08],
                         [.05, .02, .01, .02, .1, .8],
                         [.65, .3, .01, .01, .01, .02]])
        original = base.copy()
        evidence = (np.array([1, 1, 4, 5]), np.array([True, True, False, False]), np.array([False, False, True, False]))
        reviewed = PairReview.apply(base, evidence)
        np.testing.assert_array_equal(reviewed.argmax(axis=1), [1, 2, 4, 0])
        np.testing.assert_array_equal(np.sort(reviewed, axis=1), np.sort(base, axis=1))
        np.testing.assert_array_equal(base, original)

    def test_fitted_model_round_trip_and_corrupt_scale(self):
        rng = np.random.default_rng(42)
        y = np.repeat(np.arange(6), 8)
        x = rng.normal(0., .1, (len(y), 128))
        x[np.arange(len(y)), 8 * y] += 4.
        review = PairReview.fit({"x": x, "y": y}, time.monotonic() + 10.)
        expected = review.evidence(x)
        np.testing.assert_array_equal(expected[0], y)
        with tempfile.TemporaryDirectory() as work:
            path = Path(work) / "review.npz"
            review.save(path)
            restored = PairReview.load(path)
            self.assertEqual(restored.parameter_count, review.parameter_count)
            for actual, reference in zip(restored.evidence(x), expected):
                np.testing.assert_array_equal(actual, reference)
            with np.load(path, allow_pickle=False) as good:
                payload = dict(good)
            payload["scale_0"][:] = 0.
            np.savez(path, **payload)
            with self.assertRaisesRegex(ValueError, "Invalid pair review"):
                PairReview.load(path)
            path.write_bytes(b"PK\x03\x04invalid zip archive")
            with self.assertRaisesRegex(ValueError, "Invalid pair review"):
                PairReview.load(path)


if __name__ == "__main__":
    unittest.main()
