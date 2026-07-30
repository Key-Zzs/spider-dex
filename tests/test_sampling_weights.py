"""Regression coverage for bounded, tiny MJWP candidate sets."""

import unittest

import torch

from spider.optimizers.sampling import _compute_weights_impl


class SamplingWeightTest(unittest.TestCase):
    def test_single_top_candidate_is_finite(self) -> None:
        weights, invalid = _compute_weights_impl(
            torch.tensor([0.25, -0.5], dtype=torch.float32), num_samples=2, temperature=0.2
        )
        self.assertTrue(torch.isfinite(weights).all())
        self.assertFalse(invalid.any())
        self.assertAlmostEqual(float(weights.sum()), 1.0)


if __name__ == "__main__":
    unittest.main()
