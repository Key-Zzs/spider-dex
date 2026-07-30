"""Regression tests for the self-contained Wuji Hand2 Beta1 adapter."""

from __future__ import annotations

import unittest

from tools.validate_wuji_hand2 import validate


class WujiHand2AssetsTest(unittest.TestCase):
    """Exercise static assets, MuJoCo contracts, hold behavior, and staging."""

    def test_assets_and_runtime_staging(self) -> None:
        stats = validate()
        self.assertEqual(stats["right"]["nu"], 26)
        self.assertEqual(stats["left"]["nu"], 26)
        self.assertEqual(stats["bimanual"]["nu"], 52)


if __name__ == "__main__":
    unittest.main()
