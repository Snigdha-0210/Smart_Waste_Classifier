import math
import numpy as np
import pytest
import torch
import torch.nn.functional as F


class TestOODMetrics:
    """Test suite for Out-Of-Distribution (OOD) Guard V1 mathematical logic."""

    def test_shannon_entropy_calculation(self):
        """Verify Shannon entropy computation on uniform vs peak probability distributions."""
        # Uniform distribution across 6 classes (maximum entropy / highest uncertainty)
        uniform_probs = torch.tensor([1/6] * 6)
        max_entropy = -torch.sum(uniform_probs * torch.log(uniform_probs + 1e-12)).item()
        expected_max = math.log(6)
        assert pytest.approx(max_entropy, rel=1e-3) == expected_max

        # Deterministic / highly confident distribution (minimum entropy / lowest uncertainty)
        confident_probs = torch.tensor([0.999, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002])
        low_entropy = -torch.sum(confident_probs * torch.log(confident_probs + 1e-12)).item()
        assert low_entropy < 0.05
        assert low_entropy < max_entropy

    def test_prediction_margin_delta(self):
        """Verify margin delta calculation (top1 confidence - top2 confidence)."""
        probs = torch.tensor([0.85, 0.10, 0.02, 0.01, 0.01, 0.01])
        sorted_probs, _ = torch.sort(probs, descending=True)
        margin_delta = (sorted_probs[0] - sorted_probs[1]).item()

        assert pytest.approx(margin_delta, rel=1e-3) == 0.75

    def test_ood_flag_logic(self):
        """Verify OOD flag triggers on high entropy and low confidence."""
        entropy_threshold = 1.2
        confidence_threshold = 0.55

        # Case 1: In-distribution sample
        in_dist_conf = 0.94
        in_dist_entropy = 0.25
        is_ood_1 = (in_dist_conf < confidence_threshold) or (in_dist_entropy > entropy_threshold)
        assert not is_ood_1

        # Case 2: Out-of-distribution sample (e.g. random noise or human face)
        ood_conf = 0.35
        ood_entropy = 1.55
        is_ood_2 = (ood_conf < confidence_threshold) or (ood_entropy > entropy_threshold)
        assert is_ood_2
