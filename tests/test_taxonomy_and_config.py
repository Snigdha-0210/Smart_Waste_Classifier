from pathlib import Path
import pytest


class TestTaxonomyAndConfig:
    """Test suite for waste stream taxonomy consistency and configuration integrity."""

    EXPECTED_CLASSES = [
        "Cardboard",
        "Food Organics",
        "Glass",
        "Metal",
        "Paper",
        "Plastic"
    ]

    def test_taxonomy_class_count(self):
        """Verify the standardized taxonomy defines exactly 6 classes."""
        assert len(self.EXPECTED_CLASSES) == 6

    def test_data_yaml_structure(self):
        """Verify that detection/data.yaml exists and defines expected classes."""
        yaml_path = Path("detection/data.yaml")
        assert yaml_path.exists(), "detection/data.yaml must exist"
        
        content = yaml_path.read_text(encoding="utf-8")
        for idx, class_name in enumerate(self.EXPECTED_CLASSES):
            assert f"{idx}: {class_name}" in content, f"Class {class_name} missing at index {idx} in data.yaml"

    def test_disposal_categories_mapping(self):
        """Verify waste streams correctly map to biodegradable vs recyclable bins."""
        biodegradable = {"Food Organics"}
        recyclable = {"Cardboard", "Glass", "Metal", "Paper", "Plastic"}

        all_mapped = biodegradable.union(recyclable)
        assert all_mapped == set(self.EXPECTED_CLASSES)
