import pytest
import torch
import torch.nn as nn
from model import WasteCNN
from model_resnet import WasteResNet


class TestModelArchitectures:
    """Test suite for neural network architectures in Smart Waste Classifier."""

    def test_waste_cnn_initialization(self):
        """Verify baseline WasteCNN initializes properly with 6 classes."""
        model = WasteCNN(num_classes=6)
        assert isinstance(model, nn.Module)
        assert hasattr(model, "features")
        assert hasattr(model, "classifier")

    def test_waste_cnn_forward_pass(self):
        """Verify baseline WasteCNN forward pass and output shape."""
        model = WasteCNN(num_classes=6)
        model.eval()
        dummy_input = torch.randn(2, 3, 128, 128)
        with torch.no_grad():
            output = model(dummy_input)
        assert output.shape == (2, 6)
        assert not torch.isnan(output).any()

    def test_waste_resnet_initialization(self):
        """Verify WasteResNet initializes with default 6 classes."""
        model = WasteResNet(num_classes=6)
        assert isinstance(model, nn.Module)
        assert model.model.fc.out_features == 6

    def test_waste_resnet_custom_classes(self):
        """Verify WasteResNet can be initialized with arbitrary class counts."""
        model = WasteResNet(num_classes=10)
        assert model.model.fc.out_features == 10

    def test_waste_resnet_forward_pass(self):
        """Verify WasteResNet forward pass with standard 224x224 RGB tensors."""
        model = WasteResNet(num_classes=6)
        model.eval()
        dummy_input = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        assert output.shape == (2, 6)
        assert not torch.isnan(output).any()

    def test_waste_resnet_gradient_flow(self):
        """Verify gradient computation flow during backpropagation."""
        model = WasteResNet(num_classes=6)
        model.train()
        dummy_input = torch.randn(2, 3, 224, 224)
        target = torch.tensor([0, 5], dtype=torch.long)
        criterion = nn.CrossEntropyLoss()
        output = model(dummy_input)
        loss = criterion(output, target)
        loss.backward()
        
        # Check that gradients exist for the final fully connected layer
        assert model.model.fc.weight.grad is not None
        assert not torch.isnan(model.model.fc.weight.grad).any()
