import pytest
import torch
from src.model.model import BaseModel, RegressionNN, TransformerRegressionNN


def test_base_model():
    model = BaseModel(input_dim=10)
    assert model.input_dim == 10


@pytest.mark.parametrize("input_dim, n_neurons", [(10, [32, 16]), (20, [64, 32, 16])])
def test_regression_nn(input_dim, n_neurons):
    model = RegressionNN(input_dim, n_neurons)
    assert isinstance(model, RegressionNN)

    x = torch.randn(5, input_dim)
    output = model(x)
    assert output.shape == (5, 1)


@pytest.mark.parametrize("input_dim, transformer_dim, nhead", [(10, 64, 4), (20, 128, 8)])
def test_transformer_regression_nn(input_dim, transformer_dim, nhead):
    model = TransformerRegressionNN(input_dim, transformer_dim, nhead)
    assert isinstance(model, TransformerRegressionNN)
    x = torch.randn(5, input_dim)
    output = model(x)
    assert output.shape == (5, 1)
