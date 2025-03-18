import torch.nn as nn
import torch


class BaseModel(nn.Module):
    def __init__(self, input_dim, *args, **kwargs):
        super().__init__()
        self.input_dim = input_dim


class RegressionNN(BaseModel):
    def __init__(self, input_dim, n_neurons, activation=nn.ReLU):
        super(RegressionNN, self).__init__(input_dim)

        layers = []
        in_features = input_dim

        for out_features in n_neurons:
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.BatchNorm1d(out_features))
            layers.append(activation())
            in_features = out_features  # Update input size for next layer

        layers.append(nn.Linear(in_features, 1))  # Output layer

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        assert isinstance(x, torch.Tensor)
        assert x.shape[-1] == self.input_dim
        return self.model(x)


class TransformerRegressionNN(BaseModel):
    def __init__(self, input_dim, transformer_dim=64, nhead=4):
        """
        A simple regression model that uses a single Transformer encoder layer.

        Parameters:
            input_dim (int): Dimensionality of the input features.
            transformer_dim (int): Dimension of the transformer embedding.
            nhead (int): Number of attention heads in the transformer layer.
        """
        super(TransformerRegressionNN, self).__init__(input_dim)
        self.embedding = nn.Linear(input_dim, transformer_dim)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=nhead)
        self.output_layer = nn.Linear(transformer_dim, 1)

    def forward(self, x):
        assert isinstance(x, torch.Tensor)
        assert x.shape[-1] == self.input_dim
        embed = self.embedding(x)
        embed = embed.unsqueeze(0)
        transformer_out = self.transformer_layer(embed)
        transformer_out = transformer_out.squeeze(0)
        out = self.output_layer(transformer_out)
        return out
