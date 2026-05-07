import torch
import math


class FeedForwardLayer(torch.nn.Module):
    def __init__(self, d_model, d_ff, nb_layers, dropout=0.1, lid=None, dtype=torch.float32, device='cpu'):
        """
        Initialize the FeedForward module.

        Args:
            d_model: Model dimension (D)
            d_ff: Feed-forward dimension (usually 4 * D)
            nb_layers: Number of layers
            dropout: Dropout rate
            lid: Layer ID (optional, for identification purposes)
            dtype: Data type for the tensors
            device: Device to run the layer on
        """
        super().__init__()
        # Store hyper-parameters
        self.d_model = d_model
        self.d_ff = d_ff
        self.nb_layers = nb_layers
        self.dropout = dropout
        self.device = device
        self.lid = lid
        self.layer_type = 'FeedForwardLayer'

        # Linear layers for the feed-forward network
        self.linear1 = torch.nn.Linear(d_model, d_ff, bias=False, device=device, dtype=dtype) # [D, D_ff]
        self.linear2 = torch.nn.Linear(d_ff, d_model, bias=False, device=device, dtype=dtype) # [D_ff, D]

        # Calcul des écarts-types
        std_in = 1.0 / math.sqrt(d_model) 
        std_residual = (1.0 / math.sqrt(d_ff)) / math.sqrt(2 * nb_layers)

        # Initialisation 
        torch.nn.init.normal_(self.linear1.weight, mean=0.0, std=std_in)
        if self.linear1.bias is not None:
            torch.nn.init.zeros_(self.linear1.bias)

        torch.nn.init.normal_(self.linear2.weight, mean=0.0, std=std_residual)
        if self.linear2.bias is not None:
            torch.nn.init.zeros_(self.linear2.bias)

        # Activation function
        self.activation = torch.nn.SiLU()
            
        # Norm & Dropout layer
        self.norm = torch.nn.RMSNorm(d_model, eps=1e-8, device=device, dtype=dtype)
        self.dropout_layer = torch.nn.Dropout(dropout)
    
    def forward(self, x, aim=None):
        """
        Args:
            x: Input tensor of shape (B, T, M, D) where B is batch size, T is sequence length, M is number of units, D is model dimension

        Returns:
            output: Tensor of shape (B, T, M, D)
        """
        # Apply first linear layer + activation
        x = self.norm(x) # (B, T, M, D)
        x = self.linear1(x)  # (B, T, M, D_ff)
        x = self.activation(x) # (B, T, M, D_ff)

        # Apply dropout
        x = self.dropout_layer(x) # (B, T, M, D_ff)

        # Apply second linear layer
        x = self.linear2(x)  # (B, T, M, D)

        return x

    def to(self, device):
        """Move the layer to the specified device."""
        self.device = device
        super(FeedForwardLayer, self).to(device)
        self.linear1 = self.linear1.to(device)
        self.linear2 = self.linear2.to(device)
        self.norm = self.norm.to(device)
        return self


    def __repr__(self):
        txt = "FeedForwardLayer("
        txt += f"\n  (Norm): RMSNorm{tuple(self.norm.weight.shape)},"
        txt += f"\n  (Linear1): Tensor{tuple(self.linear1.weight.shape)},"
        txt += f"\n  (Linear2): Tensor{tuple(self.linear2.weight.shape)},"
        txt += "\n)"
        return txt