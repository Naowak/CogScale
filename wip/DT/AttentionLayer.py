import torch
import torch.nn as nn
import math


class AttentionLayer(nn.Module):

    def __init__(self, d_model, nb_layers, num_heads=8, dropout=0.1, self_attention=True, lid=None, dtype=torch.float32, device='cpu'):
        """Initialize the AttentionLayer module.
            
        Args:
            d_model: Model dimension (D)
            nb_layers: Number of layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            self_attention: If True, use self-attention; else, use encoder-decoder attention
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model doit être divisible par num_heads"
        # Store hyper-parameters
        self.d_model = d_model
        self.nb_layers = nb_layers
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension par tête
        self.self_attention = self_attention
        self.lid = lid
        self.dtype = dtype
        self.device = device
        self.layer_type = 'AttentionLayer'

        # Linear projection for Q, K, V
        self.W_q = nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype) # [D, D]
        self.W_k = nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype) # [D, D]
        self.W_v = nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype) # [D, D]

        # Output projection
        self.W_o = nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype) # [D, D]

        # Std compute
        std_normal = 1.0 / math.sqrt(d_model) # Logique type Xavier
        std_residual = std_normal / math.sqrt(2 * nb_layers) # Rescaling OpenAI

        # Initialization Q, K, V
        for module in [self.W_q, self.W_k, self.W_v]:
            torch.nn.init.normal_(module.weight, mean=0.0, std=std_normal)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        # Initialization Residual
        torch.nn.init.normal_(self.W_o.weight, mean=0.0, std=std_residual)
        if self.W_o.bias is not None:
            torch.nn.init.zeros_(self.W_o.bias)

        # Norm & Dropout & scaling factor
        self.norm = nn.RMSNorm(d_model, eps=1e-8, device=device, dtype=dtype)
        self.dropout_layer = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, x, state=None, aim=None):
        """
        Args:
            x: Input tensor of shape (B, T, M, D)
            state: Optional state tensor for encoder-decoder attention of shape (B, T, M, D)
        Returns:
            output: Tensor of shape (B, T, M, D)
        """
        # Check state (not used in self-attention)
        if state is None and not self.self_attention:
            raise ValueError("State must be provided for non-self-attention.")

        batch_size, seq_len, units, _ = x.shape
        x = self.norm(x) # (B, T, M, D)

        # Linear projections
        if self.self_attention:
            Q = self.W_q(x) # (B, T, M, D)
            K = self.W_k(x) # (B, T, M, D)
            V = self.W_v(x) # (B, T, M, D)
        else:
            Q = self.W_q(x) # (B, T, M, D)
            K = self.W_k(state) # (B, T, M, D)
            V = self.W_v(state) # (B, T, M, D)

        # Reshape for multi-head: (B, T, H, M, D)
        Q = Q.view(batch_size, seq_len, units, self.num_heads, self.d_k).transpose(2, 3) # (B, T, H, M, d_k)
        K = K.view(batch_size, seq_len, units, self.num_heads, self.d_k).transpose(2, 3) # (B, T, H, M, d_k)
        V = V.view(batch_size, seq_len, units, self.num_heads, self.d_k).transpose(2, 3) # (B, T, H, M, d_k)

        # Compute attention scores: Q @ K^T / sqrt(d_k)
        scores = Q @ K.transpose(-2, -1) / self.scale # (B, T, H, M, M)

        # Softmax along the last dimension (M)
        attn_weights = torch.softmax(scores, dim=-1) # (B, T, H, M, M)
        
        # Track attention score if aim is provided
        if aim is not None:
            for t in range(attn_weights.shape[1]):
                for h in range(attn_weights.shape[2]):
                    for m1 in range(attn_weights.shape[3]):
                        for m2 in range(attn_weights.shape[4]):
                            aim.track(attn_weights[0, t, h, m1, m2], name=f"AttentionScores_Layer{self.lid}_Head{h}_Unit{m1}_to_{m2}")
        
        attn_weights = self.dropout_layer(attn_weights) # (B, T, H, M, M)
        

        # Apply attention on V
        attn_output = attn_weights @ V # (B, T, H, M, d_k)

        # Concatenate heads
        attn_output = attn_output.transpose(2, 3).contiguous() # (B, T, M, H, d_k)
        attn_output = attn_output.view(batch_size, seq_len, units, self.d_model) # (B, T, M, D)

        # Final projection
        output = self.W_o(attn_output) # (B, T, M, D)

        return output
    
    # def forward(self, x, state=None, aim=None):
    #     """
    #     Args:
    #         x: Input tensor of shape (B, T, M, D)
    #         state: Optional state tensor for encoder-decoder attention of shape (B, T, M, D)
    #     Returns:
    #         output: Tensor of shape (B, T, M, D)
    #     """
    #     # Check state (not used in self-attention)
    #     if state is None and not self.self_attention:
    #         raise ValueError("State must be provided for non-self-attention.")

    #     batch_size, seq_len, units, _ = x.shape
    #     x = self.norm(x) # (B, T, M, D)

    #     # Linear projections
    #     if self.self_attention:
    #         Q = self.W_q(x) # (B, T, M, D)
    #         K = self.W_k(x) # (B, T, M, D)
    #         V = self.W_v(x) # (B, T, M, D)
    #     else:
    #         Q = self.W_q(x) # (B, T, M, D)
    #         K = self.W_k(state) # (B, T, M, D)
    #         V = self.W_v(state) # (B, T, M, D)

    #     # 1. Reshape for multi-head attention 
    #     Q = Q.view(batch_size, seq_len, units, self.num_heads, self.d_k) # (B, T, M, H, d_k)
    #     K = K.view(batch_size, seq_len, units, self.num_heads, self.d_k) # (B, T, M, H, d_k)
    #     V = V.view(batch_size, seq_len, units, self.num_heads, self.d_k) # (B, T, M, H, d_k)

    #     # 2. Compute attention scores using einsum for efficiency
    #     scores = torch.einsum('btmhd,btnhd->bthmn', Q, K) / self.scale # (B, T, H, M, M)

    #     # Softmax sur la dernière dimension (les "keys" M, c'est-à-dire 'n')
    #     attn_weights = torch.softmax(scores, dim=-1) # (B, T, H, M, M)
        
    #     # Track attention score if aim is provided
    #     if aim is not None:
    #         for t in range(attn_weights.shape[1]):
    #             for h in range(attn_weights.shape[2]):
    #                 for m1 in range(attn_weights.shape[3]):
    #                     for m2 in range(attn_weights.shape[4]):
    #                         aim.track(attn_weights[0, t, h, m1, m2], name=f"AttentionScores_Layer{self.lid}_Head{h}_Unit{m1}_to_{m2}")
        
    #     attn_weights = self.dropout_layer(attn_weights) # (B, T, H, M, M)

    #     # 3. Apply attention on V using einsum
    #     attn_output = torch.einsum('bthmn,btnhd->btmhd', attn_weights, V) # (B, T, M, H, d_k)

    #     # 4. Concatenate heads
    #     attn_output = attn_output.reshape(batch_size, seq_len, units, self.d_model)

    #     # 5. Final projection
    #     output = self.W_o(attn_output) # (B, T, M, D)

    #     return output

    def to(self, device):
        """Move the layer to the specified device."""
        self.device = device
        super(AttentionLayer, self).to(device)
        self.W_q = self.W_q.to(device)
        self.W_k = self.W_k.to(device)
        self.W_v = self.W_v.to(device)
        self.W_o = self.W_o.to(device)
        self.norm = self.norm.to(device)
        return self

    def __repr__(self):
        txt = "AttentionLayer("
        txt += f"\n  (Norm): RMSNorm{tuple(self.norm.weight.shape)},"
        txt += f"\n  (Q): Tensor{tuple(self.W_q.weight.shape)},"
        txt += f"\n  (K): Tensor{tuple(self.W_k.weight.shape)},"
        txt += f"\n  (V): Tensor{tuple(self.W_v.weight.shape)},"
        txt += f"\n  (O): Tensor{tuple(self.W_o.weight.shape)},"
        txt += "\n)"
        return txt
