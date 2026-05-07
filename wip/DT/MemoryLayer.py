import torch
import numpy as np
import math
# from libs.DT.triton.multihead.ssm_triton_multihead import parallel_complex_ssm
try:
    from DT.triton.fused.ssm_triton_fused import parallel_fused_ssm
except ImportError:
    parallel_fused_ssm = None

class MemoryLayer(torch.nn.Module):
    """Implements a reservoir network."""

    def __init__(self, units=None, neurons=None, input_dim=None, output_dim=None, d_conv=4, res_connectivity=0.2, 
                 input_connectivity=0.2, device='cpu', dtype=torch.float32, complex_dtype=torch.complex64):
        """
        Create a reservoir with the given parameters.

        Args:
        - units (int): Number of reservoirs.
        - neurons (int): Number of neurons in each reservoir.
        - input_dim (int): Input dimension.
        - output_dim (int): Output dimension.
        - res_connectivity (float): Connectivity of the recurrent weight matrix.
        - input_connectivity (float): Connectivity of the input weight matrix.
        - device (str): Device to use ('cpu' or 'cuda').
        - dtype (torch.dtype): Data type of the tensors.
        - complex_dtype (torch.dtype): Data type for complex tensors.
        """
        super().__init__()
        # Check the parameters
        if units is None or neurons is None or input_dim is None or output_dim is None:
            raise ValueError("You must provide the number of units, neurons and input/output dimension")
        
        # Store the parameters
        self.units = units # M
        self.neurons = neurons # R
        self.input_dim = input_dim # D
        self.output_dim = output_dim # D
        self.d_conv = d_conv
        self.res_connectivity = res_connectivity
        self.input_connectivity = input_connectivity
        self.device = device
        self.dtype = dtype
        self.complex_dtype = complex_dtype
        self.layer_type = 'MemoryLayer'

        with torch.no_grad():
            # Projection
            scale_proj = 1.0 / math.sqrt(input_dim) # Xavier scaling
            scale_win = 1.0 / math.sqrt(input_dim) # Xavier scaling
            scale_wout = 1.0 / math.sqrt(neurons) # Xavier scaling
            proj = _initialize_matrix((units, input_dim, input_dim), 1.0, distribution='normal', dtype=dtype, device=device) * scale_proj

            # Conv1d
            self.channels = self.units * self.input_dim
            self.conv1d = torch.nn.Conv1d(
                in_channels=self.channels,
                out_channels=self.channels,
                kernel_size=self.d_conv,
                groups=self.channels, # Depthwise strict : 1 filtre par canal de chaque unité
                padding=self.d_conv - 1,
                device=device,
                dtype=dtype
            )

            # --- 1. Adaptive Leak Rate (Log-Uniform Initialization) ---
            lr_min = 0.01
            lr_max = 1
            lr_bias_target = torch.exp(torch.rand(self.units, 1, 1, dtype=dtype, device=device) * (math.log(lr_max) - math.log(lr_min)) + math.log(lr_min))
            lr_bias = torch.logit(lr_bias_target, eps=1e-6)
            adaptive_lr = _initialize_matrix((units, input_dim, 1), 1.0, distribution='normal', dtype=dtype, device=device)

            # --- 2. Orthogonalité parfaite (Valeurs propres sur le cercle unité) ---
            theta = torch.rand((units, neurons), dtype=dtype, device=device) * 2 * math.pi
            Lambda_complex = torch.complex(torch.cos(theta), torch.sin(theta))

            # --- 3. Win et Wout projetés ---
            Win_real = _initialize_matrix((units, input_dim, neurons), input_connectivity, distribution='normal', dtype=dtype, device=device) * scale_win
            Win_imag = _initialize_matrix((units, input_dim, neurons), input_connectivity, distribution='normal', dtype=dtype, device=device) * scale_win
            Win_complex = torch.complex(Win_real, Win_imag)

            Wout_real = _initialize_matrix((units, neurons, output_dim), 1.0, distribution='normal', dtype=dtype, device=device) * scale_wout
            Wout_imag = _initialize_matrix((units, neurons, output_dim), 1.0, distribution='normal', dtype=dtype, device=device) * scale_wout
            Wout_complex = torch.complex(Wout_real, Wout_imag)

            # --- 4. Enregistrement des paramètres ---
            self.proj = torch.nn.Parameter(proj) 
            self.adaptive_lr = torch.nn.Parameter(adaptive_lr)
            self.lr_bias = torch.nn.Parameter(lr_bias)
            
            self.Win_real = torch.nn.Parameter(Win_complex.real.contiguous())
            self.Win_imag = torch.nn.Parameter(Win_complex.imag.contiguous())
            
            self.Lambda_real = torch.nn.Parameter(Lambda_complex.real.contiguous())
            self.Lambda_imag = torch.nn.Parameter(Lambda_complex.imag.contiguous())
            
            self.Wout_real = torch.nn.Parameter(Wout_complex.real.contiguous())
            self.Wout_imag = torch.nn.Parameter(Wout_complex.imag.contiguous())
            self.norm = torch.nn.RMSNorm(output_dim, eps=1e-8, device=device, dtype=dtype)
            

    def forward(self, X, state=None, chunk_size=256, aim=None):
        """
        Forward pass of the reservoir network.
        
        Parameters:
        - X (torch.Tensor): Input tensor [batch, time, input_dim].
        - state (torch.Tensor, optional): Initial states [batch, units, neurons].
        - chunk_size (int): Chunk size for triton SSM computation.

        Returns:
        - new_state (torch.Tensor): Updated state [batch, time, units, neurons].
        - output (torch.Tensor): Output tensor [batch, time, units, output_dim].
        """  
        # Select the forward computation method
        if parallel_fused_ssm is None:
            # print("Warning: Triton fused SSM is not available, using standard computation.")
            f = self._forward_computation
        else:
            # print("Using Triton fused SSM for computation.")
            f = self._forward_computation_triton

        return f(X, state, chunk_size=chunk_size, aim=aim)
    
    def to(self, device):
        """Move the layer to the specified device."""
        self.device = device
        super(MemoryLayer, self).to(device)
        self.proj = self.proj.to(device)
        self.conv1d = self.conv1d.to(device)
        self.adaptive_lr = self.adaptive_lr.to(device)
        self.lr_bias = self.lr_bias.to(device)
        # self.temperature = self.temperature.to(device)
        self.Win_real = self.Win_real.to(device)
        self.Win_imag = self.Win_imag.to(device)
        self.Lambda_real = self.Lambda_real.to(device)
        self.Lambda_imag = self.Lambda_imag.to(device)
        self.Wout_real = self.Wout_real.to(device)
        self.Wout_imag = self.Wout_imag.to(device)
        self.norm = self.norm.to(device)
        return self

    # def _forward_computation(self, X, state=None, aim=None, **kwargs):
    #     """
    #     Version standard.
    #     """
    #     # Gestion des états : Unpacking du tuple
    #     if state is not None and isinstance(state, tuple):
    #         ssm_state, conv_state = state
    #     else:
    #         ssm_state = None 
    #         conv_state = None

    #     # Apply RMSNorm to the input
    #     X = self.norm(X) # [B, T, D] (applied before projection)

    #     # Reconstruct complex matrices
    #     Win_ = torch.complex(self.Win_real, self.Win_imag) # [M, D, R]
    #     Lambda = torch.complex(self.Lambda_real, self.Lambda_imag) # [M, R]
    #     Wout_ = torch.complex(self.Wout_real, self.Wout_imag) # [M, R, D]

    #     # Extract constants from X and reshape it
    #     batch_size = X.shape[0]
    #     seq_len = X.shape[1]
    #     X = X.view(batch_size, seq_len, 1, 1, self.input_dim) # [B, T, 1, 1, D]
    #     X_proj = X @ self.proj # [B, T, 1, 1, D] @ [M, D, D] = [B, T, M, 1, D]

    #     # Conv1D
    #     x_conv_in = X_proj.squeeze(3).view(batch_size, seq_len, self.units * self.input_dim) # [B, T, M*D]
    #     x_conv_in = x_conv_in.transpose(1, 2) # [B, M*D, T]

    #     if conv_state is not None:
    #         # Generation mode : Add the 3 previous tokens before the new one
    #         x_conv_full = torch.cat([conv_state, x_conv_in], dim=-1) # [B, M*D, (K-1) + T]
    #         x_conv_out = torch.nn.functional.conv1d(
    #             x_conv_full, 
    #             weight=self.conv1d.weight, 
    #             bias=self.conv1d.bias, 
    #             groups=self.conv1d.groups
    #         )
    #         # Cache Update
    #         new_conv_state = x_conv_full[..., -(self.d_conv - 1):] # [B, M*D, K-1]
    #     else:
    #         # Training mode / First Token : Use the normal layer with its padding
    #         x_conv_out = self.conv1d(x_conv_in)[..., :seq_len] # [B, M*D, T]
    #         # Initialize the cache for the rest with the last (K-1) elements
    #         new_conv_state = x_conv_in[..., -(self.d_conv - 1):] # [B, M*D, K-1]

    #     x_conv_out = x_conv_out.transpose(1, 2).view(batch_size, seq_len, self.units, 1, self.input_dim)
    #     X_conv = torch.nn.functional.silu(x_conv_out)

    #     # Reshape the state, if not provided, initialize it
    #     if ssm_state is None:
    #         ssm_state = torch.zeros(batch_size, 1, self.units, 1, self.neurons, dtype=self.complex_dtype, device=self.device) # [B, 1, M, 1, R]
    #     else:
    #         ssm_state = ssm_state.view(batch_size, 1, self.units, 1, self.neurons).to(self.complex_dtype) # [B, 1, M, 1, R]

    #     # Compute lr with adaptive leak rate
    #     lr_logits = (X_conv @ self.adaptive_lr) + self.lr_bias # [B, T, M, 1, 1]
    #     lr = torch.sigmoid(lr_logits) # [B, T, M, 1, 1]

    #     # Track leak rates if aim is provided
    #     if aim is not None:
    #         for t in range(lr.shape[1]):
    #             for m in range(lr.shape[2]):
    #                 aim.track(lr[0, t, m], name=f'leak_rate_{m+1}') if aim is not None else None
        
    #     # Prepare Win_ and Lambda
    #     Win_ = lr * Win_ # [B, T, M, 1, 1] * [M, D, R] = [B, T, M, D, R]
    #     Lambda = Lambda.view(1, 1, self.units, 1, self.neurons) # [1, 1, M, 1, R]
    #     Lambda = lr * Lambda + (1 - lr) # [B, T, M, 1, 1] * [1, 1, M, 1, R] = [B, T, M, 1, R]
        
    #     # Compute feed and concat the initial state
    #     feed = X_conv.to(self.complex_dtype) @ Win_ # [B, T, M, 1, R]
    #     feed = torch.cat((ssm_state, feed), dim=1) # [B, T+1, M, 1, R]
    #     feed = feed.view(batch_size, 1, seq_len+1, self.units, self.neurons) # [B, 1, T+1, M, R]

    #     # Compute echos, states and apply RMSNorm
    #     Lambda_compact = _compute_Lambda_compact(Lambda.squeeze(3)) # [B, (T^2+T)/2, M, R]
    #     feed = _compute_feed_compact(feed.squeeze(1)) # [B, (T^2+T)/2, M, R]
    #     echos = (feed * Lambda_compact) # [B, (T^2+T)/2, M, R]
    #     new_ssm_state = _compute_new_state(echos, seq_len) # [B, T, M, R]

    #     # Compute the output
    #     output = (new_ssm_state.unsqueeze(3) @ Wout_).real.to(self.dtype).squeeze(3) # [B, T, M, D]
    #     output = torch.nn.functional.silu(output.to(self.dtype)) # Apply activation function [B, T, M, D]
    #     new_state = (new_ssm_state, new_conv_state) # [B, T, M, R], [B, M*D, K-1]

    #     # Track output if aim is provided
    #     if aim is not None:
    #         for t in range(output.shape[1]):
    #             for m in range(output.shape[2]):
    #                 for d in range(min(output.shape[3], 4)):
    #                     aim.track(output[:, t, m, d], name=f'output_{m+1}_{d+1}', epoch=None)

    #     return output, new_state  # [B, T, M, D], [B, T, M, R]

    def _forward_computation(self, X, state=None, aim=None, **kwargs):
        """
        Version PyTorch pure optimisée (sans explosion mémoire O(T^2)).
        """
        # Gestion des états
        if state is not None and isinstance(state, tuple):
            ssm_state, conv_state = state
        else:
            ssm_state = None 
            conv_state = None

        X = self.norm(X)
        batch_size, seq_len, _ = X.shape

        # Reconstruct complex matrices
        Win_ = torch.complex(self.Win_real, self.Win_imag)
        Lambda = torch.complex(self.Lambda_real, self.Lambda_imag)
        Wout_ = torch.complex(self.Wout_real, self.Wout_imag)

        # 1. Projections
        X_expanded = X.view(batch_size, seq_len, 1, 1, self.input_dim)
        X_proj = torch.einsum('btxyi,mio->btmyo', X_expanded, self.proj)

        # Conv1D
        x_conv_in = X_proj.squeeze(3).view(batch_size, seq_len, self.units * self.input_dim)
        x_conv_in = x_conv_in.transpose(1, 2)

        if conv_state is not None:
            x_conv_full = torch.cat([conv_state, x_conv_in], dim=-1)
            x_conv_out = torch.nn.functional.conv1d(
                x_conv_full, 
                weight=self.conv1d.weight, 
                bias=self.conv1d.bias, 
                groups=self.conv1d.groups
            )
            new_conv_state = x_conv_full[..., -(self.d_conv - 1):]
        else:
            x_conv_out = self.conv1d(x_conv_in)[..., :seq_len]
            new_conv_state = x_conv_in[..., -(self.d_conv - 1):]

        x_conv_out = x_conv_out.transpose(1, 2).view(batch_size, seq_len, self.units, 1, self.input_dim)
        X_conv = torch.nn.functional.silu(x_conv_out)

        # 2. Calcul du Leak Rate (Adaptive)
        lr_proj = torch.einsum('btmxd,mdy->bmyt', X_conv, self.adaptive_lr)
        lr_logits = lr_proj + self.lr_bias.view(1, self.units, 1, 1)
        lr = torch.sigmoid(lr_logits).contiguous() # [B, M, 1, T]

        if aim is not None:
            for t in range(seq_len):
                for m in range(self.units):
                    aim.track(lr[0, m, 0, t], name=f'leak_rate_{m+1}')

        # 3. Préparation des variables pour la récurrence temporelle
        u_pre = torch.einsum('btmxd,mdr->bmrt', X_conv.to(self.complex_dtype), Win_).contiguous()
        
        # Application du leak_rate sur Win_ (u_pre) et Lambda
        u_pre_t = u_pre * lr # [B, M, R, T]
        
        lam = Lambda.view(1, self.units, self.neurons, 1) # [1, M, R, 1]
        lam_t = lr * lam + (1 - lr) # [B, M, R, T]

        if ssm_state is None:
            ssm_state = torch.zeros(batch_size, self.units, self.neurons, dtype=self.complex_dtype, device=self.device)
        else:
            ssm_state = ssm_state.view(batch_size, self.units, self.neurons).to(self.complex_dtype)

        # 4. Boucle de récurrence 
        # On permute pour itérer rapidement sur l'axe du temps : [B, M, R, T] -> [T, B, M, R]
        lam_t_seq = lam_t.permute(3, 0, 1, 2)
        u_pre_t_seq = u_pre_t.permute(3, 0, 1, 2)
        
        h = ssm_state
        states = []
        
        for t in range(seq_len):
            h = lam_t_seq[t] * h + u_pre_t_seq[t]
            states.append(h)
            
        new_ssm_state = torch.stack(states, dim=1) # [B, T, M, R]

        # 5. Output Computation
        output = torch.einsum('btmr,mro->btmo', new_ssm_state, Wout_).real.to(self.dtype)
        output = torch.nn.functional.silu(output)
        
        if aim is not None:
            for t in range(seq_len):
                for m in range(self.units):
                    for d in range(min(self.output_dim, 4)):
                        aim.track(output[0, t, m, d], name=f'output_{m+1}_{d+1}', epoch=None)

        new_state = (new_ssm_state, new_conv_state) # [B, T, M, R], [B, M*D, K-1]

        return output, new_state

    def _forward_computation_triton(self, X, state=None, chunk_size=256, aim=None): # Chunk réduit pour Fused (meilleure occupation)
        """
        Version Fused & Memory Optimized.
        """ 
         # Gestion des états : Unpacking du tuple
        if state is not None and isinstance(state, tuple):
            ssm_state, conv_state = state
        else:
            ssm_state = None 
            conv_state = None

        # Apply RMSNorm to the input
        X = self.norm(X) # [B, T, D] (applied before projection)

        # Reconstruct complex matrices
        Win_ = torch.complex(self.Win_real, self.Win_imag)
        Lambda = torch.complex(self.Lambda_real, self.Lambda_imag)
        Wout_ = torch.complex(self.Wout_real, self.Wout_imag)

        # Extract constants from X
        batch_size = X.shape[0]
        seq_len = X.shape[1]
        
        # 1. Projections
        X = X.view(batch_size, seq_len, 1, 1, self.input_dim) # [B, T, 1, 1, D]
        X_proj = torch.einsum('btxyi,mio->btmyo', X, self.proj) # [B, T, M, 1, D]

        # Conv1D
        x_conv_in = X_proj.squeeze(3).view(batch_size, seq_len, self.units * self.input_dim) # [B, T, M*D]
        x_conv_in = x_conv_in.transpose(1, 2) # [B, M*D, T]

        if conv_state is not None:
            # Generation mode (T=1) : Add the 3 previous tokens before the new one
            x_conv_full = torch.cat([conv_state, x_conv_in], dim=-1) # [B, M*D, (K-1) + T]
            x_conv_out = torch.nn.functional.conv1d(
                x_conv_full, 
                weight=self.conv1d.weight, 
                bias=self.conv1d.bias, 
                groups=self.conv1d.groups
            )
            # Cache Update
            new_conv_state = x_conv_full[..., -(self.d_conv - 1):] # [B, M*D, K-1]
        else:
            # Training mode / First Token : Use the normal layer with its padding
            x_conv_out = self.conv1d(x_conv_in)[..., :seq_len] # [B, M*D, T]
            # Initialize the cache for the rest with the last (K-1) elements
            new_conv_state = x_conv_in[..., -(self.d_conv - 1):] # [B, M*D, K-1]

        x_conv_out = x_conv_out.transpose(1, 2).view(batch_size, seq_len, self.units, 1, self.input_dim)
        X_conv = torch.nn.functional.silu(x_conv_out)

        if ssm_state is None:
            ssm_state = torch.zeros(batch_size, self.units, self.neurons, dtype=self.complex_dtype, device=self.device) # [B, M, R]

        # 2. Calcul du Leak Rate
        lr_proj = torch.einsum('btmxd,mdy->bmyt', X_conv, self.adaptive_lr) # [B, T, M, 1, D] @ [M, D, 1] -> [B, T, M, 1, 1] -> [B, M, 1, T]
        lr_logits = lr_proj + self.lr_bias.view(1, self.units, 1, 1) # [B, M, 1, T]
        lr = torch.sigmoid(lr_logits).contiguous() # [B, M, 1, T]
        del lr_proj

        # 3. Calcul de U_pre
        u_pre = torch.einsum('btmxd,mdr->bmrt', X_conv.to(self.complex_dtype), Win_).contiguous() # [B, M, R, T]
        
        # 4. Préparation Lambda
        lam = Lambda.unsqueeze(0).expand(batch_size, -1, -1).contiguous() # [B, M, R]

        # 5. Préparation State
        h_init = ssm_state.contiguous() # [B, M, R]

        # 6. SSM Computation
        new_ssm_state = parallel_fused_ssm(u_pre, lr, lam, h_init, chunk_size=chunk_size) # [B, M, R, T]
        new_ssm_state = new_ssm_state.permute(0, 3, 1, 2) # [B, T, M, R]
        del u_pre, lr, lam, h_init

        # 7. Output Computation
        output = torch.einsum('btmr,mro->btmo', new_ssm_state, Wout_).real.to(self.dtype) # [B, T, M, D]
        output = torch.nn.functional.silu(output)
        new_state = (new_ssm_state, new_conv_state) # [B, T, M, R], [B, M*D, K-1]

        return output, new_state # [B, T, M, D], [B, T, M, R]

    def __repr__(self):
        txt = "MemoryLayer("
        txt += f"\n  (norm): RMSNorm{tuple(self.norm.weight.shape)},"
        txt += f"\n  (proj): Tensor{tuple(self.proj.shape)},"
        txt += f"\n  (adaptive_lr): Tensor{tuple(self.adaptive_lr.shape)},"
        txt += f"\n  (lr_bias): Tensor{tuple(self.lr_bias.shape)},"
        # txt += f"\n  (temperature): Tensor{tuple(self.temperature.shape)},"
        txt += f"\n  (Win_real): Tensor{tuple(self.Win_real.shape)},"
        txt += f"\n  (Win_imag): Tensor{tuple(self.Win_imag.shape)},"
        txt += f"\n  (Lambda_real): Tensor{tuple(self.Lambda_real.shape)},"
        txt += f"\n  (Lambda_imag): Tensor{tuple(self.Lambda_imag.shape)},"
        txt += f"\n  (Wout_real): Tensor{tuple(self.Wout_real.shape)},"
        txt += f"\n  (Wout_imag): Tensor{tuple(self.Wout_imag.shape)},"
        txt += "\n)"
        return txt


def _initialize_matrix(shape, connectivity, distribution='normal', dtype=torch.float32, device='cpu'):
    """
    Initialize a matrix with a given shape and connectivity.

    Args:
    - shape (tuple): Shape of the matrix.
    - connectivity (float): Connectivity of the matrix.
    - distribution (str): Distribution of the matrix values ('normal' or 'bernoulli').
    - kwargs: Additional arguments for the distribution.

    Returns:
    - torch.Tensor: Initialized matrix.
    """
    if distribution == 'normal':
        matrix = torch.tensor(np.random.normal(size=shape, loc=0, scale=1), device=device, dtype=dtype)
        mask = _fixed_bernoulli(shape, connectivity, device=device, dtype=dtype)
        return matrix * mask
    
    elif distribution == 'uniform':
        matrix = torch.rand(size=shape, device=device, dtype=dtype)
        mask = _fixed_bernoulli(shape, connectivity, device=device, dtype=dtype)
        return matrix * mask

    elif distribution == 'bernoulli':
        return torch.bernoulli(torch.full(shape, connectivity, device=device, dtype=dtype))
    
    elif distribution == 'fixed_bernoulli':
        return _fixed_bernoulli(shape, connectivity, device=device)
    
    else:
        raise ValueError("Unsupported distribution type")

def _get_spectral_radius(matrix):
    """
    Get the spectral radius of a matrix.

    Args:
    - matrix (torch.Tensor): The matrix to analyze.

    Returns:
    - float: The spectral radius of the matrix.
    """
    device = str(matrix.device)
    # Compute the eigenvalues
    if 'mps' in device:
        # MPS does not support eigvals
        eigenvalues = torch.linalg.eigvals(matrix.to('cpu').to(torch.float64))#.to(matrix.device) # So we temporarily move the matrix to the CPU
    else:
        eigenvalues = torch.linalg.eigvals(matrix.to(torch.float64)) 
    
    # Compute the maximum eigenvalue
    abs_eigenvalue = torch.sqrt(eigenvalues.real**2 + eigenvalues.imag**2) # Cuda does not support torch.abs on Complex Number
    spectral_radius = torch.max(abs_eigenvalue, dim=-1).values

    return spectral_radius.to(matrix.dtype).to(matrix.device)

def _fixed_bernoulli(shape, connectivity, device='cpu', dtype=torch.float32):
    """
    Generate a connectivity matrix with a given shape and connectivity.

    Args:
    - shape (tuple): Shape of the matrix (head, line, column) or (line, column).
    - connectivity (float): Connectivity of the matrix.

    Every column has the same number of ones. (This constraint allows sparse matrix multiplication)

    Returns:
    - torch.Tensor: Connectivity matrix (head, line, column).
    """

    # Check the connectivity
    if not 0 < connectivity <= 1:
        raise ValueError("Connectivity must be > 0 et <= 1")
    
    # If len(shape) == 2, add a dimension
    if len(shape) == 2:
        shape = (1, shape[0], shape[1])
    
    # Init matrix & nb connections
    nb_connections = max(1, int(connectivity * shape[-2])) # At least one connection
    matrix = torch.zeros(shape, device=device)
    
    # For each column, set the connections
    for head in range(shape[-3]):
        for col in range(shape[-1]):
            indices = torch.randperm(shape[-2])[:nb_connections]
            matrix[head, indices, col] = 1
    
    return matrix.to(dtype)

def _decompose_matrix(W):
    """
    Décompose la matrice W en valeurs propres et vecteurs propres.
    
    Args:
        W (torch.Tensor): Matrice à décomposer.
        
    Returns:
        Lambda (torch.Tensor): Matrice diagonale des valeurs propres.
        P (torch.Tensor): Matrice des vecteurs propres.
        P_inv (torch.Tensor): Inverse de la matrice des vecteurs propres.
    """
    # Construction de la décomposition en valeurs propres
    eigenvalues, eigenvectors = torch.linalg.eig(W.to(torch.float64)) # High precision for stability
    Lambda = torch.diag(eigenvalues)
    P = eigenvectors
    P_inv = torch.linalg.inv(P)

    return Lambda, P, P_inv

def _compute_Lambda_compact(Lambda):
    """
    Compute the compact representation of Lambda for efficient computation.

    Args:
        Lambda: Tensor of shape (B, T, M, R) representing eigenvalues (R) combined with adaptive leak rates for each time step.

    Returns:
        Lambda_compact: Tensor of shape (B, (T^2+T)/2, M, R) representing the compacted version of Lambda.
    """
    B, T, M, R = Lambda.shape
    Lambda_compact = torch.zeros((B, (T**2 + T) // 2 + T, M, R), dtype=Lambda.dtype, device=Lambda.device)

    # Fill the compact representation
    idx = 0
    for t in range(1, T+1):
        Lambda_compact[:, idx:idx + t] = torch.cumprod(Lambda[:, :t].flip(1), dim=1).flip(1)
        Lambda_compact[:, idx + t] = 1  # Identity for the product when k=t
        idx += t + 1

    return Lambda_compact

def _compute_feed_compact(feed):
    """
    Apply the compacted Lambda to the feed tensor.

    Args:
        feed: Tensor of shape (B, T+1, M, R) representing the feed inputs.

    Returns:
        new_state: Tensor of shape (B, T, M, R) representing the new states after applying Lambda.
    """
    B, T_plus_1, M, R = feed.shape
    T = T_plus_1 - 1
    feed_compact = torch.zeros((B, (T**2 + T) // 2 + T, M, R), dtype=feed.dtype, device=feed.device)

    idx = 0
    for t in range(1, T+1):
        feed_compact[:, idx:idx + t+1] = feed[:, :t+1]
        idx += t + 1

    return feed_compact

def _compute_new_state(echos, T):
    """
    Compute the new state from the echos tensor.
    Args:
        echos: Tensor of shape (B, (T^2+T)/2, M, R) representing the echos.
    Returns:
        new_state: Tensor of shape (B, T, M, R) representing the new states.
    """
    B, _, M, R = echos.shape
    new_state = torch.zeros((B, T, M, R), dtype=echos.dtype, device=echos.device)

    idx = 0
    for t in range(T):
        new_state[:, t] = torch.sum(echos[:, idx:idx + t + 2], dim=1)
        idx += t + 2

    return new_state


def _generate_eigenvalues(neurons, units, r_min=0.95, r_max=0.999):
    for _ in range(units):
        u1 = torch.rand((units, neurons)) # Uniforme [0, 1]
        r = r_min + (r_max - r_min) * u1
        theta = torch.rand((units, neurons)) * 2 * np.pi
    lambda_real = r * torch.cos(theta)
    lambda_imag = r * torch.sin(theta)
    return lambda_real, lambda_imag

def _compute_loss_aux(lr, alpha=0.1, beta=0.1):
    """Compute an auxiliary loss to encourage leak rates to be well distributed between units.
    
    Args:
        lr (torch.Tensor): Leak rates of shape (B, T, M).
        alpha (float): Weight for the sparsity loss.
        beta (float): Weight for the diversity loss.
    """
    target_sparsity = 1.0 / lr.shape[2]
    mean_lr = torch.mean(lr, dim=(0,1))  # Mean over batch and time
    sparsity_loss = torch.mean((mean_lr - target_sparsity) ** 2)

    diversity_loss = torch.mean(torch.var(lr, dim=(0, 1)))  # Variance over batch

    return alpha * sparsity_loss - beta * diversity_loss

