import torch
import triton
import triton.language as tl

# -------------------------------------------------------------------------
# KERNEL FORWARD
# Calcul: h_t = A_t * h_{t-1} + u_t
# Avec u_t = (X_t @ B_t) pré-calculé avant d'entrer dans le kernel
# -------------------------------------------------------------------------

@triton.jit
def ssm_scan_fwd_kernel(
    # Pointeurs
    H_ptr,      # Output state [Batch, Time, State]
    U_ptr,      # Input (X@B) [Batch, Time, State]
    A_ptr,      # Decay/Transition [Batch, Time, State]
    # Strides (pas mémoire)
    stride_b_batch, stride_b_time, stride_b_state,
    stride_h_batch, stride_h_time, stride_h_state,
    stride_a_batch, stride_a_time, stride_a_state,
    # Dimensions
    n_steps,    # Longueur de la séquence (Timestep)
    BLOCK_SIZE_N: tl.constexpr # Dimension de l'état (doit être puissance de 2 pour la performance)
):
    # On parallélise sur l'axe Batch. 
    # Chaque bloc CUDA s'occupe d'une séquence complète pour un élément du batch.
    pid = tl.program_id(0)
    
    # Offsets initiaux pour ce batch
    off_state = tl.arange(0, BLOCK_SIZE_N)
    
    # Pointeurs vers le début des données pour ce batch spécifique
    u_batch_ptr = U_ptr + (pid * stride_b_batch)
    a_batch_ptr = A_ptr + (pid * stride_a_batch)
    h_batch_ptr = H_ptr + (pid * stride_h_batch)

    # Initialisation de l'état caché h_{-1} à 0
    h_curr = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)

    # Boucle temporelle (Scan)
    # C'est ici que Triton brille : h_curr reste en registres/SRAM
    for t in range(n_steps):
        # Chargement de u_t et A_t
        # On utilise les strides temporelles pour avancer
        offs_u = t * stride_b_time + off_state
        offs_a = t * stride_a_time + off_state
        
        # Masque non nécessaire si BLOCK_SIZE_N >= state_dim et padding fait en amont,
        # mais on assume ici que les dimensions matchent le block size ou padding.
        u_t = tl.load(u_batch_ptr + offs_u)
        a_t = tl.load(a_batch_ptr + offs_a)

        # La récurrence : h_t = A_t * h_{t-1} + u_t
        h_curr = a_t * h_curr + u_t

        # Sauvegarde de h_t en mémoire globale pour la backprop plus tard
        offs_h = t * stride_h_time + off_state
        tl.store(h_batch_ptr + offs_h, h_curr)


# -------------------------------------------------------------------------
# KERNEL BACKWARD
# Calcul des gradients dH, dA, dU en remontant le temps
# -------------------------------------------------------------------------

@triton.jit
def ssm_scan_bwd_kernel(
    # Inputs (Gradients entrants et valeurs forward)
    D_H_out_ptr, # Gradient venant de la perte (via Y) [Batch, Time, State]
    H_ptr,       # Valeurs de h calculées au forward [Batch, Time, State]
    A_ptr,       # Valeurs de A [Batch, Time, State]
    
    # Outputs (Gradients à calculer)
    D_U_ptr,     # Gradient pour l'input u [Batch, Time, State]
    D_A_ptr,     # Gradient pour A [Batch, Time, State]
    
    # Strides
    stride_dh_batch, stride_dh_time, stride_dh_state,
    stride_h_batch, stride_h_time, stride_h_state,
    stride_a_batch, stride_a_time, stride_a_state,
    stride_du_batch, stride_du_time, stride_du_state,
    stride_da_batch, stride_da_time, stride_da_state,
    
    n_steps,
    BLOCK_SIZE_N: tl.constexpr
):
    pid = tl.program_id(0)
    off_state = tl.arange(0, BLOCK_SIZE_N)

    # Pointeurs de base pour ce batch
    dh_out_batch_ptr = D_H_out_ptr + (pid * stride_dh_batch)
    h_batch_ptr = H_ptr + (pid * stride_h_batch)
    a_batch_ptr = A_ptr + (pid * stride_a_batch)
    du_batch_ptr = D_U_ptr + (pid * stride_du_batch)
    da_batch_ptr = D_A_ptr + (pid * stride_da_batch)

    # Accumulateur pour le gradient qui remonte le temps (dh_{t+1} * A_{t+1})
    d_h_next = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)

    # On itère à l'envers : de T-1 à 0
    for t in range(n_steps - 1, -1, -1):
        # Offsets
        offs_time_curr = t * stride_h_time
        offs_state_idx = off_state
        
        # 1. Charger le gradient venant de la sortie Y à cet instant t
        # dL/dh_t_output
        d_h_curr_out = tl.load(dh_out_batch_ptr + offs_time_curr + offs_state_idx)
        
        # 2. Charger A_t pour le calcul local et H_{t-1} pour le gradient de A
        a_t = tl.load(a_batch_ptr + t * stride_a_time + offs_state_idx)
        
        # Cas spécial pour h_{t-1} : si t=0, h_{-1} est 0
        h_prev = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)
        if t > 0:
            h_prev = tl.load(h_batch_ptr + (t - 1) * stride_h_time + offs_state_idx)

        # 3. Gradient total sur h_t
        # dL/dh_t = (dL/dy_t * C) + (dL/dh_{t+1} * A_{t+1})
        # d_h_curr_out contient déjà (dL/dy_t * C) calculé en Python
        # d_h_next contient (dL/dh_{t+1} * A_{t+1}) accumulé au tour précédent
        d_h_total = d_h_curr_out + d_h_next

        # 4. Gradients par rapport aux paramètres
        # h_t = A_t * h_{t-1} + u_t
        
        # dL/du_t = dL/dh_t * 1
        tl.store(du_batch_ptr + t * stride_du_time + offs_state_idx, d_h_total)
        
        # dL/dA_t = dL/dh_t * h_{t-1}
        d_a_t = d_h_total * h_prev
        tl.store(da_batch_ptr + t * stride_da_time + offs_state_idx, d_a_t)

        # 5. Préparer le gradient pour l'étape précédente (t-1)
        # La contribution de ce pas de temps au précédent est : dL/dh_t * A_t
        d_h_next = d_h_total * a_t


# -------------------------------------------------------------------------
# WRAPPER AUTOGRAD PYTORCH
# -------------------------------------------------------------------------

class SSMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, a):
        # u: [Batch, Time, State] (résultat de X @ B)
        # a: [Batch, Time, State]
        
        # Vérifications de forme
        batch_size, n_steps, state_dim = u.shape
        assert a.shape == u.shape, "Dimensions mismatch between U and A"
        
        # Pour simplifier, on prend la puissance de 2 supérieure
        BLOCK_SIZE_N = triton.next_power_of_2(state_dim)
        
        # Allocation sortie
        h = torch.empty_like(u)
        
        # Lancement du Kernel Triton
        # Grid : autant de programmes que de séquences dans le batch
        grid = (batch_size, )
        
        ssm_scan_fwd_kernel[grid](
            h, u, a,
            u.stride(0), u.stride(1), u.stride(2),
            h.stride(0), h.stride(1), h.stride(2),
            a.stride(0), a.stride(1), a.stride(2),
            n_steps,
            BLOCK_SIZE_N=BLOCK_SIZE_N
        )
        
        ctx.save_for_backward(u, a, h)
        return h

    @staticmethod
    def backward(ctx, dh_out):
        # dh_out: gradient venant de la projection suivante (Y = hC)
        # C'est dL/dh (partie output)
        
        u, a, h = ctx.saved_tensors
        batch_size, n_steps, state_dim = u.shape
        BLOCK_SIZE_N = triton.next_power_of_2(state_dim)
        
        du = torch.empty_like(u)
        da = torch.empty_like(a)
        
        # Lancement du Kernel Backward
        grid = (batch_size, )
        
        ssm_scan_bwd_kernel[grid](
            dh_out, h, a,
            du, da,
            dh_out.stride(0), dh_out.stride(1), dh_out.stride(2),
            h.stride(0), h.stride(1), h.stride(2),
            a.stride(0), a.stride(1), a.stride(2),
            du.stride(0), du.stride(1), du.stride(2),
            da.stride(0), da.stride(1), da.stride(2),
            n_steps,
            BLOCK_SIZE_N=BLOCK_SIZE_N
        )
        
        return du, da

# -------------------------------------------------------------------------
# MODULE PRINCIPAL
# -------------------------------------------------------------------------

def custom_ssm_forward(X, A, B, C):
    """
    Entrée :
      X : [Batch, Timestep, Input Dim]
      A : [Batch, Timestep, State Dim]
      B : [Batch, Timestep, Input Dim, State Dim]
      C : [State Dim, Output Dim]
    
    Sortie :
      Y : [Batch, Timestep, Output Dim]
    """
    # 1. Préparation de l'entrée U = X * B
    # B varie avec le temps, donc c'est un batch matmul ou einsum
    # X: (b, t, i), B: (b, t, i, n) -> U: (b, t, n)
    u = torch.einsum('bti,btin->btn', X, B)
    
    # S'assurer que les tenseurs sont contigus pour Triton
    u = u.contiguous()
    A = A.contiguous()
    
    # 2. Le Scan SSM via Triton
    h = SSMFunction.apply(u, A)
    
    # 3. Projection de sortie Y = h * C
    # h: (b, t, n), C: (n, o) -> Y: (b, t, o)
    y = torch.matmul(h, C)
    
    return y