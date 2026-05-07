import torch
import triton
import triton.language as tl

# -------------------------------------------------------------------------
# OPERATEUR ASSOCIATIF (CORRIGÉ)
# -------------------------------------------------------------------------

@triton.jit
def first_order_op(a_left, u_left, a_right, u_right):
    """
    Opération binaire associative pour le SSM : (a_L, u_L) • (a_R, u_R)
    
    Math:
    h_total = A_R * h_L + u_R
    A_total = A_R * A_L
    
    Args:
        a_left, u_left : Paramètres du bloc "tôt" (indices petits)
        a_right, u_right : Paramètres du bloc "tard" (indices grands)
    """
    # A_new = A_Right * A_Left
    a_new = a_right * a_left
    
    # U_new = A_Right * U_Left + U_Right
    u_new = a_right * u_left + u_right
    
    return a_new, u_new

# -------------------------------------------------------------------------
# KERNEL FORWARD (PARALLEL SCAN)
# -------------------------------------------------------------------------

@triton.jit
def parallel_scan_fwd_kernel(
    H_ptr, U_ptr, A_ptr,
    stride_row, stride_time, # stride_row = saut pour changer de (Batch,Dim), stride_time = 1
    n_steps,
    BLOCK_SIZE_TIME: tl.constexpr
):
    pid = tl.program_id(0) # Index aplati (Batch * Dim)
    
    # Pointeurs de base pour la séquence courante
    u_ptr = U_ptr + pid * stride_row
    a_ptr = A_ptr + pid * stride_row
    h_ptr = H_ptr + pid * stride_row

    # Chargement des données
    # On charge toute la timeline d'un coup (jusqu'à BLOCK_SIZE)
    t_offs = tl.arange(0, BLOCK_SIZE_TIME)
    mask = t_offs < n_steps

    # Initialisation neutre: U=0, A=1 (identité multiplicative)
    # Si on dépasse n_steps, on pad avec l'identité pour ne pas fausser le scan
    u_vals = tl.load(u_ptr + t_offs * stride_time, mask=mask, other=0.0)
    a_vals = tl.load(a_ptr + t_offs * stride_time, mask=mask, other=1.0)

    # --- LE SCAN PARALLELE ---
    # Triton applique first_order_op de manière arborescente
    res_a, res_u = tl.associative_scan((a_vals, u_vals), 0, first_order_op)

    # Sauvegarde
    tl.store(h_ptr + t_offs * stride_time, res_u, mask=mask)


# -------------------------------------------------------------------------
# KERNEL BACKWARD (REVERSE SCAN CORRIGÉ)
# -------------------------------------------------------------------------

@triton.jit
def parallel_scan_bwd_kernel(
    DH_out_ptr, H_ptr, A_ptr,      # Inputs
    DU_ptr, DA_ptr,                # Outputs
    stride_row, stride_time,
    n_steps,
    BLOCK_SIZE_TIME: tl.constexpr
):
    pid = tl.program_id(0)
    
    # Pointeurs de base
    offset = pid * stride_row
    dh_out_ptr = DH_out_ptr + offset
    h_ptr = H_ptr + offset
    a_ptr = A_ptr + offset
    du_ptr = DU_ptr + offset
    da_ptr = DA_ptr + offset

    # --- 1. PRÉPARATION DU REVERSE SCAN ---
    # Le backward est un scan inversé (de T-1 vers 0).
    # d_curr = dL/dh_curr + A_{curr+1} * d_{curr+1}
    # Structure : Y = U + A * Y_prev (mais temps inversé)
    # Input Scan "U" = dL/dh (le gradient entrant)
    # Input Scan "A" = A (mais décalé de +1 vers la gauche)
    
    t_offs = tl.arange(0, BLOCK_SIZE_TIME)
    
    # Inversion des indices pour lire depuis la fin
    # rev_idx[0] correspond à T-1, rev_idx[1] à T-2...
    rev_idx = (n_steps - 1) - t_offs
    mask_rev = (rev_idx >= 0) & (rev_idx < n_steps)

    # Chargement du gradient dL/dh dans l'ordre inverse
    dh_vals = tl.load(dh_out_ptr + rev_idx * stride_time, mask=mask_rev, other=0.0)
    
    # Chargement de A pour le backward
    # Attention: le gradient à t dépend de A_{t+1}.
    # Donc pour le pas de scan inversé k (qui correspond au temps t), on veut A_{t+1}.
    # Indice t = rev_idx. On veut A à rev_idx + 1.
    a_idx_shift = rev_idx + 1
    mask_a = (a_idx_shift > 0) & (a_idx_shift < n_steps) # A[T] est hors bornes (0)
    
    # A_vals[k] contient A_{t+1}
    a_vals_shifted = tl.load(a_ptr + a_idx_shift * stride_time, mask=mask_a, other=0.0)

    # --- 2. REVERSE SCAN ---
    # On applique exactement la même logique associative : (A_R * U_L + U_R)
    # Ici "Gauche" dans le scan signifie "Futur" (car on a renversé le temps)
    # et "Droite" signifie "Passé immédiat".
    # Donc on accumule l'effet du futur vers le présent.
    _, dh_total_rev = tl.associative_scan((a_vals_shifted, dh_vals), 0, first_order_op)

    # dh_total_rev est maintenant dans l'ordre inverse (index 0 = temps T-1)
    
    # --- 3. CALCUL DES GRADIENTS FINAUX (DU, DA) ---
    
    # dL/du_t = dL/dh_total_t
    # On stocke directement dU (en utilisant les indices inversés pour remettre à l'endroit)
    tl.store(du_ptr + rev_idx * stride_time, dh_total_rev, mask=mask_rev)
    
    # dL/dA_t = dL/dh_t * h_{t-1}
    # On a besoin de h_{t-1}. 
    # Pour le temps t = rev_idx, on veut h à rev_idx - 1.
    h_idx_prev = rev_idx - 1
    mask_h = (h_idx_prev >= 0) & (h_idx_prev < n_steps)
    h_prev = tl.load(h_ptr + h_idx_prev * stride_time, mask=mask_h, other=0.0)
    
    da_vals = dh_total_rev * h_prev
    tl.store(da_ptr + rev_idx * stride_time, da_vals, mask=mask_rev)


# -------------------------------------------------------------------------
# WRAPPER PYTHON
# -------------------------------------------------------------------------

class SSMParaFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, a):
        # u: [Batch, Time, Dim] -> On veut scan sur Time
        batch, steps, dim = u.shape
        BLOCK_TIME = triton.next_power_of_2(steps)
        
        # Transposition critique : [Batch, Dim, Time]
        # Cela rend l'axe temporel contigu en dernier (stride=1)
        # Et fusionne (Batch, Dim) en une seule grille de calcul
        u_t = u.permute(0, 2, 1).contiguous()
        a_t = a.permute(0, 2, 1).contiguous()
        h_t = torch.empty_like(u_t)
        
        # Grid: Chaque threadblock traite une séquence (Batch * Dim)
        grid = (batch * dim, )
        
        # Stride Row: combien d'éléments sauter pour passer à la séquence suivante ?
        # Comme c'est contigu [B*D, T], le saut est 'steps' (si T est aligné)
        # Mais triton gère les pointeurs. u_t.stride(0) dans la vue [B*D, T] donne 'steps'.
        # On passe explicitement le stride de la dimension 0 de la vue fusionnée.
        stride_row = u_t.stride(0) * u_t.stride(1) if len(u_t.stride()) > 2 else u_t.stride(0)
        # En fait plus simple: u_t est [Batch, Dim, Time]. 
        # Si on le voit comme [Batch*Dim, Time], le stride_row est le stride de Dim (qui est Time).
        stride_row = steps 
        
        parallel_scan_fwd_kernel[grid](
            h_t, u_t, a_t,
            stride_row, 1, # Stride Time est 1 car contiguous
            steps,
            BLOCK_SIZE_TIME=BLOCK_TIME
        )
        
        # Retour au format original
        h = h_t.permute(0, 2, 1)
        ctx.save_for_backward(u, a, h)
        return h

    @staticmethod
    def backward(ctx, dh_out):
        u, a, h = ctx.saved_tensors
        batch, steps, dim = u.shape
        BLOCK_TIME = triton.next_power_of_2(steps)
        
        # Transposition et contiguous
        dh_out_t = dh_out.permute(0, 2, 1).contiguous()
        h_t = h.permute(0, 2, 1).contiguous()
        a_t = a.permute(0, 2, 1).contiguous()
        
        du_t = torch.empty_like(dh_out_t)
        da_t = torch.empty_like(dh_out_t)
        
        grid = (batch * dim, )
        stride_row = steps
        
        parallel_scan_bwd_kernel[grid](
            dh_out_t, h_t, a_t,
            du_t, da_t,
            stride_row, 1,
            steps,
            BLOCK_SIZE_TIME=BLOCK_TIME
        )
        
        return du_t.permute(0, 2, 1), da_t.permute(0, 2, 1)

def custom_ssm_parallel(X, A, B, C):
    u = torch.einsum('bti,btin->btn', X, B)
    h = SSMParaFunction.apply(u, A)
    y = torch.matmul(h, C)
    return y