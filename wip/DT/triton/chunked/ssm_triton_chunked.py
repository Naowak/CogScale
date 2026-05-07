import torch
import triton
import triton.language as tl

# =========================================================================
# 1. TRITON KERNELS (Clean & Robust)
# =========================================================================

@triton.jit
def first_order_op(a_left, u_left, a_right, u_right):
    """
    Opérateur associatif (Scan Forward).
    Combine (a_L, u_L) avec (a_R, u_R) => (a_L * a_R, a_R * u_L + u_R).
    """
    a_new = a_right * a_left
    u_new = a_right * u_left + u_right
    return a_new, u_new

@triton.jit
def first_order_op_bwd(a_left, u_left, a_right, u_right):
    """
    Opérateur associatif (Scan Backward).
    """
    a_new = a_right * a_left
    u_new = a_right * u_left + u_right
    return a_new, u_new

# --- FORWARD KERNELS ---

@triton.jit
def chunk_scan_fwd_kernel(
    U_ptr, A_ptr,               # Inputs
    H_intra_ptr, A_intra_ptr,   # Output Partial
    L_u_ptr, L_a_ptr,           # Output Block Summaries
    stride_batch, stride_dim, stride_seq,
    seq_len, chunk_size: tl.constexpr
):
    pid_batch = tl.program_id(0)
    pid_dim   = tl.program_id(1)
    pid_chunk = tl.program_id(2)

    offs_base = pid_batch * stride_batch + pid_dim * stride_dim
    offs_chunk = pid_chunk * chunk_size
    t_offs = tl.arange(0, chunk_size)
    global_idx = offs_chunk + t_offs

    mask = global_idx < seq_len

    u = tl.load(U_ptr + offs_base + global_idx * stride_seq, mask=mask, other=0.0)
    a = tl.load(A_ptr + offs_base + global_idx * stride_seq, mask=mask, other=1.0)

    scan_a, scan_u = tl.associative_scan((a, u), 0, first_order_op)

    tl.store(H_intra_ptr + offs_base + global_idx * stride_seq, scan_u, mask=mask)
    tl.store(A_intra_ptr + offs_base + global_idx * stride_seq, scan_a, mask=mask)
    
    last_idx = chunk_size - 1
    block_u = tl.sum(tl.where(t_offs == last_idx, scan_u, 0.0), axis=0)
    block_a = tl.sum(tl.where(t_offs == last_idx, scan_a, 0.0), axis=0)
    
    offs_l = pid_batch * (tl.num_programs(1) * tl.num_programs(2)) + \
             pid_dim * tl.num_programs(2) + pid_chunk
    
    tl.store(L_u_ptr + offs_l, block_u)
    tl.store(L_a_ptr + offs_l, block_a)

@triton.jit
def chunk_update_fwd_kernel(
    H_intra_ptr, A_intra_ptr, 
    Global_H_ptr,             
    H_final_ptr,              
    stride_batch, stride_dim, stride_seq,
    seq_len, chunk_size: tl.constexpr
):
    pid_batch = tl.program_id(0)
    pid_dim   = tl.program_id(1)
    pid_chunk = tl.program_id(2)

    offs_base = pid_batch * stride_batch + pid_dim * stride_dim
    offs_chunk = pid_chunk * chunk_size
    t_offs = tl.arange(0, chunk_size)
    global_idx = offs_chunk + t_offs
    mask = global_idx < seq_len

    offs_carry = pid_batch * (tl.num_programs(1) * tl.num_programs(2)) + \
                 pid_dim * tl.num_programs(2) + pid_chunk
    carry = tl.load(Global_H_ptr + offs_carry)

    a = tl.load(A_intra_ptr + offs_base + global_idx * stride_seq, mask=mask, other=0.0) 
    h_intra = tl.load(H_intra_ptr + offs_base + global_idx * stride_seq, mask=mask, other=0.0)

    h_final = h_intra + carry * a

    tl.store(H_final_ptr + offs_base + global_idx * stride_seq, h_final, mask=mask)


# --- BACKWARD KERNELS ---

@triton.jit
def chunk_scan_bwd_kernel(
    D_out_ptr, A_ptr,           
    Du_intra_ptr, A_rev_ptr,    
    L_du_ptr, L_a_ptr,          
    stride_batch, stride_dim, stride_seq,
    seq_len, chunk_size: tl.constexpr
):
    pid_batch = tl.program_id(0)
    pid_dim   = tl.program_id(1)
    pid_chunk = tl.program_id(2)

    offs_base = pid_batch * stride_batch + pid_dim * stride_dim
    offs_chunk = pid_chunk * chunk_size
    
    t_offs = tl.arange(0, chunk_size)
    rev_t_offs = (chunk_size - 1) - t_offs
    global_idx = offs_chunk + rev_t_offs
    mask = global_idx < seq_len
    
    d_out = tl.load(D_out_ptr + offs_base + global_idx * stride_seq, mask=mask, other=0.0)
    
    idx_a = global_idx + 1
    mask_a = idx_a < seq_len 
    a_shifted = tl.load(A_ptr + offs_base + idx_a * stride_seq, mask=mask_a, other=0.0)
    
    scan_a_rev, scan_du = tl.associative_scan((a_shifted, d_out), 0, first_order_op_bwd)
    
    tl.store(Du_intra_ptr + offs_base + global_idx * stride_seq, scan_du, mask=mask)
    tl.store(A_rev_ptr + offs_base + global_idx * stride_seq, scan_a_rev, mask=mask)
    
    last_idx = chunk_size - 1
    block_du = tl.sum(tl.where(t_offs == last_idx, scan_du, 0.0), axis=0)
    block_a = tl.sum(tl.where(t_offs == last_idx, scan_a_rev, 0.0), axis=0)
    
    offs_l = pid_batch * (tl.num_programs(1) * tl.num_programs(2)) + \
             pid_dim * tl.num_programs(2) + pid_chunk
             
    tl.store(L_du_ptr + offs_l, block_du)
    tl.store(L_a_ptr + offs_l, block_a)


@triton.jit
def chunk_final_bwd_kernel(
    Du_intra_ptr, A_rev_ptr, H_ptr, H_init_ptr, # Inputs
    Global_Grad_ptr,                            # Input (Carries)
    Du_final_ptr, Da_final_ptr,                 # Outputs
    stride_batch, stride_dim, stride_seq,
    seq_len, chunk_size: tl.constexpr,
    DIM: tl.constexpr,              # FIX: On passe DIM pour calculer l'offset H_init
    HAS_INIT: tl.constexpr
):
    pid_batch = tl.program_id(0)
    pid_dim   = tl.program_id(1)
    pid_chunk = tl.program_id(2)

    offs_base = pid_batch * stride_batch + pid_dim * stride_dim
    offs_chunk = pid_chunk * chunk_size
    t_offs = tl.arange(0, chunk_size)
    global_idx = offs_chunk + t_offs
    mask = global_idx < seq_len
    
    # 1. Carry
    offs_carry = pid_batch * (tl.num_programs(1) * tl.num_programs(2)) + \
                 pid_dim * tl.num_programs(2) + pid_chunk
    carry = tl.load(Global_Grad_ptr + offs_carry)
    
    # 2. Load Partials
    du_intra = tl.load(Du_intra_ptr + offs_base + global_idx * stride_seq, mask=mask, other=0.0)
    a_rev = tl.load(A_rev_ptr + offs_base + global_idx * stride_seq, mask=mask, other=0.0)
    
    # 3. Patching
    du_final = du_intra + carry * a_rev
    tl.store(Du_final_ptr + offs_base + global_idx * stride_seq, du_final, mask=mask)
    
    # 4. Calcul de dA
    idx_prev = global_idx - 1
    
    h_prev = tl.zeros([chunk_size], dtype=tl.float32) 
    
    # H[t-1] standard
    mask_h = (idx_prev >= 0) & (idx_prev < seq_len)
    val_h = tl.load(H_ptr + offs_base + idx_prev * stride_seq, mask=mask_h, other=0.0)
    h_prev = tl.where(idx_prev >= 0, val_h, h_prev)
    
    # H_init pour t=0
    if HAS_INIT:
        # FIX: Ne pas utiliser stride_batch (qui est pour 3D tensor) pour H_init (2D tensor)
        # H_init est [B, D] contigu.
        # Offset = pid_batch * DIM + pid_dim
        offs_init = pid_batch * DIM + pid_dim
        val_init = tl.load(H_init_ptr + offs_init)
        h_prev = tl.where(global_idx == 0, val_init, h_prev)
        
    da = du_final * h_prev
    tl.store(Da_final_ptr + offs_base + global_idx * stride_seq, da, mask=mask)


# =========================================================================
# 2. WRAPPER AUTOGRAD (PyTorch Glue)
# =========================================================================

class SSMParaFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, a, h_init, chunk_size):
        u = u.contiguous()
        a = a.contiguous()
        if h_init is not None: h_init = h_init.contiguous()
        
        B, D, L = u.shape
        num_chunks = (L + chunk_size - 1) // chunk_size
        
        h_intra = torch.empty_like(u)
        a_intra = torch.empty_like(a)
        L_u = torch.empty((B, D, num_chunks), device=u.device, dtype=u.dtype)
        L_a = torch.empty((B, D, num_chunks), device=a.device, dtype=a.dtype)
        
        grid = (B, D, num_chunks)
        
        chunk_scan_fwd_kernel[grid](
            u, a, 
            h_intra, a_intra, 
            L_u, L_a,
            u.stride(0), u.stride(1), u.stride(2),
            L, chunk_size=chunk_size
        )
        
        carries = torch.empty_like(L_u)
        curr = h_init.clone() if h_init is not None else torch.zeros((B, D), device=u.device, dtype=u.dtype)
        
        for i in range(num_chunks):
            carries[:,:,i] = curr
            curr = L_a[:,:,i] * curr + L_u[:,:,i]
            
        h_final = torch.empty_like(u)
        chunk_update_fwd_kernel[grid](
            h_intra, a_intra, carries, h_final,
            u.stride(0), u.stride(1), u.stride(2),
            L, chunk_size=chunk_size
        )
        
        ctx.save_for_backward(u, a, h_init, h_final)
        ctx.chunk_size = chunk_size
        ctx.has_init = (h_init is not None)
        return h_final

    @staticmethod
    def backward(ctx, grad_output):
        u, a, h_init, h_final = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        
        B, D, L = u.shape
        chunk_size = ctx.chunk_size
        num_chunks = (L + chunk_size - 1) // chunk_size
        grid = (B, D, num_chunks)
        
        du_intra = torch.empty_like(u)
        a_rev = torch.empty_like(a)
        
        L_du = torch.empty((B, D, num_chunks), device=u.device, dtype=u.dtype)
        L_a_rev = torch.empty((B, D, num_chunks), device=u.device, dtype=u.dtype)
        
        chunk_scan_bwd_kernel[grid](
            grad_output, a,
            du_intra, a_rev,
            L_du, L_a_rev,
            grad_output.stride(0), grad_output.stride(1), grad_output.stride(2),
            L, chunk_size=chunk_size
        )
        
        grad_carries = torch.zeros((B, D, num_chunks), device=u.device, dtype=u.dtype)
        curr_g = torch.zeros((B, D), device=u.device, dtype=u.dtype)
        
        for i in range(num_chunks - 1, -1, -1):
            grad_carries[:,:,i] = curr_g
            curr_g = curr_g * L_a_rev[:,:,i] + L_du[:,:,i]
            
        d_h_init = None
        if ctx.has_init:
            d_h_init = curr_g * a[:,:,0]
            
        du_final = torch.empty_like(u)
        da_final = torch.empty_like(a)
        
        h_init_ptr = h_init if ctx.has_init else u 
        
        chunk_final_bwd_kernel[grid](
            du_intra, a_rev, h_final, h_init_ptr,
            grad_carries,
            du_final, da_final,
            u.stride(0), u.stride(1), u.stride(2),
            L, chunk_size=chunk_size,
            DIM=D, # FIX: Passage de DIM
            HAS_INIT=ctx.has_init
        )
        
        return du_final, da_final, d_h_init, None

def parallel_ssm(u, a, h_init=None, chunk_size=1024):
    return SSMParaFunction.apply(u, a, h_init, chunk_size)