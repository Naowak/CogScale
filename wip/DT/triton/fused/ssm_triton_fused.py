import torch
import triton
import triton.language as tl

# =========================================================================
# 1. HELPERS COMPLEXES
# =========================================================================

@triton.jit
def complex_mul(r_a, i_a, r_b, i_b):
    r_out = r_a * r_b - i_a * i_b
    i_out = r_a * i_b + i_a * r_b
    return r_out, i_out

@triton.jit
def complex_mul_conj(r_a, i_a, r_b, i_b):
    # A * conj(B)
    r_out = r_a * r_b + i_a * i_b
    i_out = i_a * r_b - r_a * i_b
    return r_out, i_out

@triton.jit
def fused_op(ar_l, ai_l, ur_l, ui_l, ar_r, ai_r, ur_r, ui_r):
    ar_new, ai_new = complex_mul(ar_r, ai_r, ar_l, ai_l)
    tr, ti = complex_mul(ar_r, ai_r, ur_l, ui_l)
    ur_new = tr + ur_r
    ui_new = ti + ui_r
    return ar_new, ai_new, ur_new, ui_new

@triton.jit
def fused_op_bwd(ar_l, ai_l, ur_l, ui_l, ar_r, ai_r, ur_r, ui_r):
    ar_new, ai_new = complex_mul(ar_r, ai_r, ar_l, ai_l)
    tr, ti = complex_mul(ar_r, ai_r, ur_l, ui_l)
    ur_new = tr + ur_r
    ui_new = ti + ui_r
    return ar_new, ai_new, ur_new, ui_new

# =========================================================================
# 2. FUSED FORWARD KERNEL
# =========================================================================

@triton.jit
def fused_chunk_scan_fwd_kernel(
    U_pre_r_ptr, U_pre_i_ptr, LR_ptr, Lam_r_ptr, Lam_i_ptr,
    H_intra_r_ptr, H_intra_i_ptr, A_intra_r_ptr, A_intra_i_ptr,
    L_u_r_ptr, L_u_i_ptr, L_a_r_ptr, L_a_i_ptr,
    s_u_b, s_u_d, s_u_t, s_lr_b, s_lr_d, s_lr_t, s_lam_b, s_lam_d,
    L, chunk_size: tl.constexpr
):
    pid_bh = tl.program_id(0); pid_d = tl.program_id(1); pid_c = tl.program_id(2)
    t_offs = tl.arange(0, chunk_size); offs_t = pid_c * chunk_size + t_offs; mask_t = offs_t < L

    base_lam = pid_bh * s_lam_b + pid_d * s_lam_d
    lam_r = tl.load(Lam_r_ptr + base_lam); lam_i = tl.load(Lam_i_ptr + base_lam)

    base_u = pid_bh * s_u_b + pid_d * s_u_d
    u_pre_r = tl.load(U_pre_r_ptr + base_u + offs_t * s_u_t, mask=mask_t, other=0.0)
    u_pre_i = tl.load(U_pre_i_ptr + base_u + offs_t * s_u_t, mask=mask_t, other=0.0)

    base_lr = pid_bh * s_lr_b
    lr = tl.load(LR_ptr + base_lr + offs_t * s_lr_t, mask=mask_t, other=0.0)

    a_r = lr * lam_r + (1.0 - lr); a_i = lr * lam_i
    u_r = u_pre_r * lr; u_i = u_pre_i * lr

    scan_ar, scan_ai, scan_ur, scan_ui = tl.associative_scan((a_r, a_i, u_r, u_i), 0, fused_op)

    off_store = base_u + offs_t * s_u_t
    tl.store(H_intra_r_ptr + off_store, scan_ur, mask=mask_t)
    tl.store(H_intra_i_ptr + off_store, scan_ui, mask=mask_t)
    tl.store(A_intra_r_ptr + off_store, scan_ar, mask=mask_t)
    tl.store(A_intra_i_ptr + off_store, scan_ai, mask=mask_t)

    last_idx = chunk_size - 1
    b_ur = tl.sum(tl.where(t_offs == last_idx, scan_ur, 0.0), axis=0)
    b_ui = tl.sum(tl.where(t_offs == last_idx, scan_ui, 0.0), axis=0)
    b_ar = tl.sum(tl.where(t_offs == last_idx, scan_ar, 0.0), axis=0)
    b_ai = tl.sum(tl.where(t_offs == last_idx, scan_ai, 0.0), axis=0)
    
    n_chunks = tl.num_programs(2)
    off_l = pid_bh * (tl.num_programs(1) * n_chunks) + pid_d * n_chunks + pid_c
    tl.store(L_u_r_ptr + off_l, b_ur); tl.store(L_u_i_ptr + off_l, b_ui)
    tl.store(L_a_r_ptr + off_l, b_ar); tl.store(L_a_i_ptr + off_l, b_ai)

@triton.jit
def complex_chunk_update_fwd_kernel(
    Hr_intra_ptr, Hi_intra_ptr, Ar_intra_ptr, Ai_intra_ptr,
    Global_Hr_ptr, Global_Hi_ptr, Hr_final_ptr, Hi_final_ptr,
    stride_batch, stride_dim, stride_seq, seq_len, chunk_size: tl.constexpr
):
    pid_batch = tl.program_id(0); pid_dim = tl.program_id(1); pid_chunk = tl.program_id(2)
    offs_base = pid_batch * stride_batch + pid_dim * stride_dim
    t_offs = tl.arange(0, chunk_size); global_idx = pid_chunk * chunk_size + t_offs; mask = global_idx < seq_len

    offs_carry = pid_batch * (tl.num_programs(1) * tl.num_programs(2)) + pid_dim * tl.num_programs(2) + pid_chunk
    carry_r = tl.load(Global_Hr_ptr + offs_carry * 2); carry_i = tl.load(Global_Hi_ptr + offs_carry * 2)

    ar = tl.load(Ar_intra_ptr + offs_base + global_idx * stride_seq, mask=mask, other=0.0)
    ai = tl.load(Ai_intra_ptr + offs_base + global_idx * stride_seq, mask=mask, other=0.0)
    hr = tl.load(Hr_intra_ptr + offs_base + global_idx * stride_seq, mask=mask, other=0.0)
    hi = tl.load(Hi_intra_ptr + offs_base + global_idx * stride_seq, mask=mask, other=0.0)

    tr, ti = complex_mul(carry_r, carry_i, ar, ai)
    tl.store(Hr_final_ptr + offs_base + global_idx * stride_seq, hr + tr, mask=mask)
    tl.store(Hi_final_ptr + offs_base + global_idx * stride_seq, hi + ti, mask=mask)

# =========================================================================
# 3. FULLY FUSED BACKWARD KERNELS
# =========================================================================

@triton.jit
def fused_chunk_scan_bwd_kernel(
    Gr_out_ptr, Gi_out_ptr, LR_ptr, Lam_r_ptr, Lam_i_ptr, 
    Gur_intra_ptr, Gui_intra_ptr, Ar_rev_ptr, Ai_rev_ptr,       
    Lr_gu_ptr, Li_gu_ptr, Lr_a_ptr, Li_a_ptr, 
    s_u_b, s_u_d, s_u_t, s_lr_b, s_lr_d, s_lr_t, s_lam_b, s_lam_d,
    L, chunk_size: tl.constexpr
):
    pid_bh = tl.program_id(0); pid_d = tl.program_id(1); pid_c = tl.program_id(2)
    t_offs = tl.arange(0, chunk_size); rev_t_offs = (chunk_size - 1) - t_offs
    global_idx = pid_c * chunk_size + rev_t_offs; mask = global_idx < L
    
    base_u = pid_bh * s_u_b + pid_d * s_u_d
    base_lr = pid_bh * s_lr_b
    base_lam = pid_bh * s_lam_b + pid_d * s_lam_d

    gr = tl.load(Gr_out_ptr + base_u + global_idx * s_u_t, mask=mask, other=0.0)
    gi = tl.load(Gi_out_ptr + base_u + global_idx * s_u_t, mask=mask, other=0.0)
    
    idx_a = global_idx + 1; mask_a = idx_a < L
    lam_r = tl.load(Lam_r_ptr + base_lam); lam_i = tl.load(Lam_i_ptr + base_lam)
    lr_shift = tl.load(LR_ptr + base_lr + idx_a * s_lr_t, mask=mask_a, other=0.0)
    
    ar_shifted = lr_shift * lam_r + (1.0 - lr_shift)
    ai_shifted = lr_shift * lam_i
    ai_shifted_conj = -ai_shifted
    
    scan_ar, scan_ai, scan_gr, scan_gi = tl.associative_scan((ar_shifted, ai_shifted_conj, gr, gi), 0, fused_op_bwd)
    
    off_store = base_u + global_idx * s_u_t
    tl.store(Gur_intra_ptr + off_store, scan_gr, mask=mask)
    tl.store(Gui_intra_ptr + off_store, scan_gi, mask=mask)
    tl.store(Ar_rev_ptr + off_store, scan_ar, mask=mask)
    tl.store(Ai_rev_ptr + off_store, scan_ai, mask=mask)
    
    last_idx = chunk_size - 1
    b_gr = tl.sum(tl.where(t_offs == last_idx, scan_gr, 0.0), axis=0)
    b_gi = tl.sum(tl.where(t_offs == last_idx, scan_gi, 0.0), axis=0)
    b_ar = tl.sum(tl.where(t_offs == last_idx, scan_ar, 0.0), axis=0)
    b_ai = tl.sum(tl.where(t_offs == last_idx, scan_ai, 0.0), axis=0)
    
    n_chunks = tl.num_programs(2)
    off_l = pid_bh * (tl.num_programs(1) * n_chunks) + pid_d * n_chunks + pid_c
    tl.store(Lr_gu_ptr + off_l, b_gr); tl.store(Li_gu_ptr + off_l, b_gi)
    tl.store(Lr_a_ptr + off_l, b_ar); tl.store(Li_a_ptr + off_l, b_ai)

@triton.jit
def complex_chunk_final_bwd_kernel_fully_fused(
    # Inputs
    Gur_intra_ptr, Gui_intra_ptr, Ar_rev_ptr, Ai_rev_ptr, 
    Hr_ptr, Hi_ptr, Hr_init_ptr, Hi_init_ptr,
    Global_Gr_ptr, Global_Gi_ptr,
    
    # Context for chain rule
    U_pre_r_ptr, U_pre_i_ptr, LR_ptr, Lam_r_ptr, Lam_i_ptr,
    
    # Direct Gradients Outputs
    Dupre_r_ptr, Dupre_i_ptr,
    Dlr_ptr, # REAL output
    Dlam_r_ptr, Dlam_i_ptr,
    
    # Strides Input
    s_u_b, s_u_d, s_u_t,
    s_lr_b, s_lr_d, s_lr_t,
    s_lam_b, s_lam_d,
    
    # Strides Output Real (For Dlr_ptr)
    s_rout_b, s_rout_d, s_rout_t,
    
    seq_len, chunk_size: tl.constexpr, DIM: tl.constexpr, HAS_INIT: tl.constexpr
):
    pid_batch, pid_dim, pid_chunk = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    offs_base = pid_batch * s_u_b + pid_dim * s_u_d
    t_offs = tl.arange(0, chunk_size)
    global_idx = pid_chunk * chunk_size + t_offs
    mask = global_idx < seq_len
    
    # 1. Reconstruct dU
    offs_carry = pid_batch * (tl.num_programs(1) * tl.num_programs(2)) + pid_dim * tl.num_programs(2) + pid_chunk
    carry_r = tl.load(Global_Gr_ptr + offs_carry * 2)
    carry_i = tl.load(Global_Gi_ptr + offs_carry * 2)
    
    gur = tl.load(Gur_intra_ptr + offs_base + global_idx * s_u_t, mask=mask, other=0.0)
    gui = tl.load(Gui_intra_ptr + offs_base + global_idx * s_u_t, mask=mask, other=0.0)
    ar_rev = tl.load(Ar_rev_ptr + offs_base + global_idx * s_u_t, mask=mask, other=0.0)
    ai_rev = tl.load(Ai_rev_ptr + offs_base + global_idx * s_u_t, mask=mask, other=0.0)
    
    tr, ti = complex_mul(carry_r, carry_i, ar_rev, ai_rev)
    du_r = gur + tr
    du_i = gui + ti
    
    # 2. Reconstruct dA
    idx_prev = global_idx - 1
    hr_prev = tl.zeros([chunk_size], dtype=tl.float32); hi_prev = tl.zeros([chunk_size], dtype=tl.float32)
    mask_h = (idx_prev >= 0) & (idx_prev < seq_len)
    val_hr = tl.load(Hr_ptr + offs_base + idx_prev * s_u_t, mask=mask_h, other=0.0)
    val_hi = tl.load(Hi_ptr + offs_base + idx_prev * s_u_t, mask=mask_h, other=0.0)
    hr_prev = tl.where(idx_prev >= 0, val_hr, hr_prev)
    hi_prev = tl.where(idx_prev >= 0, val_hi, hi_prev)
    
    if HAS_INIT:
        offs_init = pid_batch * DIM + pid_dim
        val_hr_init = tl.load(Hr_init_ptr + offs_init * 2)
        val_hi_init = tl.load(Hi_init_ptr + offs_init * 2 + 1) # Corrected +1 removal previously, put back +1 if init is contiguous complex
        # wait, if h_init is [B, D] complex64, contiguous. 
        # real is at index*2, imag at index*2+1.
        # If pointers are .real and .imag from torch, then +0 is correct.
        # Assuming pointers are passed as .real and .imag from python wrapper:
        val_hr_init = tl.load(Hr_init_ptr + offs_init * 2)
        val_hi_init = tl.load(Hi_init_ptr + offs_init * 2)
        hr_prev = tl.where(global_idx == 0, val_hr_init, hr_prev)
        hi_prev = tl.where(global_idx == 0, val_hi_init, hi_prev)
    
    da_r, da_i = complex_mul_conj(du_r, du_i, hr_prev, hi_prev)
    
    # 3. Chain Rule
    
    # Load Context
    upre_r = tl.load(U_pre_r_ptr + offs_base + global_idx * s_u_t, mask=mask, other=0.0)
    upre_i = tl.load(U_pre_i_ptr + offs_base + global_idx * s_u_t, mask=mask, other=0.0)
    
    base_lr = pid_batch * s_lr_b 
    lr = tl.load(LR_ptr + base_lr + global_idx * s_lr_t, mask=mask, other=0.0)
    
    base_lam = pid_batch * s_lam_b + pid_dim * s_lam_d
    lam_r = tl.load(Lam_r_ptr + base_lam); lam_i = tl.load(Lam_i_ptr + base_lam)
    
    # A. dL/dU_pre
    dupre_r = du_r * lr
    dupre_i = du_i * lr
    tl.store(Dupre_r_ptr + offs_base + global_idx * s_u_t, dupre_r, mask=mask)
    tl.store(Dupre_i_ptr + offs_base + global_idx * s_u_t, dupre_i, mask=mask)
    
    # B. dL/dLR (Scalar Real)
    # term1 = Re(dU * conj(U_pre))
    term1 = du_r * upre_r + du_i * upre_i
    # term2 = Re(dA * conj(Lam - 1))
    lam_minus_1_r = lam_r - 1.0
    term2 = da_r * lam_minus_1_r + da_i * lam_i
    
    dlr_val = term1 + term2
    
    # Store dLR (Real Tensor) -> Use specific strides s_rout_*
    offs_out_real = pid_batch * s_rout_b + pid_dim * s_rout_d + global_idx * s_rout_t
    tl.store(Dlr_ptr + offs_out_real, dlr_val, mask=mask)
    
    # C. dL/dLambda (Complex)
    dlam_r = da_r * lr
    dlam_i = da_i * lr
    # dLambda has same layout as U_pre (complex)
    tl.store(Dlam_r_ptr + offs_base + global_idx * s_u_t, dlam_r, mask=mask)
    tl.store(Dlam_i_ptr + offs_base + global_idx * s_u_t, dlam_i, mask=mask)


# =========================================================================
# 4. WRAPPER AUTOGRAD FUSED
# =========================================================================

class FusedComplexSSM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u_pre, lr, lam, h_init, chunk_size):
        ctx.is_multihead = (u_pre.ndim == 4)
        if ctx.is_multihead:
            B, H, D, L = u_pre.shape
            u_pre_flat = u_pre.view(B*H, D, L)
            lr_flat = lr.view(B*H, 1, L)
            lam_flat = lam.view(B*H, D)
            if h_init is not None: h_init_flat = h_init.view(B*H, D)
            B_eff = B*H
        else:
            B_eff, D, L = u_pre.shape
            u_pre_flat, lr_flat, lam_flat = u_pre, lr, lam
            h_init_flat = h_init
            B, H = B_eff, 1

        u_pre_flat = u_pre_flat.contiguous()
        lr_flat = lr_flat.contiguous()
        lam_flat = lam_flat.contiguous()
        if h_init is not None: h_init_flat = h_init_flat.contiguous()

        num_chunks = (L + chunk_size - 1) // chunk_size
        
        h_intra = torch.zeros_like(u_pre_flat)
        a_intra = torch.zeros_like(u_pre_flat)
        
        L_u_r = torch.zeros((B_eff, D, num_chunks), device=u_pre.device, dtype=torch.float32)
        L_u_i = torch.zeros((B_eff, D, num_chunks), device=u_pre.device, dtype=torch.float32)
        L_a_r = torch.zeros((B_eff, D, num_chunks), device=u_pre.device, dtype=torch.float32)
        L_a_i = torch.zeros((B_eff, D, num_chunks), device=u_pre.device, dtype=torch.float32)
        
        grid = (B_eff, D, num_chunks)
        
        fused_chunk_scan_fwd_kernel[grid](
            u_pre_flat.real, u_pre_flat.imag,
            lr_flat,
            lam_flat.real, lam_flat.imag,
            h_intra.real, h_intra.imag,
            a_intra.real, a_intra.imag,
            L_u_r, L_u_i, L_a_r, L_a_i,
            u_pre_flat.stride(0)*2, u_pre_flat.stride(1)*2, u_pre_flat.stride(2)*2,
            lr_flat.stride(0), lr_flat.stride(1), lr_flat.stride(2),
            lam_flat.stride(0)*2, lam_flat.stride(1)*2,
            L, chunk_size
        )
        
        L_u = torch.complex(L_u_r, L_u_i)
        L_a = torch.complex(L_a_r, L_a_i)
        carries = torch.zeros_like(L_u)
        curr = h_init_flat.clone() if h_init is not None else torch.zeros((B_eff, D), device=u_pre.device, dtype=u_pre.dtype)
        
        for i in range(num_chunks):
            carries[:,:,i] = curr
            curr = L_a[:,:,i] * curr + L_u[:,:,i]
            
        h_final = torch.zeros_like(u_pre_flat)
        
        complex_chunk_update_fwd_kernel[grid](
            h_intra.real, h_intra.imag, a_intra.real, a_intra.imag,
            carries.real, carries.imag,
            h_final.real, h_final.imag,
            u_pre_flat.stride(0)*2, u_pre_flat.stride(1)*2, u_pre_flat.stride(2)*2,
            L, chunk_size
        )
        
        ctx.save_for_backward(u_pre_flat, lr_flat, lam_flat, h_init_flat, h_final)
        ctx.chunk_size = chunk_size
        ctx.has_init = (h_init is not None)
        ctx.orig_shape = (B, H, D, L) if ctx.is_multihead else (B_eff, D, L)
        
        if ctx.is_multihead:
            return h_final.view(B, H, D, L)
        return h_final

    @staticmethod
    def backward(ctx, grad_output):
        u_pre, lr, lam, h_init, h_final = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        
        if ctx.is_multihead:
            B, H, D, L = ctx.orig_shape
            grad_output = grad_output.reshape(B*H, D, L)
            
        B_eff, D, L = u_pre.shape
        chunk_size = ctx.chunk_size
        num_chunks = (L + chunk_size - 1) // chunk_size
        grid = (B_eff, D, num_chunks)
        
        du_intra = torch.zeros_like(u_pre)
        a_rev = torch.zeros_like(u_pre)
        
        L_du_r = torch.zeros((B_eff, D, num_chunks), device=u_pre.device, dtype=torch.float32)
        L_du_i = torch.zeros((B_eff, D, num_chunks), device=u_pre.device, dtype=torch.float32)
        L_ar_rev = torch.zeros((B_eff, D, num_chunks), device=u_pre.device, dtype=torch.float32)
        L_ai_rev = torch.zeros((B_eff, D, num_chunks), device=u_pre.device, dtype=torch.float32)
        
        fused_chunk_scan_bwd_kernel[grid](
            grad_output.real, grad_output.imag,
            lr, lam.real, lam.imag,
            du_intra.real, du_intra.imag,
            a_rev.real, a_rev.imag,
            L_du_r, L_du_i, L_ar_rev, L_ai_rev,
            u_pre.stride(0)*2, u_pre.stride(1)*2, u_pre.stride(2)*2,
            lr.stride(0), lr.stride(1), lr.stride(2),
            lam.stride(0)*2, lam.stride(1)*2,
            L, chunk_size
        )
        
        L_du = torch.complex(L_du_r, L_du_i)
        L_a_rev = torch.complex(L_ar_rev, L_ai_rev)
        
        grad_carries = torch.zeros_like(L_du)
        curr_g = torch.zeros((B_eff, D), device=u_pre.device, dtype=u_pre.dtype)
        for i in range(num_chunks - 1, -1, -1):
            grad_carries[:,:,i] = curr_g
            curr_g = curr_g * L_a_rev[:,:,i] + L_du[:,:,i]
            
        d_h_init = None
        if ctx.has_init:
            lr0 = lr[:, :, 0] 
            a0 = lr0 * lam + (1.0 - lr0)
            d_h_init = curr_g * a0.conj()
            
        # --- FULLY FUSED BACKWARD ---
        d_upre = torch.zeros_like(u_pre) # Complex
        d_lr_map = torch.zeros((B_eff, D, L), device=u_pre.device, dtype=torch.float32) # REAL
        d_lam_map = torch.zeros_like(u_pre) # Complex
        
        hr_init_ptr = h_init.real if ctx.has_init else u_pre.real 
        hi_init_ptr = h_init.imag if ctx.has_init else u_pre.imag
        
        complex_chunk_final_bwd_kernel_fully_fused[grid](
            du_intra.real, du_intra.imag,
            a_rev.real, a_rev.imag,
            h_final.real, h_final.imag,
            hr_init_ptr, hi_init_ptr,
            grad_carries.real, grad_carries.imag,
            
            # Context
            u_pre.real, u_pre.imag,
            lr,
            lam.real, lam.imag,
            
            # Outputs
            d_upre.real, d_upre.imag,
            d_lr_map, # Real Ptr
            d_lam_map.real, d_lam_map.imag,
            
            u_pre.stride(0)*2, u_pre.stride(1)*2, u_pre.stride(2)*2,
            lr.stride(0), lr.stride(1), lr.stride(2),
            lam.stride(0)*2, lam.stride(1)*2,
            
            # Real Strides for d_lr_map
            d_lr_map.stride(0), d_lr_map.stride(1), d_lr_map.stride(2),
            
            L, chunk_size,
            DIM=D, HAS_INIT=ctx.has_init
        )
        
        # Reductions Python
        d_lr = d_lr_map.sum(dim=1, keepdim=True) # Sum over D
        d_lam = d_lam_map.sum(dim=2) # Sum over L
        
        if ctx.is_multihead:
            d_upre = d_upre.view(B, H, D, L)
            d_lr = d_lr.view(B, H, 1, L)
            d_lam = d_lam.view(B, H, D)
            if d_h_init is not None: d_h_init = d_h_init.view(B, H, D)
            
        return d_upre, d_lr, d_lam, d_h_init, None

def parallel_fused_ssm(u_pre, lr, lam, h_init=None, chunk_size=2048):
    return FusedComplexSSM.apply(u_pre, lr, lam, h_init, chunk_size)