import torch
import triton
import triton.language as tl

# =========================================================================
# 1. HELPERS COMPLEXES (JIT)
# =========================================================================

@triton.jit
def complex_mul(r_a, i_a, r_b, i_b):
    """ (a + ib) * (c + id) """
    r_out = r_a * r_b - i_a * i_b
    i_out = r_a * i_b + i_a * r_b
    return r_out, i_out

@triton.jit
def complex_mul_conj(r_a, i_a, r_b, i_b):
    """ (a + ib) * conj(c + id) = (a + ib) * (c - id) """
    r_out = r_a * r_b + i_a * i_b
    i_out = i_a * r_b - r_a * i_b
    return r_out, i_out

@triton.jit
def complex_first_order_op(
    ar_l, ai_l, ur_l, ui_l, 
    ar_r, ai_r, ur_r, ui_r
):
    """ (A_new, U_new) = (A_r, U_r) • (A_l, U_l) """
    ar_new, ai_new = complex_mul(ar_r, ai_r, ar_l, ai_l)
    
    term_r, term_i = complex_mul(ar_r, ai_r, ur_l, ui_l)
    ur_new = term_r + ur_r
    ui_new = term_i + ui_r

    return ar_new, ai_new, ur_new, ui_new

@triton.jit
def complex_first_order_op_bwd(
    ar_l, ai_l, ur_l, ui_l, 
    ar_r, ai_r, ur_r, ui_r
):
    """ Meme operateur structurellement """
    ar_new, ai_new = complex_mul(ar_r, ai_r, ar_l, ai_l)
    term_r, term_i = complex_mul(ar_r, ai_r, ur_l, ui_l)
    ur_new = term_r + ur_r
    ui_new = term_i + ui_r
    return ar_new, ai_new, ur_new, ui_new

# =========================================================================
# 2. FORWARD KERNELS
# =========================================================================

@triton.jit
def complex_chunk_scan_fwd_kernel(
    Ur_ptr, Ui_ptr, Ar_ptr, Ai_ptr,
    Hr_intra_ptr, Hi_intra_ptr, Ar_intra_ptr, Ai_intra_ptr,
    Lr_u_ptr, Li_u_ptr, Lr_a_ptr, Li_a_ptr,
    stride_batch, stride_dim, stride_seq,
    seq_len, chunk_size: tl.constexpr
):
    # pid_batch ici correspond à (Batch * Head)
    pid_batch, pid_dim, pid_chunk = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    
    offs_base = pid_batch * stride_batch + pid_dim * stride_dim
    offs_chunk = pid_chunk * chunk_size
    t_offs = tl.arange(0, chunk_size)
    global_idx = offs_chunk + t_offs
    mask = global_idx < seq_len

    # Load Inputs
    ur = tl.load(Ur_ptr + offs_base + global_idx * stride_seq, mask=mask, other=0.0)
    ui = tl.load(Ui_ptr + offs_base + global_idx * stride_seq, mask=mask, other=0.0)
    ar = tl.load(Ar_ptr + offs_base + global_idx * stride_seq, mask=mask, other=1.0)
    ai = tl.load(Ai_ptr + offs_base + global_idx * stride_seq, mask=mask, other=0.0)

    # Scan
    scan_ar, scan_ai, scan_ur, scan_ui = tl.associative_scan(
        (ar, ai, ur, ui), 0, complex_first_order_op
    )

    # Store Partials
    tl.store(Hr_intra_ptr + offs_base + global_idx * stride_seq, scan_ur, mask=mask)
    tl.store(Hi_intra_ptr + offs_base + global_idx * stride_seq, scan_ui, mask=mask)
    tl.store(Ar_intra_ptr + offs_base + global_idx * stride_seq, scan_ar, mask=mask)
    tl.store(Ai_intra_ptr + offs_base + global_idx * stride_seq, scan_ai, mask=mask)
    
    # Store Summaries (Block Reduction)
    last_idx = chunk_size - 1
    
    block_ur = tl.sum(tl.where(t_offs == last_idx, scan_ur, 0.0), axis=0)
    block_ui = tl.sum(tl.where(t_offs == last_idx, scan_ui, 0.0), axis=0)
    block_ar = tl.sum(tl.where(t_offs == last_idx, scan_ar, 0.0), axis=0)
    block_ai = tl.sum(tl.where(t_offs == last_idx, scan_ai, 0.0), axis=0)
    
    # Layout dense pour les summaries (car alloués séparément en float32)
    offs_l = pid_batch * (tl.num_programs(1) * tl.num_programs(2)) + \
             pid_dim * tl.num_programs(2) + pid_chunk
    
    tl.store(Lr_u_ptr + offs_l, block_ur)
    tl.store(Li_u_ptr + offs_l, block_ui)
    tl.store(Lr_a_ptr + offs_l, block_ar)
    tl.store(Li_a_ptr + offs_l, block_ai)

@triton.jit
def complex_chunk_update_fwd_kernel(
    Hr_intra_ptr, Hi_intra_ptr, Ar_intra_ptr, Ai_intra_ptr,
    Global_Hr_ptr, Global_Hi_ptr,
    Hr_final_ptr, Hi_final_ptr,
    stride_batch, stride_dim, stride_seq,
    seq_len, chunk_size: tl.constexpr
):
    pid_batch, pid_dim, pid_chunk = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    offs_base = pid_batch * stride_batch + pid_dim * stride_dim
    offs_chunk = pid_chunk * chunk_size
    t_offs = tl.arange(0, chunk_size)
    global_idx = offs_chunk + t_offs
    mask = global_idx < seq_len

    # Load Carry
    # CORRECTION CRITIQUE: carries est un tenseur complexe (R, I, R, I...)
    # L'offset calculé ici est l'index "élément".
    # Pour accéder au float32 correspondant via .real/.imag pointer, il faut multiplier par 2.
    offs_carry = pid_batch * (tl.num_programs(1) * tl.num_programs(2)) + \
                 pid_dim * tl.num_programs(2) + pid_chunk
    
    carry_r = tl.load(Global_Hr_ptr + offs_carry * 2) # * 2 pour saut complexe
    carry_i = tl.load(Global_Hi_ptr + offs_carry * 2)

    # Load Partials
    ar = tl.load(Ar_intra_ptr + offs_base + global_idx * stride_seq, mask=mask, other=0.0)
    ai = tl.load(Ai_intra_ptr + offs_base + global_idx * stride_seq, mask=mask, other=0.0)
    hr = tl.load(Hr_intra_ptr + offs_base + global_idx * stride_seq, mask=mask, other=0.0)
    hi = tl.load(Hi_intra_ptr + offs_base + global_idx * stride_seq, mask=mask, other=0.0)

    # Update
    term_r, term_i = complex_mul(carry_r, carry_i, ar, ai)
    hf_r = hr + term_r
    hf_i = hi + term_i

    tl.store(Hr_final_ptr + offs_base + global_idx * stride_seq, hf_r, mask=mask)
    tl.store(Hi_final_ptr + offs_base + global_idx * stride_seq, hf_i, mask=mask)

# =========================================================================
# 3. BACKWARD KERNELS
# =========================================================================

@triton.jit
def complex_chunk_scan_bwd_kernel(
    Gr_out_ptr, Gi_out_ptr, Ar_ptr, Ai_ptr,
    Gur_intra_ptr, Gui_intra_ptr, Ar_rev_ptr, Ai_rev_ptr,
    Lr_gu_ptr, Li_gu_ptr, Lr_a_ptr, Li_a_ptr,
    stride_batch, stride_dim, stride_seq,
    seq_len, chunk_size: tl.constexpr
):
    pid_batch, pid_dim, pid_chunk = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    offs_base = pid_batch * stride_batch + pid_dim * stride_dim
    offs_chunk = pid_chunk * chunk_size
    
    t_offs = tl.arange(0, chunk_size)
    rev_t_offs = (chunk_size - 1) - t_offs
    global_idx = offs_chunk + rev_t_offs
    mask = global_idx < seq_len
    
    gr = tl.load(Gr_out_ptr + offs_base + global_idx * stride_seq, mask=mask, other=0.0)
    gi = tl.load(Gi_out_ptr + offs_base + global_idx * stride_seq, mask=mask, other=0.0)
    
    idx_a = global_idx + 1
    mask_a = idx_a < seq_len
    
    ar_shifted = tl.load(Ar_ptr + offs_base + idx_a * stride_seq, mask=mask_a, other=0.0)
    ai_shifted = tl.load(Ai_ptr + offs_base + idx_a * stride_seq, mask=mask_a, other=0.0)
    
    # CONJUGAISON (Wirtinger)
    ai_shifted_conj = -ai_shifted 
    
    scan_ar, scan_ai, scan_gr, scan_gi = tl.associative_scan(
        (ar_shifted, ai_shifted_conj, gr, gi), 0, complex_first_order_op_bwd
    )
    
    tl.store(Gur_intra_ptr + offs_base + global_idx * stride_seq, scan_gr, mask=mask)
    tl.store(Gui_intra_ptr + offs_base + global_idx * stride_seq, scan_gi, mask=mask)
    tl.store(Ar_rev_ptr + offs_base + global_idx * stride_seq, scan_ar, mask=mask)
    tl.store(Ai_rev_ptr + offs_base + global_idx * stride_seq, scan_ai, mask=mask)
    
    last_idx = chunk_size - 1
    b_gr = tl.sum(tl.where(t_offs == last_idx, scan_gr, 0.0), axis=0)
    b_gi = tl.sum(tl.where(t_offs == last_idx, scan_gi, 0.0), axis=0)
    b_ar = tl.sum(tl.where(t_offs == last_idx, scan_ar, 0.0), axis=0)
    b_ai = tl.sum(tl.where(t_offs == last_idx, scan_ai, 0.0), axis=0)
    
    offs_l = pid_batch * (tl.num_programs(1) * tl.num_programs(2)) + \
             pid_dim * tl.num_programs(2) + pid_chunk
             
    tl.store(Lr_gu_ptr + offs_l, b_gr)
    tl.store(Li_gu_ptr + offs_l, b_gi)
    tl.store(Lr_a_ptr + offs_l, b_ar)
    tl.store(Li_a_ptr + offs_l, b_ai)

@triton.jit
def complex_chunk_final_bwd_kernel(
    Gur_intra_ptr, Gui_intra_ptr, Ar_rev_ptr, Ai_rev_ptr, 
    Hr_ptr, Hi_ptr, Hr_init_ptr, Hi_init_ptr,
    Global_Gr_ptr, Global_Gi_ptr,
    Dur_final_ptr, Dui_final_ptr, Dar_final_ptr, Dai_final_ptr,
    stride_batch, stride_dim, stride_seq,
    seq_len, chunk_size: tl.constexpr, DIM: tl.constexpr, HAS_INIT: tl.constexpr
):
    pid_batch, pid_dim, pid_chunk = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    offs_base = pid_batch * stride_batch + pid_dim * stride_dim
    offs_chunk = pid_chunk * chunk_size
    t_offs = tl.arange(0, chunk_size)
    global_idx = offs_chunk + t_offs
    mask = global_idx < seq_len
    
    # CORRECTION CRITIQUE: grad_carries est complexe (R I R I)
    offs_carry = pid_batch * (tl.num_programs(1) * tl.num_programs(2)) + \
                 pid_dim * tl.num_programs(2) + pid_chunk
    carry_r = tl.load(Global_Gr_ptr + offs_carry * 2) # * 2
    carry_i = tl.load(Global_Gi_ptr + offs_carry * 2)
    
    gur = tl.load(Gur_intra_ptr + offs_base + global_idx * stride_seq, mask=mask, other=0.0)
    gui = tl.load(Gui_intra_ptr + offs_base + global_idx * stride_seq, mask=mask, other=0.0)
    ar_rev = tl.load(Ar_rev_ptr + offs_base + global_idx * stride_seq, mask=mask, other=0.0)
    ai_rev = tl.load(Ai_rev_ptr + offs_base + global_idx * stride_seq, mask=mask, other=0.0)
    
    term_r, term_i = complex_mul(carry_r, carry_i, ar_rev, ai_rev)
    du_r = gur + term_r
    du_i = gui + term_i
    
    tl.store(Dur_final_ptr + offs_base + global_idx * stride_seq, du_r, mask=mask)
    tl.store(Dui_final_ptr + offs_base + global_idx * stride_seq, du_i, mask=mask)
    
    idx_prev = global_idx - 1
    hr_prev = tl.zeros([chunk_size], dtype=tl.float32)
    hi_prev = tl.zeros([chunk_size], dtype=tl.float32)
    
    mask_h = (idx_prev >= 0) & (idx_prev < seq_len)
    val_hr = tl.load(Hr_ptr + offs_base + idx_prev * stride_seq, mask=mask_h, other=0.0)
    val_hi = tl.load(Hi_ptr + offs_base + idx_prev * stride_seq, mask=mask_h, other=0.0)
    hr_prev = tl.where(idx_prev >= 0, val_hr, hr_prev)
    hi_prev = tl.where(idx_prev >= 0, val_hi, hi_prev)
    
    if HAS_INIT:
        offs_init = pid_batch * DIM + pid_dim
        # Attention: H_init est aussi complexe, donc offset * 2 pour float ptr
        val_hr_init = tl.load(Hr_init_ptr + offs_init * 2) 
        # CORRECTION: Hi_init_ptr pointe déjà sur Imag. +1 décale de 1 float vers la partie Réelle suivante.
        # Il faut utiliser *2 pour sauter au Imag suivant.
        val_hi_init = tl.load(Hi_init_ptr + offs_init * 2)
        hr_prev = tl.where(global_idx == 0, val_hr_init, hr_prev)
        hi_prev = tl.where(global_idx == 0, val_hi_init, hi_prev)
        
    da_r, da_i = complex_mul_conj(du_r, du_i, hr_prev, hi_prev)
    
    tl.store(Dar_final_ptr + offs_base + global_idx * stride_seq, da_r, mask=mask)
    tl.store(Dai_final_ptr + offs_base + global_idx * stride_seq, da_i, mask=mask)

# =========================================================================
# 4. WRAPPER AUTOGRAD (MULTI-HEAD COMPATIBLE)
# =========================================================================

class ComplexSSMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, a, h_init, chunk_size):
        # --- PRE-PROCESSING SHAPES ---
        # Input Expected: [B, H, D, L] or [B, D, L]
        # We normalize everything to [Batch_Eff, D, L] where Batch_Eff = B * H
        
        ctx.is_multihead = (u.ndim == 4)
        if ctx.is_multihead:
            B, H, D, L = u.shape
            # Flatten Batch and Head
            u = u.view(B*H, D, L)
            a = a.view(B*H, D, L)
            if h_init is not None:
                h_init = h_init.view(B*H, D)
            # Batch Size effective
            B_eff = B * H
        else:
            B_eff, D, L = u.shape
            B, H = B_eff, 1

        # Assurer la contiguïté pour Triton (après le view)
        u = u.contiguous()
        a = a.contiguous()
        if h_init is not None: h_init = h_init.contiguous()
        
        num_chunks = (L + chunk_size - 1) // chunk_size
        
        h_intra = torch.zeros_like(u)
        a_intra = torch.zeros_like(a)
        
        # Allocations "Dense" Float pour les résumés
        L_u_r = torch.zeros((B_eff, D, num_chunks), device=u.device, dtype=torch.float32)
        L_u_i = torch.zeros((B_eff, D, num_chunks), device=u.device, dtype=torch.float32)
        L_a_r = torch.zeros((B_eff, D, num_chunks), device=u.device, dtype=torch.float32)
        L_a_i = torch.zeros((B_eff, D, num_chunks), device=u.device, dtype=torch.float32)
        
        grid = (B_eff, D, num_chunks)
        
        # NOTE: On multiplie les strides par 2 car on traite des pointeurs float32 
        # sur des données complex64 entrelacées [R, I, R, I...]
        complex_chunk_scan_fwd_kernel[grid](
            u.real, u.imag, a.real, a.imag,
            h_intra.real, h_intra.imag, a_intra.real, a_intra.imag,
            L_u_r, L_u_i, L_a_r, L_a_i,
            u.stride(0)*2, u.stride(1)*2, u.stride(2)*2,
            L, chunk_size=chunk_size
        )
        
        # Recombination pour le Step 2 Python
        L_u = torch.complex(L_u_r, L_u_i)
        L_a = torch.complex(L_a_r, L_a_i)
        
        carries = torch.zeros_like(L_u)
        curr = h_init.clone() if h_init is not None else torch.zeros((B_eff, D), device=u.device, dtype=u.dtype)
        
        for i in range(num_chunks):
            carries[:,:,i] = curr
            curr = L_a[:,:,i] * curr + L_u[:,:,i]
            
        h_final = torch.zeros_like(u)
        
        complex_chunk_update_fwd_kernel[grid](
            h_intra.real, h_intra.imag, a_intra.real, a_intra.imag,
            carries.real, carries.imag,
            h_final.real, h_final.imag,
            u.stride(0)*2, u.stride(1)*2, u.stride(2)*2,
            L, chunk_size=chunk_size
        )
        
        # Save tensors (Flattened state)
        ctx.save_for_backward(u, a, h_init, h_final)
        ctx.chunk_size = chunk_size
        ctx.has_init = (h_init is not None)
        ctx.orig_shape = (B, H, D, L) if ctx.is_multihead else (B_eff, D, L)
        
        # Reshape Output to original
        if ctx.is_multihead:
            return h_final.view(B, H, D, L)
        return h_final

    @staticmethod
    def backward(ctx, grad_output):
        u, a, h_init, h_final = ctx.saved_tensors
        
        # Flatten grad if needed (should match saved u shape)
        if ctx.is_multihead:
            B, H, D, L = ctx.orig_shape
            grad_output = grad_output.reshape(B*H, D, L)
            
        grad_output = grad_output.contiguous()
        
        B_eff, D, L = u.shape
        chunk_size = ctx.chunk_size
        num_chunks = (L + chunk_size - 1) // chunk_size
        grid = (B_eff, D, num_chunks)
        
        du_intra = torch.zeros_like(u)
        a_rev = torch.zeros_like(a)
        
        # Separate Floats for summaries
        L_du_r = torch.zeros((B_eff, D, num_chunks), device=u.device, dtype=torch.float32)
        L_du_i = torch.zeros((B_eff, D, num_chunks), device=u.device, dtype=torch.float32)
        L_ar_rev = torch.zeros((B_eff, D, num_chunks), device=u.device, dtype=torch.float32)
        L_ai_rev = torch.zeros((B_eff, D, num_chunks), device=u.device, dtype=torch.float32)
        
        complex_chunk_scan_bwd_kernel[grid](
            grad_output.real, grad_output.imag,
            a.real, a.imag,
            du_intra.real, du_intra.imag,
            a_rev.real, a_rev.imag,
            L_du_r, L_du_i, L_ar_rev, L_ai_rev,
            u.stride(0)*2, u.stride(1)*2, u.stride(2)*2,
            L, chunk_size=chunk_size
        )
        
        L_du = torch.complex(L_du_r, L_du_i)
        L_a_rev = torch.complex(L_ar_rev, L_ai_rev)
        
        grad_carries = torch.zeros((B_eff, D, num_chunks), device=u.device, dtype=u.dtype)
        curr_g = torch.zeros((B_eff, D), device=u.device, dtype=u.dtype)
        
        for i in range(num_chunks - 1, -1, -1):
            grad_carries[:,:,i] = curr_g
            curr_g = curr_g * L_a_rev[:,:,i] + L_du[:,:,i]
            
        d_h_init = None
        if ctx.has_init:
            d_h_init = curr_g * a[:,:,0].conj()
            
        du_final = torch.zeros_like(u)
        da_final = torch.zeros_like(a)
        
        hr_init_ptr = h_init.real if ctx.has_init else u.real
        hi_init_ptr = h_init.imag if ctx.has_init else u.imag
        
        complex_chunk_final_bwd_kernel[grid](
            du_intra.real, du_intra.imag,
            a_rev.real, a_rev.imag,
            h_final.real, h_final.imag,
            hr_init_ptr, hi_init_ptr,
            grad_carries.real, grad_carries.imag,
            du_final.real, du_final.imag,
            da_final.real, da_final.imag,
            u.stride(0)*2, u.stride(1)*2, u.stride(2)*2,
            L, chunk_size=chunk_size,
            DIM=D, HAS_INIT=ctx.has_init
        )
        
        # Reshape gradients to original
        if ctx.is_multihead:
            B, H, D, L = ctx.orig_shape
            du_final = du_final.view(B, H, D, L)
            da_final = da_final.view(B, H, D, L)
            if d_h_init is not None:
                d_h_init = d_h_init.view(B, H, D)
        
        return du_final, da_final, d_h_init, None

def parallel_complex_ssm(u, a, h_init=None, chunk_size=1024):
    """
    Entrée: [Batch, Head, Dim, Length] ou [Batch, Dim, Length]
    Sortie: Même shape
    """
    return ComplexSSMFunction.apply(u, a, h_init, chunk_size)