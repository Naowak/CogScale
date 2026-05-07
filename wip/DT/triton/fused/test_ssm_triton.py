import torch
import sys

from libs.DT.triton.fused.ssm_triton_fused import parallel_fused_ssm



# try:
#     from fused_chunked_complex_ssm import parallel_fused_ssm
# except ImportError:
#     print("❌ 'fused_chunked_complex_ssm.py' not found")
#     sys.exit(1)

def naive_ssm_loop(u_pre, lr, lam, h_init=None):
    # Reconstruct A and U explicitly (Materialization)
    lam_expanded = lam.unsqueeze(-1) # [B, H, D, 1]
    a = lr * lam_expanded + (1.0 - lr)
    u = u_pre * lr
    
    # Scan
    B, H, D, T = u.shape
    h = torch.zeros((B, H, D), device=u.device, dtype=u.dtype) if h_init is None else h_init.clone()
    h_out = []
    
    for t in range(T):
        h = a[..., t] * h + u[..., t]
        h_out.append(h)
        
    return torch.stack(h_out, dim=-1)

def run_test():
    torch.manual_seed(42)
    device = 'cuda'
    
    B, H, D, T = 2, 4, 32, 1024
    CHUNK = 256
    
    print(f"Testing Fused SSM [B={B}, H={H}, D={D}, T={T}]...")
    
    # Inputs
    u_pre = torch.randn(B, H, D, T, device=device, dtype=torch.complex64, requires_grad=True)
    lr = torch.rand(B, H, 1, T, device=device, dtype=torch.float32, requires_grad=True)
    
    # STABILIZATION: Lambda must be <= 1 for stability over T=1024 steps
    raw_lam = torch.randn(B, H, D, device=device, dtype=torch.complex64)
    # Normalize to max magnitude 0.95 to prevent gradient explosion
    mag = raw_lam.abs()
    scale = torch.where(mag > 0.95, 0.95 / mag, torch.ones_like(mag))
    lam = (raw_lam * scale).requires_grad_(True)
    
    h_init = torch.randn(B, H, D, device=device, dtype=torch.complex64, requires_grad=True)
    
    # 1. Fused
    y_fused = parallel_fused_ssm(u_pre, lr, lam, h_init, chunk_size=CHUNK)
    
    # 2. Naive
    y_ref = naive_ssm_loop(u_pre, lr, lam, h_init)
    
    # Check Forward
    diff = (y_fused - y_ref).abs().max()
    print(f"Forward Diff: {diff:.6f} {'✅' if diff < 1e-3 else '❌'}")
    
    # Check Backward
    # Reset grads
    u_pre.grad = None; lr.grad = None; lam.grad = None; h_init.grad = None
    
    # Reference Backward
    y_ref_new = naive_ssm_loop(u_pre, lr, lam, h_init)
    y_ref_new.real.sum().backward()
    
    g_upre_ref = u_pre.grad.clone()
    g_lr_ref = lr.grad.clone()
    g_lam_ref = lam.grad.clone()
    g_h_init_ref = h_init.grad.clone()
    
    # Zero grads again
    u_pre.grad = None; lr.grad = None; lam.grad = None; h_init.grad = None
    
    # Fused Backward
    y_fused_new = parallel_fused_ssm(u_pre, lr, lam, h_init, chunk_size=CHUNK)
    y_fused_new.real.sum().backward()
    
    print("\nChecking Gradients:")
    for name, g_calc, g_true in [
        ("U_pre", u_pre.grad, g_upre_ref),
        ("LR", lr.grad, g_lr_ref),
        ("Lambda", lam.grad, g_lam_ref),
        ("H_init", h_init.grad, g_h_init_ref)
    ]:
        err = (g_calc - g_true).abs().max()
        # Tolerances: Lambda/LR can have slightly higher numerical drift due to recurrence
        tol = 1e-2 if name in ["Lambda", "LR"] else 1e-3
        print(f"Grad {name} Diff: {err:.6f} {'✅' if err < tol else '❌'}")

if __name__ == "__main__":
    run_test()