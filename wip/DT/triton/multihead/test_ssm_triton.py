import torch
import time
import sys
from libs.DT.triton.multihead.ssm_triton import parallel_complex_ssm

# --- REFERENCE PYTORCH (MULTI-HEAD NAIF) ---
def naive_multihead_ssm(u, a, h_init=None):
    """
    Naive implementation that loops over time.
    Supports [B, H, D, L] naturally via broadcasting.
    """
    # u: [B, H, D, L]
    B, H, D, L = u.shape
    h = torch.zeros((B, H, D), device=u.device, dtype=u.dtype) if h_init is None else h_init.clone()
    h_out = []
    
    for t in range(L):
        # h: [B, H, D]
        # a[..., t]: [B, H, D]
        h = a[..., t] * h + u[..., t]
        h_out.append(h)
    
    # Stack over time -> [B, H, D, L] (Time last)
    return torch.stack(h_out, dim=-1)

def run_test():
    if not torch.cuda.is_available():
        print("❌ CUDA non disponible.")
        return
        
    device = torch.device("cuda")
    torch.manual_seed(42)

    # Config Multi-Head
    BATCH, HEADS, DIM, TIME, CHUNK = 4, 8, 64, 32000, 2048
    print(f"\n{'='*60}")
    print(f"=== TEST MULTI-HEAD [B={BATCH}, H={HEADS}, D={DIM}, L={TIME}] ===")
    print(f"{'='*60}")

    # --- INPUTS COMPLEXES 4D ---
    # Shape: [B, H, D, L]
    
    # U = (Rand + i Rand)
    u_real = torch.randn(BATCH, HEADS, DIM, TIME, device=device)
    u_imag = torch.randn(BATCH, HEADS, DIM, TIME, device=device)
    u = torch.complex(u_real, u_imag).requires_grad_(True)
    
    # A = (Rand + i Rand)
    a_real = torch.rand(BATCH, HEADS, DIM, TIME, device=device) * 0.9 + 0.05
    a_imag = torch.rand(BATCH, HEADS, DIM, TIME, device=device) * 0.1
    a = torch.complex(a_real, a_imag).requires_grad_(True)
    
    # H_init [B, H, D]
    h_init = torch.randn(BATCH, HEADS, DIM, device=device, dtype=torch.complex64, requires_grad=True)

    # Clones Ref
    u_ref = u.clone().detach().requires_grad_(True)
    a_ref = a.clone().detach().requires_grad_(True)
    h_ref = h_init.clone().detach().requires_grad_(True)

    # 1. Forward Pass
    print("Calcul Forward...")
    # Le kernel attend [B, H, D, L] et gère le reste
    y_ours = parallel_complex_ssm(u, a, h_init, CHUNK)
    y_ref = naive_multihead_ssm(u_ref, a_ref, h_ref)
    
    # Check Difference
    diff = (y_ours - y_ref).abs().max()
    is_fwd_ok = diff < 1e-3
    print(f"Forward Diff: {diff:.6f} {'✅' if is_fwd_ok else '❌'}")

    # 2. Backward Pass
    print("Calcul Backward (Loss = Re(Sum(y))...")
    loss_ours = y_ours.real.sum()
    loss_ref = y_ref.real.sum()
    
    loss_ours.backward()
    loss_ref.backward()

    # 3. Validation Gradients
    grads_ok = True
    for name, g_ours, g_ref in [
        ("U", u.grad, u_ref.grad),
        ("A", a.grad, a_ref.grad),
        ("H_init", h_init.grad, h_ref.grad)
    ]:
        if g_ours is None or g_ref is None:
            print(f"Grad {name} manquant ❌")
            grads_ok = False
            continue
            
        err = (g_ours - g_ref).abs().max()
        tol = 5e-2 if name == "A" else 1e-3
        
        status = "✅" if err < tol else "❌"
        print(f"Grad {name} Diff: {err:.6f} {status}")
        if err > tol: grads_ok = False

    if is_fwd_ok and grads_ok:
        print("\n>>> SUCCESS: MULTI-HEAD COMPLEX ALGO VALIDÉ <<<")
    else:
        print("\n>>> FAILURE <<<")

    # --- BENCHMARK ---
    print(f"\n{'='*60}")
    print("=== BENCHMARK MULTI-HEAD ===")
    print(f"{'='*60}")
    B_B, H_B, D_B, T_B = BATCH, HEADS, DIM, TIME
    
    u_b = torch.randn(B_B, H_B, D_B, T_B, device=device, dtype=torch.complex64, requires_grad=True)
    a_b = torch.randn(B_B, H_B, D_B, T_B, device=device, dtype=torch.complex64, requires_grad=True)
    h_b = torch.randn(B_B, H_B, D_B, device=device, dtype=torch.complex64, requires_grad=True)
    
    # Warmup
    for _ in range(3):
        y = parallel_complex_ssm(u_b, a_b, h_b, CHUNK)
        y.real.sum().backward()
    torch.cuda.synchronize()
    
    start = time.time()
    ITER = 20
    for _ in range(ITER):
        y = parallel_complex_ssm(u_b, a_b, h_b, CHUNK)
        y.real.sum().backward()
        u_b.grad = None; a_b.grad = None; h_b.grad = None
    torch.cuda.synchronize()
    
    dt = (time.time() - start) / ITER
    print(f"Temps moyen (Fwd+Bwd MH): {dt*1000:.2f} ms")
    
    # Throughput metrics
    # Elements processed per second (Total elements in sequence)
    total_elements = B_B * H_B * D_B * T_B
    throughput_elems = total_elements / dt
    
    print(f"Débit Eléments: {throughput_elems/1e6:.2f} M_elem/s")
    print(f"Débit Séquence: {1.0/dt:.1f} it/s")

if __name__ == "__main__":
    run_test()