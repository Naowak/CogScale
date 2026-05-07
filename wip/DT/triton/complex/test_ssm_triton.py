import torch
import time
import sys
from libs.DT.triton.complex.ssm_triton_complex import parallel_complex_ssm

# --- REFERENCE PYTORCH COMPLEXE ---
def naive_complex_ssm_loop(u, a, h_init=None):
    B, D, L = u.shape
    h = torch.zeros((B, D), device=u.device, dtype=u.dtype) if h_init is None else h_init.clone()
    h_out = []
    
    for t in range(L):
        # Opération complexe native PyTorch
        h = a[:,:,t] * h + u[:,:,t]
        h_out.append(h)
    return torch.stack(h_out, dim=2)

def run_test():
    if not torch.cuda.is_available():
        print("❌ CUDA non disponible.")
        return
        
    device = torch.device("cuda")
    torch.manual_seed(42)

    BATCH, DIM, TIME, CHUNK = 8, 64, 32000, 4096
    print(f"\n{'='*60}")
    print(f"=== TEST VALIDATION COMPLEXE (Fwd+Bwd) | T={TIME} ===")
    print(f"{'='*60}")

    # --- INPUTS COMPLEXES ---
    # U = (Rand + i Rand)
    u_real = torch.randn(BATCH, DIM, TIME, device=device)
    u_imag = torch.randn(BATCH, DIM, TIME, device=device)
    u = torch.complex(u_real, u_imag).requires_grad_(True)
    
    # A = (Rand + i Rand) - Normalisation pour éviter explosion
    a_real = torch.rand(BATCH, DIM, TIME, device=device) * 0.9 + 0.05
    a_imag = torch.rand(BATCH, DIM, TIME, device=device) * 0.1
    a = torch.complex(a_real, a_imag).requires_grad_(True)
    
    # H_init - Correction c64 -> complex64
    h_init = torch.randn(BATCH, DIM, device=device, dtype=torch.complex64, requires_grad=True)

    # Clones Ref
    u_ref = u.clone().detach().requires_grad_(True)
    a_ref = a.clone().detach().requires_grad_(True)
    h_ref = h_init.clone().detach().requires_grad_(True)

    # 1. Forward Pass
    print("Calcul Forward...")
    y_ours = parallel_complex_ssm(u, a, h_init, CHUNK)
    y_ref = naive_complex_ssm_loop(u_ref, a_ref, h_ref)
    
    # Check Difference
    diff = (y_ours - y_ref).abs().max()
    is_fwd_ok = diff < 1e-3
    print(f"Forward Diff: {diff:.6f} {'✅' if is_fwd_ok else '❌'}")

    # 2. Backward Pass
    print("Calcul Backward (Loss = Re(Sum(y))...")
    # "retour dans les réels pour l'output... en prenant juste la partie réelle"
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
        
        if err > tol:
            grads_ok = False
            # Debug info pour comprendre où ça diverge
            idx = (g_ours - g_ref).abs().argmax()
            print(f"  -> Max err idx: {idx}, Ours: {g_ours.flatten()[idx]:.4f}, Ref: {g_ref.flatten()[idx]:.4f}")


    if is_fwd_ok and grads_ok:
        print("\n>>> SUCCESS: COMPLEX ALGO VALIDÉ <<<")
    else:
        print("\n>>> FAILURE <<<")

    # --- BENCHMARK ---
    print(f"\n{'='*60}")
    print("=== BENCHMARK COMPLEXE ===")
    print(f"{'='*60}")
    B_B, D_B, T_B = BATCH, DIM, TIME # Un peu moins long que float32 car c64 = 2x mémoire
    
    u_b = torch.randn(B_B, D_B, T_B, device=device, dtype=torch.complex64, requires_grad=True)
    a_b = torch.randn(B_B, D_B, T_B, device=device, dtype=torch.complex64, requires_grad=True)
    h_b = torch.randn(B_B, D_B, device=device, dtype=torch.complex64, requires_grad=True)
    
    # Warmup
    for _ in range(5):
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
    print(f"Temps moyen (Fwd+Bwd Complex): {dt*1000:.2f} ms")
    print(f"Débit: {1.0/dt:.1f} it/s")

if __name__ == "__main__":
    run_test()