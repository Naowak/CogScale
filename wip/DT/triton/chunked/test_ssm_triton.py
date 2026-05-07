import torch
import time
import sys
from libs.DT.triton.chunked.ssm_triton_chunked import parallel_ssm

# Import du fichier Triton
# try:
#     from parallel_ssm_scan import parallel_ssm
# except ImportError:
#     print("❌ 'parallel_ssm_scan.py' introuvable.")
#     sys.exit(1)

# --- REFERENCE PYTORCH (Lente mais exacte) ---
def naive_ssm_loop(u, a, h_init=None):
    B, D, L = u.shape
    h = torch.zeros((B, D), device=u.device, dtype=u.dtype) if h_init is None else h_init.clone()
    h_out = []
    # h_t = a_t * h_{t-1} + u_t
    for t in range(L):
        h = a[:,:,t] * h + u[:,:,t]
        h_out.append(h)
    return torch.stack(h_out, dim=2)

def run_test():
    if not torch.cuda.is_available():
        print("❌ CUDA non disponible.")
        return
        
    device = torch.device("cuda")
    torch.manual_seed(42)

    # Paramètres raisonnables pour validation
    BATCH, DIM, TIME, CHUNK = 8, 32, 2048, 512
    print(f"\n{'='*60}")
    print(f"=== TEST VALIDATION (Fwd+Bwd) | T={TIME} ===")
    print(f"{'='*60}")

    # --- INPUTS ---
    # Pour que .grad soit peuplé, les tenseurs doivent être des "leaf tensors".
    # On crée les données d'abord, puis on enable le gradient.
    
    # U: Input
    u_data = torch.randn(BATCH, DIM, TIME, device=device)
    u = u_data.clone().detach().requires_grad_(True)
    
    # A: Poids récurrents (proche de 1 pour stabilité)
    a_data = torch.rand(BATCH, DIM, TIME, device=device) * 0.9 + 0.05
    a = a_data.clone().detach().requires_grad_(True)
    
    # H_init: État initial
    h_data = torch.randn(BATCH, DIM, device=device)
    h_init = h_data.clone().detach().requires_grad_(True)

    # --- CLONES DE RÉFÉRENCE ---
    u_ref = u.clone().detach().requires_grad_(True)
    a_ref = a.clone().detach().requires_grad_(True)
    h_ref = h_init.clone().detach().requires_grad_(True)

    # 1. Forward Pass
    print("Calcul Forward...")
    y_ours = parallel_ssm(u, a, h_init, CHUNK)
    y_ref = naive_ssm_loop(u_ref, a_ref, h_ref)
    
    diff = (y_ours - y_ref).abs().max()
    is_fwd_ok = diff < 1e-3
    print(f"Forward Diff: {diff:.6f} {'✅' if is_fwd_ok else '❌'}")

    # 2. Backward Pass
    print("Calcul Backward...")
    # Loss arbitraire mais qui dépend de tout le monde
    loss_ours = (y_ours * y_ours).mean() 
    loss_ref = (y_ref * y_ref).mean()
    
    loss_ours.backward()
    loss_ref.backward()

    # 3. Validation Gradients
    grads_ok = True
    for name, g_ours, g_ref in [
        ("U", u.grad, u_ref.grad),
        ("A", a.grad, a_ref.grad),
        ("H_init", h_init.grad, h_ref.grad)
    ]:
        if g_ours is None:
            print(f"Grad {name} manquant dans Triton ❌")
            grads_ok = False
            continue
        if g_ref is None:
            print(f"Grad {name} manquant dans Ref ❌")
            grads_ok = False
            continue
            
        err = (g_ours - g_ref).abs().max()
        # A est multiplicatif et récurrent, l'erreur s'accumule plus vite
        tol = 5e-2 if name == "A" else 1e-3 
        
        status = "✅" if err < tol else "❌"
        print(f"Grad {name} Diff: {err:.6f} {status}")
        
        if err > tol: 
            grads_ok = False
            idx = (g_ours - g_ref).abs().argmax()
            print(f"  -> Max err idx: {idx}, Ours: {g_ours.flatten()[idx]:.4f}, Ref: {g_ref.flatten()[idx]:.4f}")

    if is_fwd_ok and grads_ok:
        print("\n>>> SUCCESS: ALGORITHME VALIDÉ (Fwd + Bwd) <<<")
    else:
        print("\n>>> FAILURE: VERIFIER IMPLEMENTATION <<<")

    # --- BENCHMARK ---
    print(f"\n{'='*60}")
    print("=== BENCHMARK (Fwd + Bwd) ===")
    print(f"{'='*60}")
    B_B, D_B, T_B = 8, 256, 16384 # 16k context length
    print(f"Config: B={B_B}, D={D_B}, T={T_B}, Chunk={CHUNK}")
    
    # Données Bench
    u_b = torch.randn(B_B, D_B, T_B, device=device, requires_grad=True)
    a_b = torch.rand(B_B, D_B, T_B, device=device, requires_grad=True)
    h_b = torch.randn(B_B, D_B, device=device, requires_grad=True)

    # Warmup
    print("Warmup Triton...")
    for _ in range(5):
        y = parallel_ssm(u_b, a_b, h_b, CHUNK)
        y.sum().backward()
    torch.cuda.synchronize()

    print("Benchmarking Triton...")
    start = time.time()
    ITER = 50
    for _ in range(ITER):
        y = parallel_ssm(u_b, a_b, h_b, CHUNK)
        y.sum().backward()
        
        # Reset grad pour éviter l'accumulation (bien que pas grave pour le temps)
        u_b.grad = None; a_b.grad = None; h_b.grad = None
        
    torch.cuda.synchronize()
    
    dt_triton = (time.time() - start) / ITER
    print(f"Temps moyen Triton (Fwd+Bwd): {dt_triton*1000:.2f} ms")
    
    # Métriques SSM (Memory Bound généralement)
    # Données lues/écrites par itération (Fwd + Bwd approx 3 passes)
    # Size = B * D * T * 4 bytes (float32)
    size_gb = (B_B * D_B * T_B * 4) / 1e9
    # On lit U, A, H_init, on écrit H. Puis Backward... 
    # Approx 8-10x la taille du tenseur global en mouvement mémoire
    bandwidth = (size_gb * 8) / dt_triton 
    
    print(f"Débit Séquence: {1.0/dt_triton:.1f} it/s")
    print(f"Bande passante approx: {bandwidth:.2f} GB/s")

    # --- COMPARAISON NAIVE ---
    print(f"\n{'='*60}")
    print("=== COMPARAISON AVEC PYTORCH NAIF (BOUCLE) ===")
    print(f"{'='*60}")
    
    # On réduit le nombre d'itérations pour le naif car c'est très lent
    ITER_NAIVE = 5
    print(f"Lancement de {ITER_NAIVE} itérations pour le benchmark Naif (soyez patient)...")
    
    # Warmup Naif (1 seule iter pour compiler si besoin)
    naive_ssm_loop(u_b, a_b, h_b).sum().backward()
    torch.cuda.synchronize()
    
    start_naive = time.time()
    for _ in range(ITER_NAIVE):
        y = naive_ssm_loop(u_b, a_b, h_b)
        y.sum().backward()
        # Reset grad
        u_b.grad = None; a_b.grad = None; h_b.grad = None
    torch.cuda.synchronize()
    
    dt_naive = (time.time() - start_naive) / ITER_NAIVE
    print(f"Temps moyen Naif (Fwd+Bwd):   {dt_naive*1000:.2f} ms")
    print(f"Temps moyen Triton (Fwd+Bwd): {dt_triton*1000:.2f} ms")
    
    speedup = dt_naive / dt_triton
    print(f"\n>>> ACCELERATION (SPEEDUP): {speedup:.2f}x <<<")

if __name__ == "__main__":
    run_test()