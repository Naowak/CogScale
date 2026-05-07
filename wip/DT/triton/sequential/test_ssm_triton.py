import torch
import time
from libs.DT.triton.sequential.ssm_triton_seq import custom_ssm_forward

# -------------------------------------------------------------------------
# IMPLEMENTATION DE REFERENCE (PYTORCH PUR)
# -------------------------------------------------------------------------
def naive_ssm_pytorch(X, A, B, C):
    """
    Implémentation lente mais lisible pour vérification.
    """
    batch_size, n_steps, _ = X.shape
    state_dim = A.shape[2]
    
    # 1. Compute U = X @ B
    # U[b, t, n] = sum_i (X[b, t, i] * B[b, t, i, n])
    U = torch.einsum('bti,btin->btn', X, B)
    
    # 2. Sequential Scan
    H = []
    h_curr = torch.zeros(batch_size, state_dim, device=X.device, dtype=X.dtype)
    
    for t in range(n_steps):
        u_t = U[:, t, :]      # [Batch, State]
        a_t = A[:, t, :]      # [Batch, State]
        
        # h_t = A_t * h_{t-1} + u_t
        h_curr = a_t * h_curr + u_t
        H.append(h_curr)
        
    H = torch.stack(H, dim=1) # [Batch, Time, State]
    
    # 3. Output Projection
    Y = torch.matmul(H, C)
    return Y

# -------------------------------------------------------------------------
# FONCTION DE TEST
# -------------------------------------------------------------------------
def run_test():
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        print("Ce test nécessite un GPU CUDA.")
        return

    device = torch.device("cuda")
    
    # Dimensions
    BATCH = 4
    TIME = 128
    D_IN = 32
    D_STATE = 64
    D_OUT = 16
    
    print(f"Configuration: Batch={BATCH}, Time={TIME}, In={D_IN}, State={D_STATE}, Out={D_OUT}")

    # Création des données
    # Nécessitent des gradients pour tester la backprop
    X = torch.randn(BATCH, TIME, D_IN, device=device, requires_grad=True)
    A = torch.rand(BATCH, TIME, D_STATE, device=device, requires_grad=True) # A entre 0 et 1 pour stabilité
    B = torch.randn(BATCH, TIME, D_IN, D_STATE, device=device, requires_grad=True)
    C = torch.randn(D_STATE, D_OUT, device=device, requires_grad=True)

    # Copie des données pour la version PyTorch (pour ne pas mélanger les graphes)
    X_ref = X.clone().detach().requires_grad_(True)
    A_ref = A.clone().detach().requires_grad_(True)
    B_ref = B.clone().detach().requires_grad_(True)
    C_ref = C.clone().detach().requires_grad_(True)

    # --- 1. Test Forward ---
    print("\n--- Test Forward ---")
    y_triton = custom_ssm_forward(X, A, B, C)
    y_ref = naive_ssm_pytorch(X_ref, A_ref, B_ref, C_ref)
    
    # Vérification d'erreur
    diff = (y_triton - y_ref).abs().max()
    print(f"Différence Max Forward: {diff.item():.6f}")
    assert torch.allclose(y_triton, y_ref, atol=1e-4), "Mismatch Forward!"
    print(">> Forward OK")

    # --- 2. Test Backward ---
    print("\n--- Test Backward ---")
    loss_triton = y_triton.sum()
    loss_triton.backward()
    
    loss_ref = y_ref.sum()
    loss_ref.backward()
    
    # Comparaison des gradients
    grads_to_check = [
        ("X", X.grad, X_ref.grad),
        ("A", A.grad, A_ref.grad),
        ("B", B.grad, B_ref.grad),
        ("C", C.grad, C_ref.grad)
    ]
    
    for name, g_tri, g_ref in grads_to_check:
        grad_diff = (g_tri - g_ref).abs().max()
        print(f"Différence Max Grad {name}: {grad_diff.item():.6f}")
        # Note: Les erreurs d'arrondi s'accumulent vite dans les scans, tolérance un peu plus haute
        if not torch.allclose(g_tri, g_ref, atol=1e-3, rtol=1e-3):
            print(f"!! ECHEC Gradient {name} !!")
        else:
            print(f">> Gradient {name} OK")

    print("\n--- Benchmark Rapide (Temps) ---")
    print("Config: Batch=4, Time=1024, In=32, State=64, Out=16")
    # Augmentons un peu la taille pour voir la différence
    TIME_BENCH = 1024
    X_b = torch.randn(BATCH, TIME_BENCH, D_IN, device=device)
    A_b = torch.rand(BATCH, TIME_BENCH, D_STATE, device=device)
    B_b = torch.randn(BATCH, TIME_BENCH, D_IN, D_STATE, device=device)
    
    # Warmup
    for _ in range(10): custom_ssm_forward(X_b, A_b, B_b, C)

    # PyTorch Time
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        naive_ssm_pytorch(X_b, A_b, B_b, C)
    torch.cuda.synchronize()
    pytorch_dur = (time.time() - start) / 100
    
    # Triton Time
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        custom_ssm_forward(X_b, A_b, B_b, C)
    torch.cuda.synchronize()
    triton_dur = (time.time() - start) / 100
    
    print(f"Temps moyen Triton (Time={TIME_BENCH}): {triton_dur*1000:.3f} ms")
    print(f"Temps moyen PyTorch (Time={TIME_BENCH}): {pytorch_dur*1000:.3f} ms")

if __name__ == "__main__":
    run_test()