import sys
sys.path.append("libs/DT/triton/parallel/")
import torch
import time
from libs.DT.triton.parallel.ssm_triton_par import custom_ssm_parallel
from libs.DT.triton.parallel.ssm_triton_par import custom_ssm_forward as custom_ssm_sequential # Ton ancien fichier

def naive_ssm_pytorch(X, A, B, C):
    U = torch.einsum('bti,btin->btn', X, B)
    h_curr = torch.zeros(X.size(0), A.size(2), device=X.device)
    H = []
    for t in range(X.size(1)):
        h_curr = A[:, t, :] * h_curr + U[:, t, :]
        H.append(h_curr)
    H = torch.stack(H, dim=1)
    return torch.matmul(H, C)

def run_test():
    if not torch.cuda.is_available(): return
    device = torch.device("cuda")
    torch.manual_seed(42)

    # Config Longue Séquence pour voir l'intérêt du parallèle
    BATCH = 10
    TIME = 8192 # Doit être <= BLOCK_SIZE max du GPU pour cette démo simple (souvent 4096 ou 8192)
    D_IN = 16
    D_STATE = 128
    D_OUT = 16

    print(f"--- Config: T={TIME}, B={BATCH}, D={D_STATE} ---")
    
    X = torch.randn(BATCH, TIME, D_IN, device=device, requires_grad=True)
    A = torch.rand(BATCH, TIME, D_STATE, device=device, requires_grad=True)
    B = torch.randn(BATCH, TIME, D_IN, D_STATE, device=device, requires_grad=True)
    C = torch.randn(D_STATE, D_OUT, device=device, requires_grad=True)

    # 1. Check Correctness Forward
    y_para = custom_ssm_parallel(X, A, B, C)
    with torch.no_grad():
        y_ref = naive_ssm_pytorch(X, A, B, C)
    
    diff = (y_para - y_ref).abs().max()
    print(f"Différence Forward Parallèle vs Naive: {diff:.6f}")
    assert diff < 1e-3, "Erreur trop grande dans le scan parallèle"

    # 2. Check Backward
    loss = y_para.sum()
    loss.backward()
    print("Backward passé sans erreur.")
    
    # 3. Benchmark
    # On compare la version Séquentielle (V1) et Parallèle (V2)
    print("\n--- Benchmark ---")
    X_b = torch.randn(BATCH, TIME, D_IN, device=device)
    A_b = torch.rand(BATCH, TIME, D_STATE, device=device)
    B_b = torch.randn(BATCH, TIME, D_IN, D_STATE, device=device)
    
    # Warmup
    for _ in range(10): custom_ssm_parallel(X_b, A_b, B_b, C)
    
    # Mesure Parallel
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        custom_ssm_parallel(X_b, A_b, B_b, C)
    torch.cuda.synchronize()
    t_para = (time.time() - start) / 100
    
    # Mesure Sequential Fused (V1)
    # Assure-toi que ssm_triton.py est dans le dossier
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        custom_ssm_sequential(X_b, A_b, B_b, C)
    torch.cuda.synchronize()
    t_seq = (time.time() - start) / 100
    
    print(f"Temps Scan Parallèle (Triton V2): {t_para*1000:.3f} ms")
    print(f"Temps Scan Séquentiel (Triton V1): {t_seq*1000:.3f} ms")

if __name__ == "__main__":
    run_test()