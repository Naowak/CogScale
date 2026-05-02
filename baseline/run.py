# run.py
import argparse
import itertools
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import copy
import re
import os

import sys
sys.path.append("../")
import stream_dataset as sd
from models import DynamicLSTM, DynamicGRU, DynamicTransformerDecoderOnly, \
    DynamicTransformerEncoderDecoder, DynamicESN, DynamicDynamicalTransformer, \
    DynamicMamba, DynamicxLSTM

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_report_configs(filepath="report.md"):
    """
    Parse le fichier markdown pour récupérer les hyperparamètres de l'ESN.
    """
    configs = {}
    if not os.path.exists(filepath):
        print(f"Attention: {filepath} non trouvé. L'ESN ne pourra pas charger ses hyperparamètres.")
        return configs
        
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Regex pour capturer "**tache.diff.Nxxx**" et le dictionnaire "```{'N'...}```"
    pattern = r'\*\*(.*?)\*\*\n.*?```(.*?)```'
    matches = re.findall(pattern, content, re.DOTALL)
    
    for key, dict_str in matches:
        try:
            configs[key] = eval(dict_str.strip())
        except Exception as e:
            print(f"Erreur de parsing pour {key}: {e}")
            
    return configs

def get_masked_loss(preds, targets, timesteps, category):
    B = timesteps.shape[0]
    O = preds.shape[-1]
    
    p_list = [preds[i, timesteps[i], :] for i in range(B)]
    t_list = [targets[i, timesteps[i], :] for i in range(B)]
        
    p_tensor = torch.stack(p_list)
    t_tensor = torch.stack(t_list)
    
    if category == 'classification':
        t_class = torch.argmax(t_tensor, dim=-1) 
        p_flat = p_tensor.view(-1, O)            
        t_flat = t_class.view(-1)                
        loss = nn.CrossEntropyLoss()(p_flat, t_flat)
        
    elif category == 'multi_classification':
        p_flat = p_tensor.view(-1, O)
        t_flat = t_tensor.view(-1, O)
        loss = nn.BCEWithLogitsLoss()(p_flat, t_flat)
        
    elif category == 'regression':
        loss = nn.MSELoss()(p_tensor, t_tensor)
        
    return loss

def find_best_threshold(preds_logits, targets, timesteps):
    B = timesteps.shape[0]
    p_list = [preds_logits[i, timesteps[i], :] for i in range(B)]
    t_list = [targets[i, timesteps[i], :] for i in range(B)]
    
    preds = np.stack(p_list, axis=0)
    truths = np.stack(t_list, axis=0)
    
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    preds_probs = sigmoid(preds)
    
    best_thresh = 0.5
    best_score = float('inf')
    
    thresholds = np.linspace(0.01, 0.99, 100)
    
    for t in thresholds:
        preds_bin = (preds_probs >= t).astype(int)
        correct_samples = np.all(preds_bin == truths, axis=(1, 2))
        score = 1 - np.mean(correct_samples)
        
        if score < best_score:
            best_score = score
            best_thresh = t
            
    return best_thresh

def run_experiment():
    parser = argparse.ArgumentParser(description="Run Stream Dataset Evaluation")
    parser.add_argument('--model_types', nargs='+', default=['lstm'], 
                        choices=['lstm', 'gru', 'transformer_decoder', 'transformer_encdec', 'esn', 'dynamical_transformer', 'mamba', 'xlstm'], 
                        help='Type de modèle à entrainer')
    parser.add_argument('--tasks', nargs='+', default=['all'], help='Liste des tâches')
    parser.add_argument('--difficulties', nargs='+', default=['small'], choices=['small', 'medium', 'large'], help='Niveaux de difficulté')
    parser.add_argument('--sizes', nargs='+', type=int, default=[1000, 10000], help='Tailles des paramètres visées')
    parser.add_argument('--lrs', nargs='+', type=float, default=[1e-2, 3e-3, 1e-3, 3e-4, 1e-4], help='Learning rates à tester (Ignoré par ESN)')
    parser.add_argument('--seeds', type=int, default=10, help='Nombre de seeds par run')
    parser.add_argument('--epochs', type=int, default=50, help='Nombre d\'epochs (Ignoré par ESN)')
    parser.add_argument('--batch_size', type=int, default=10, help='Taille du batch (Ignoré par ESN)')
    parser.add_argument('--patience', type=int, default=10, help='Nombre d\'epochs sans amélioration (Ignoré par ESN)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device (cuda/cpu)')
    parser.add_argument('--dtype', type=str, default='float32', help='Type de données Pytorch')
    parser.add_argument('--output', type=str, default='results.csv', help='Fichier CSV de sortie')
    
    args = parser.parse_args()

    if args.tasks == ['all']:
        args.tasks = sd.tasks
    elif args.tasks == ['first_half']:
        args.tasks = sd.tasks[:len(sd.tasks)//2]
    elif args.tasks == ['second_half']:
        args.tasks = sd.tasks[len(sd.tasks)//2:]
        
    # Charger les configs ESN si nécessaire
    esn_configs = parse_report_configs('results/esn_hp_opti/report.md') if 'esn' in args.model_types else {}
    
    torch_dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    
    lr_models = [args.model_types[i] for i in range(len(args.model_types)) if args.model_types[i] != 'esn']
    results = []
    combinations = list(itertools.product(lr_models, args.tasks, args.difficulties, args.sizes, args.lrs, range(args.seeds)))
    if 'esn' in args.model_types:
        combinations += list(itertools.product(['esn'], args.tasks, args.difficulties, args.sizes, [None], range(args.seeds)))
    
    for model_type, task_name, difficulty, size, lr, seed in combinations:
        print(f"\n--- Model: {model_type.upper()} | Task: {task_name} | Diff: {difficulty} | Size: {size} | LR: {lr} | Seed: {seed} ---")
        
        try:
            task_data = sd.build_task(task_name, difficulty=difficulty, seed=seed)
        except Exception as e:
            print(f"Erreur lors du chargement de {task_name} ({difficulty}): {e}")
            continue

        set_seed(seed)
        category = task_data['category']

        # -------------------

        if model_type == 'xlstm':
            print(f"  -> Note: Le modèle xLSTM est conçu pour gérer les séquences de longueurs variables sans padding. On padd.")
            # 1. Trouver la longueur maximale (généralement X_test)
            max_seq_len = max(task_data['X_train'].shape[1], task_data['X_valid'].shape[1], task_data['X_test'].shape[1])
            
            # 2. Fonction pour rajouter des zéros à la fin de la dimension temporelle (dim 1)
            def pad_to_max(arr, max_len):
                if arr.shape[1] < max_len:
                    return np.pad(arr, ((0, 0), (0, max_len - arr.shape[1]), (0, 0)), mode='constant')
                return arr

            # 3. Application du padding
            task_data['X_train'] = pad_to_max(task_data['X_train'], max_seq_len)
            task_data['X_valid'] = pad_to_max(task_data['X_valid'], max_seq_len)
            task_data['X_test'] = pad_to_max(task_data['X_test'], max_seq_len)
            
            task_data['Y_train'] = pad_to_max(task_data['Y_train'], max_seq_len)
            task_data['Y_valid'] = pad_to_max(task_data['Y_valid'], max_seq_len)
            task_data['Y_test'] = pad_to_max(task_data['Y_test'], max_seq_len)
            
            input_dim = task_data['X_train'].shape[-1]
            output_dim = task_data['Y_train'].shape[-1]
            
            best_val_score = None
            best_threshold = 0.5
            preds_test_np = None

        #--------------------

        # ----------------------------------------------------
        # BRANCHE 1 : ECHO STATE NETWORK (ReservoirPy)
        # ----------------------------------------------------
        if model_type == 'esn':
            config_key = f"{task_name}.{difficulty}.N{size//10}"
            if config_key not in esn_configs:
                print(f"  -> Configuration ESN introuvable pour '{config_key}' dans report.md. Skip.")
                continue
                
            hparams = esn_configs[config_key]
            
            # Initialisation ESN (Plus de paramètre ridge ici)
            model = DynamicESN(
                input_dim=input_dim, 
                output_dim=output_dim, 
                N=hparams['N'], 
                lr=hparams['lr'], 
                sr=hparams['sr'], 
                input_scaling=hparams['input_scaling']
            )
            print(f"Modèle ESN initialisé avec {model.actual_params} paramètres (N={hparams['N']}, lr={hparams['lr']:.3f}, sr={hparams['sr']:.3f}, is={hparams['input_scaling']:.3f}).")
            
            # Entraînement avec recherche automatique du Ridge sur la validation
            model.fit(
                X_train=task_data['X_train'], 
                Y_train=task_data['Y_train'], 
                T_train=task_data['T_train'],
                X_valid=task_data['X_valid'], 
                Y_valid=task_data['Y_valid'], 
                T_valid=task_data['T_valid'],
                category=category
            )
            
            # Optimisation du threshold (si multi_classification)
            if category == 'multi_classification':
                preds_val_np = model.predict(task_data['X_valid'])
                best_threshold = find_best_threshold(preds_val_np, task_data['Y_valid'], task_data['T_valid'])
                print(f"  -> Meilleur threshold trouvé sur la validation : {best_threshold:.2f}")

            # Prédiction Test
            preds_test_np = model.predict(task_data['X_test'])

        # ----------------------------------------------------
        # BRANCHE 2 : RÉSEAUX DE NEURONES (PyTorch)
        # ----------------------------------------------------
        else:
            # Préparation Tensors
            X_train = torch.tensor(task_data['X_train'], dtype=torch_dtype)
            Y_train = torch.tensor(task_data['Y_train'], dtype=torch_dtype)
            T_train = torch.tensor(task_data['T_train'], dtype=torch.long)
            
            X_valid = torch.tensor(task_data['X_valid'], dtype=torch_dtype).to(device)
            Y_valid_np = task_data['Y_valid']
            T_valid_np = task_data['T_valid']
            X_test = torch.tensor(task_data['X_test'], dtype=torch_dtype).to(device)
            
            train_dataset = TensorDataset(X_train, Y_train, T_train)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            
            # Instanciation
            if model_type == 'lstm':
                model = DynamicLSTM(input_dim=input_dim, output_dim=output_dim, target_params=size).to(device, dtype=torch_dtype)
            elif model_type == 'gru':
                model = DynamicGRU(input_dim=input_dim, output_dim=output_dim, target_params=size).to(device, dtype=torch_dtype)
            elif model_type == 'transformer_decoder':
                model = DynamicTransformerDecoderOnly(input_dim=input_dim, output_dim=output_dim, target_params=size).to(device, dtype=torch_dtype)
            elif model_type == 'transformer_encdec':
                model = DynamicTransformerEncoderDecoder(input_dim=input_dim, output_dim=output_dim, target_params=size).to(device, dtype=torch_dtype)
            elif model_type == 'dynamical_transformer':
                model = DynamicDynamicalTransformer(input_dim=input_dim, output_dim=output_dim, target_params=size).to(device, dtype=torch_dtype)
            elif model_type == 'mamba':
                model = DynamicMamba(input_dim=input_dim, output_dim=output_dim, target_params=size).to(device, dtype=torch_dtype)
            elif model_type == 'xlstm':
                model = DynamicxLSTM(input_dim=input_dim, output_dim=output_dim, target_params=size, context_length=task_data['X_train'].shape[1]).to(device, dtype=torch_dtype)
            else:
                print(f"Type de modèle inconnu: {model_type}. Skip.")
                continue

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            print(f"Modèle {model_type.upper()} initialisé avec ~{model.actual_params} paramètres (Cible: {size}).")
            
            best_val_score = float('inf')
            epochs_no_improve = 0
            best_model_state = None
            
            for epoch in range(args.epochs):
                model.train()
                for bx, by, bt in train_loader:
                    bx, by, bt = bx.to(device), by.to(device), bt.to(device)
                    optimizer.zero_grad()
                    preds = model(bx)
                    loss = get_masked_loss(preds, by, bt, category)
                    loss.backward()
                    optimizer.step()

                model.eval()
                with torch.no_grad():
                    preds_val_list = []
                    for i in range(0, len(X_valid), args.batch_size):
                        batch_val = X_valid[i:i+args.batch_size]
                        preds_val_list.append(model(batch_val).cpu().numpy())
                    preds_val_np = np.concatenate(preds_val_list, axis=0)
                    
                val_score = sd.compute_score(Y=Y_valid_np, Y_hat=preds_val_np, prediction_timesteps=T_valid_np, category=category)
                
                if val_score < best_val_score:
                    best_val_score = val_score
                    epochs_no_improve = 0
                    best_model_state = copy.deepcopy(model.state_dict())
                else:
                    epochs_no_improve += 1
                    
                if args.patience > 0 and epochs_no_improve >= args.patience:
                    print(f"  -> Early stopping déclenché à l'epoch {epoch+1}/{args.epochs} (Meilleur score val: {best_val_score:.4f})")
                    break
                    
            if args.patience > 0 and best_model_state is not None:
                model.load_state_dict(best_model_state)
                    
            if category == 'multi_classification':
                model.eval()
                with torch.no_grad():
                    preds_val_list = []
                    for i in range(0, len(X_valid), args.batch_size):
                        batch_val = X_valid[i:i+args.batch_size]
                        preds_val_list.append(model(batch_val).cpu().numpy())
                    preds_val_np = np.concatenate(preds_val_list, axis=0)
                best_threshold = find_best_threshold(preds_val_np, Y_valid_np, T_valid_np)
                print(f"  -> Meilleur threshold trouvé sur la validation : {best_threshold:.2f}")

            model.eval()
            with torch.no_grad():
                preds_test_list = []
                for i in range(0, len(X_test), args.batch_size):
                    batch_test = X_test[i:i+args.batch_size]
                    preds_test_list.append(model(batch_test).cpu().numpy())
                preds_test_np = np.concatenate(preds_test_list, axis=0)

        # ----------------------------------------------------
        # ÉVALUATION FINALE (Commun aux deux branches)
        # ----------------------------------------------------
        score = sd.compute_score(
            Y=task_data['Y_test'],
            Y_hat=preds_test_np,
            prediction_timesteps=task_data['T_test'],
            category=category,
            threshold=best_threshold
        )
        print(f"Score de Test final: {score:.4f}")
        
        results.append({
            'Model': model_type.upper(),
            'Task': task_name,
            'Difficulty': difficulty,
            'Target_Params': size,
            'Actual_Params': model.actual_params,
            'LR': lr if model_type != 'esn' else 'N/A',
            'Seed': seed,
            'Category': category,
            'Best_Val_Score': best_val_score,
            'Test_Score': score
        })
        
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"\nExpériences terminées ! Résultats sauvegardés dans {args.output}")

if __name__ == "__main__":
    run_experiment()