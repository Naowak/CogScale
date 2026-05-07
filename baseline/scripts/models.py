# models.py
import math
import torch
import torch.nn as nn
import reservoirpy as rpy
from reservoirpy.nodes import Reservoir, Ridge
import numpy as np
import cogscale as cog
try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None
try:
    from xlstm import xLSTMBlockStack, xLSTMBlockStackConfig, mLSTMBlockConfig, sLSTMBlockConfig
except ImportError:
    xLSTMBlockStack = None
    xLSTMBlockStackConfig = None
    mLSTMBlockConfig = None
    sLSTMBlockConfig = None


# Désactiver les logs trop verbeux de reservoirpy pendant les expériences
rpy.verbosity(0)

class DynamicLSTM(nn.Module):
    def __init__(self, input_dim, output_dim, target_params):
        super(DynamicLSTM, self).__init__()
        
        # Résolution de l'équation du second degré pour trouver hidden_size
        # 4*H^2 + (4*I + 4 + O)*H + (O - target_params) = 0
        a = 4
        b = 4 * input_dim + 4 + output_dim
        c = output_dim - target_params
        
        delta = b**2 - 4*a*c
        if delta < 0:
            hidden_size = 1 # Fallback de sécurité, bien que rare
        else:
            hidden_size = int((-b + math.sqrt(delta)) / (2 * a))
            hidden_size = max(1, hidden_size) # Au moins 1 de dimension
            
        self.lstm = nn.LSTM(input_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_dim)
        
        self.actual_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        # x shape: [Batch, Time, Features]
        lstm_out, _ = self.lstm(x)
        logits = self.fc(lstm_out)
        return logits



class DynamicGRU(nn.Module):
    def __init__(self, input_dim, output_dim, target_params):
        super(DynamicGRU, self).__init__()
        
        # 3*H^2 + (3*I + 6 + O)*H + (O - target_params) = 0
        a = 3
        b = 3 * input_dim + 6 + output_dim
        c = output_dim - target_params
        
        delta = b**2 - 4*a*c
        if delta < 0:
            hidden_size = 1
        else:
            hidden_size = int((-b + math.sqrt(delta)) / (2 * a))
            hidden_size = max(1, hidden_size)
            
        self.gru = nn.GRU(input_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_dim)
        
        self.actual_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        # x shape: [Batch, Time, Features]
        gru_out, _ = self.gru(x)
        logits = self.fc(gru_out)
        return logits
    


class PositionalEncoding(nn.Module):
    """Encodage positionnel classique (Sin/Cos) pour donner la notion du temps au Transformer"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) # [1, max_len, d_model]

    def forward(self, x):
        # x shape: [Batch, Time, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return x

class DynamicTransformerDecoderOnly(nn.Module):
    """
    Architecture GPT-like : Uniquement le décodeur avec masque causal.
    Prédit de gauche à droite, simule l'approche autoregressive.
    """
    def __init__(self, input_dim, output_dim, target_params, num_layers=2, nhead=2):
        super(DynamicTransformerDecoderOnly, self).__init__()
        
        # Résolution pour trouver d_model (D)
        # P ≈ 12 * L * D^2 + (I + O) * D
        a = 12 * num_layers
        b = input_dim + output_dim
        c = -target_params
        
        delta = b**2 - 4*a*c
        if delta < 0:
            d_model = nhead # Fallback minimal
        else:
            d_model = int((-b + math.sqrt(delta)) / (2 * a))
            
        # Arrondir d_model pour qu'il soit divisible par nhead
        d_model = max(nhead, d_model - (d_model % nhead))
        
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Un Decoder-Only s'implémente souvent via un Encoder avec masque causal en Pytorch
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4*d_model, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, output_dim)
        
        self.actual_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        # x shape: [Batch, Time, Features]
        x_emb = self.embedding(x)
        x_emb = self.pos_encoder(x_emb)
        
        # Création du masque causal pour empêcher de "regarder dans le futur"
        seq_len = x.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        
        # Application du transformer
        out = self.transformer(x_emb, mask=causal_mask, is_causal=True)
        out = self.norm(out)
        logits = self.fc_out(out)
        return logits


class DynamicTransformerEncoderDecoder(nn.Module):
    """
    Architecture originale (Attention is All You Need - 2017).
    """
    def __init__(self, input_dim, output_dim, target_params, num_layers=2, nhead=2):
        super(DynamicTransformerEncoderDecoder, self).__init__()
        
        # Résolution pour trouver d_model (D)
        # L_enc + L_dec ≈ 28 * L * D^2 + (I + O) * D
        a = 28 * num_layers
        b = 2 * input_dim + output_dim # On utilise input_dim pour projeter src et tgt
        c = -target_params
        
        delta = b**2 - 4*a*c
        if delta < 0:
            d_model = nhead
        else:
            d_model = int((-b + math.sqrt(delta)) / (2 * a))
            
        d_model = max(nhead, d_model - (d_model % nhead))
        
        self.src_emb = nn.Linear(input_dim, d_model)
        self.tgt_emb = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead, 
            num_encoder_layers=num_layers, num_decoder_layers=num_layers,
            dim_feedforward=4*d_model, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, output_dim)
        
        self.actual_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        # Pour une tâche synchrone [B, T, I] -> [B, T, O] sans génération token par token,
        # on utilise X comme source ET cible (avec masque causal sur la cible).
        
        src = self.pos_encoder(self.src_emb(x))
        tgt = self.pos_encoder(self.tgt_emb(x))
        
        seq_len = x.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        
        out = self.transformer(src, tgt, tgt_mask=tgt_mask, tgt_is_causal=True)
        out = self.norm(out)
        logits = self.fc_out(out)
        return logits
    

class DynamicESN:
    def __init__(self, input_dim, output_dim, N, lr, sr, input_scaling):
        """
        N : Nombre de neurones dans le réservoir
        lr : Leaking rate
        sr : Spectral radius
        input_scaling : Échelle d'entrée
        """
        self.reservoir = Reservoir(N, lr=lr, sr=sr, input_scaling=input_scaling)
        self.readout = None # Sera instancié dynamiquement pendant le fit
        
        # Pour le logging (Readout linéaire uniquement)
        self.actual_params = (output_dim * N)

    def fit(self, X_train, Y_train, T_train, X_valid, Y_valid, T_valid, category):
        """
        Entraîne plusieurs readouts avec différents ridges et garde le meilleur sur le set de validation.
        """
        # 1. Extraction des états du réservoir pour le TRAIN
        states_train = []
        targets_train = []
        
        for i in range(len(X_train)):
            s = self.reservoir.run(X_train[i], reset=True) 
            t_idx = T_train[i]
            states_train.append(s[t_idx])
            targets_train.append(Y_train[i][t_idx])
            
        S_train_flat = np.vstack(states_train)
        Y_train_flat = np.vstack(targets_train)

        # 2. Extraction des états du réservoir pour la VALIDATION (calculé une seule fois !)
        S_valid_full = []
        for i in range(len(X_valid)):
            s = self.reservoir.run(X_valid[i], reset=True)
            S_valid_full.append(s)

        # 3. Recherche du meilleur paramètre Ridge
        best_score = float('inf')
        best_ridge = None
        best_readout = None
        
        # Gamme de 1e-1 à 1e-9
        ridges = [10**(-i) for i in range(1, 10)]
        
        for r in ridges:
            temp_readout = Ridge(ridge=r)
            temp_readout.fit(S_train_flat, Y_train_flat)
            
            # Génération des prédictions de validation avec ce readout temporaire
            preds_val = [temp_readout.run(s) for s in S_valid_full]
            preds_val_np = np.stack(preds_val, axis=0)
            
            # Évaluation avec la fonction native du package
            score = cog.compute_score(Y=Y_valid, Y_hat=preds_val_np, prediction_timesteps=T_valid, category=category)
            
            if score < best_score:
                best_score = score
                best_ridge = r
                best_readout = temp_readout
                
        # 4. On sauvegarde définitivement le meilleur Readout
        self.readout = best_readout
        print(f"    -> Meilleur Ridge trouvé : {best_ridge} (Val Score: {best_score:.4f})")

    def predict(self, X):
        """
        Génère les prédictions pour un ensemble de séquences.
        """
        preds = []
        for i in range(len(X)):
            s = self.reservoir.run(X[i], reset=True)
            y = self.readout.run(s)
            preds.append(y)
            
        return np.stack(preds, axis=0)
    




class DynamicMamba(nn.Module):
    def __init__(self, input_dim, output_dim, target_params, num_layers=2):
        super(DynamicMamba, self).__init__()
        if Mamba is None:
            raise ImportError("mamba_ssm n'est pas installé. Lancez: pip install mamba-ssm causal-conv1d")
        
        # Recherche binaire pour trouver le meilleur d_model
        low, high = 2, 2048
        best_d = 8
        best_diff = float('inf')
        
        while low <= high:
            mid = (low + high) // 2
            d_model = max(2, mid - (mid % 2)) # Pair pour éviter les soucis
            
            # Instanciation temporaire pour compter les paramètres
            temp_encoder = nn.Linear(input_dim, d_model)
            temp_layers = nn.ModuleList([Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) for _ in range(num_layers)])
            temp_norm = nn.LayerNorm(d_model)
            temp_decoder = nn.Linear(d_model, output_dim)
            
            params = (sum(p.numel() for p in temp_encoder.parameters()) +
                      sum(p.numel() for p in temp_layers.parameters()) +
                      sum(p.numel() for p in temp_norm.parameters()) +
                      sum(p.numel() for p in temp_decoder.parameters()))
                      
            if abs(params - target_params) < best_diff:
                best_diff = abs(params - target_params)
                best_d = d_model
                
            if params < target_params:
                low = mid + 1
            else:
                high = mid - 1
                
        # Instanciation finale avec le d_model optimal
        self.encoder = nn.Linear(input_dim, best_d)
        self.layers = nn.ModuleList([Mamba(d_model=best_d, d_state=16, d_conv=4, expand=2) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(best_d)
        self.decoder = nn.Linear(best_d, output_dim)
        
        self.actual_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        h = self.encoder(x)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        return self.decoder(h)


class DynamicxLSTM(nn.Module):
    def __init__(self, input_dim, output_dim, target_params, context_length, num_blocks=2):
        super(DynamicxLSTM, self).__init__()
        if xLSTMBlockStack is None:
            raise ImportError("xlstm n'est pas installé. Lancez: pip install xlstm")
            
        # On respecte strictement votre contrainte de couches
        self.num_blocks = num_blocks
        
        low, high = 4, 2048  # On permet de descendre très bas (4)
        best_d = 4
        best_diff = float('inf')
        
        while low <= high:
            mid = (low + high) // 2
            # xLSTM requiert souvent un d_model multiple de 4 pour diviser ses têtes d'attention
            d_model = max(4, mid - (mid % 4)) 
            
            try:
                # Instanciation temporaire (avec les configs par défaut pour éviter le crash)
                cfg = xLSTMBlockStackConfig(
                    mlstm_block=mLSTMBlockConfig(),
                    slstm_block=sLSTMBlockConfig(),
                    context_length=context_length, 
                    num_blocks=self.num_blocks,
                    embedding_dim=d_model,
                    slstm_at=[self.num_blocks // 2]  # On place le sLSTM au milieu
                )
                temp_stack = xLSTMBlockStack(cfg)
                temp_encoder = nn.Linear(input_dim, d_model)
                temp_norm = nn.LayerNorm(d_model)
                temp_decoder = nn.Linear(d_model, output_dim)
                
                params = (sum(p.numel() for p in temp_encoder.parameters()) +
                          sum(p.numel() for p in temp_stack.parameters()) +
                          sum(p.numel() for p in temp_norm.parameters()) +
                          sum(p.numel() for p in temp_decoder.parameters()))
                          
                if abs(params - target_params) < best_diff:
                    best_diff = abs(params - target_params)
                    best_d = d_model
                    
                if params < target_params:
                    low = mid + 1
                else:
                    high = mid - 1
            except Exception:
                # Si xLSTM refuse cette dimension exacte, on réduit la taille pour continuer à chercher
                high = mid - 1
                
        # Instanciation finale avec le meilleur d_model trouvé (le plus proche possible de la cible)
        self.encoder = nn.Linear(input_dim, best_d)
        cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(),
            slstm_block=sLSTMBlockConfig(),
            context_length=context_length,
            num_blocks=self.num_blocks,
            embedding_dim=best_d,
            slstm_at=[self.num_blocks // 2]
        )
        self.xlstm_stack = xLSTMBlockStack(cfg)
        self.norm = nn.LayerNorm(best_d)
        self.decoder = nn.Linear(best_d, output_dim)
        
        self.actual_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        h = self.encoder(x)
        h = self.xlstm_stack(h)
        h = self.norm(h)
        return self.decoder(h)
        