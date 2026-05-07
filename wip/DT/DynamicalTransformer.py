import torch
from DT.MemoryLayer import MemoryLayer
from DT.AttentionLayer import AttentionLayer
from DT.FeedForwardLayer import FeedForwardLayer
from torch.utils.checkpoint import checkpoint

class DynamicalTransformer(torch.nn.Module):
    def __init__(self, num_layers=1, memory_units=4, memory_dim=64, attention_dim=16, attention_heads=4, 
                 d_conv=4, dropout=0.0, memory_connectivity=0.1, input_dim=None, output_dim=None,
                 device='cpu', dtype=torch.float32, complex_dtype=torch.complex64):
        """Initialize the DynamicalTransformer module."""
        super(DynamicalTransformer, self).__init__()

        # Store hyper-parameters
        self.num_layers = num_layers
        self.memory_units = memory_units
        self.memory_dim = memory_dim
        self.attention_dim = attention_dim
        self.attention_heads = attention_heads
        self.d_conv = d_conv
        self.dropout = dropout
        self.memory_connectivity = memory_connectivity
        self.device = device
        self.dtype = dtype
        self.complex_dtype = complex_dtype
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Initialize layers
        self.input_projection = None # [I, D] Embedding Layer
        self.output_projection = None # [M*D, O]
        self.layers = torch.nn.ModuleList()

        # Optimizer & Loss
        self.optimizer = None
        self.criterion = None
        
        self.norm = torch.nn.RMSNorm(attention_dim, eps=1e-8, device=device, dtype=dtype)

        # Dropout & Norm
        if self.num_layers > 0:
             self.dropout_layer = torch.nn.Dropout(dropout)

        self._define_model()


    def forward(self, x, state=None, return_state=False, chunk_size=256, grad_checkpoint=False, aim=None):
        """
        Args:
            x: Input Embedding tensor of shape (B, T, I) where B is batch size, T is sequence length, I is input dimension
            state: Optional state tensor for encoder-decoder attention of shape (B, T, M, R)
        Returns:
            output: Tensor of shape (B, T, O) where O is output dimension
        """
        # Input projection
        step = self.input_projection(x) # (B, T, D)

        # Pass through layers
        for layer in self.layers:

            # Memory Layer
            if layer.layer_type == 'MemoryLayer':
                if grad_checkpoint:
                    memory, state = checkpoint(layer.forward, step, state=state, chunk_size=chunk_size, aim=aim, use_reentrant=False) # (B, T, M, D), (B, T, M, R)
                else:
                    memory, state = layer(step, state=state, chunk_size=chunk_size, aim=aim) # (B, T, M, D), (B, T, M, R)
                step = step.unsqueeze(2) + memory # (B, T, M, D)
                # memory = step.unsqueeze(2) + memory # (B, T, M, D)
                # step = memory # (B, T, M, D) Residual connection for memory layer
            
            # Attention Layer
            elif layer.layer_type == 'AttentionLayer':
                if grad_checkpoint:
                    out = checkpoint(layer.forward, step, state=memory, aim=aim, use_reentrant=False) # (B, T, M, D)
                else:
                    out = layer(step, state=memory, aim=aim) # (B, T, M, D)
                step = step + self.dropout_layer(out) # (B, T, M, D)

            # Feed-Forward Layer
            elif layer.layer_type == 'FeedForwardLayer':
                if grad_checkpoint:
                    out = checkpoint(layer.forward, step, aim=aim, use_reentrant=False) # (B, T, M, D)
                else:
                    out = layer(step, aim=aim) # (B, T, M, D)
                step = step + self.dropout_layer(out) # (B, T, M, D)

            # Unknown Layer
            else:
                raise ValueError("Unknown layer type in DynamicalTransformer.")

        # Output projection
        step = self.norm(step) # (B, T, M, D)
        step = step.reshape(step.shape[0], step.shape[1], -1) # (B, T, M*D)
        step = torch.nn.functional.silu(self.pre_output(step)) # [B, T, D]
        # step = self.pre_output(step) # [B, T, D]
        output = self.output_projection(step) # (B, T, O)

        if return_state:
            return output, state # (B, T, O), (B, T, M, R)
        return output # (B, T, O)

    def _define_model(self):
        """Define the model architecture, optimizer, and loss function."""
        # Input & Output projection
        self.input_projection = torch.nn.Embedding(self.input_dim, self.attention_dim, device=self.device)
        self.pre_output = torch.nn.Linear(self.memory_units * self.attention_dim, self.attention_dim, device=self.device, dtype=self.dtype)
        self.output_projection = torch.nn.Linear(self.attention_dim, self.output_dim, device=self.device, dtype=self.dtype)

        # Create Memory Layer
        memory_layer = MemoryLayer(
            units=self.memory_units,
            neurons=self.memory_dim,
            input_dim=self.attention_dim,
            output_dim=self.attention_dim,
            d_conv=self.d_conv,
            input_connectivity=self.memory_connectivity,
            res_connectivity=self.memory_connectivity,
            device=self.device,
            dtype=self.dtype,
            complex_dtype=self.complex_dtype
        )
        self.layers.append(memory_layer)

        # Create Memory-Decoder Attention Layer
        for i in range(self.num_layers):
            attention_layer = AttentionLayer(
                d_model=self.attention_dim,
                num_heads=self.attention_heads,
                nb_layers=self.num_layers,
                dropout=self.dropout,
                self_attention=i==0, # First layer is self-attention
                lid=i,
                dtype=self.dtype,
                device=self.device
            )
            feedforward_layer = FeedForwardLayer(
                d_model=self.attention_dim,
                d_ff=4*self.attention_dim,
                nb_layers=self.num_layers,
                dropout=self.dropout,
                lid=i,
                device=self.device,
                dtype=self.dtype
            )
            self.layers.append(attention_layer)
            self.layers.append(feedforward_layer)

        # print(f"Model defined with {self._count_params():_} parameters.")

    def to(self, device):
        """Move the model to the specified device."""
        self.device = device
        super(DynamicalTransformer, self).to(device)

        self.input_projection = self.input_projection.to(device)
        self.pre_output = self.pre_output.to(device)
        self.output_projection = self.output_projection.to(device)
        self.norm = self.norm.to(device)

        layers = []
        for i, layer in enumerate(self.layers):
            layers.append(layer.to(device))
        self.layers = torch.nn.ModuleList(layers)
        return self
    
    def save(self, path):
        """Save the model state to the specified path."""
        # pickle.dump(self, open(path, 'wb'))
        torch.save(self, path)

    @staticmethod
    def load(path, device='cpu'):
        model = torch.load(path, map_location=device)
        return model

    def _count_params(self):
        """Count the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters()) # if p.requires_grad
    
    def __repr__(self):
        txt = "DynamicalTransformer("
        txt += f"\n  Input Projection: Embedding{tuple((self.input_dim, self.attention_dim))},"
        for layer in self.layers:
            txt += f"\n  Layer: {layer.__repr__()}"
        txt += f"\n  Norm: RMSNorm{tuple(self.norm.weight.shape)},"
        txt += f"\n  Pre-Output Projection: Tensor{tuple((self.memory_units * self.attention_dim, self.attention_dim))},"
        txt += f"\n  Output Projection: Tensor{tuple((self.attention_dim * self.memory_units, self.output_dim))},"
        txt += "\n)"
        return txt
    
     # def run_training(self, vocab_size, X_train, Y_train, X_valid=None, Y_valid=None,
    #                  epochs=100, batch_size=32, learning_rate=1e-3, weight_decay=1e-5, classification=False,
    #                  patience=5, min_delta=1e-5, path=None, chunk_size=256):
    #     """Run the training loop for the model."""
    #     # Convert data to tensors
    #     # X_train = torch.tensor(X_train, dtype=torch.long, device=self.device) # Embeddings
    #     # Y_train = torch.tensor(Y_train, dtype=self.dtype, device=self.device)

    #     # if X_valid is not None and Y_valid is not None:
    #     #     X_valid = torch.tensor(X_valid, dtype=torch.long, device=self.device)
    #     #     Y_valid = torch.tensor(Y_valid, dtype=self.dtype, device=self.device)

    #     X_train = X_train.to(device=self.device)
    #     Y_train = Y_train.to(device=self.device)

    #     if X_valid is not None and Y_valid is not None:
    #         X_valid = X_valid.to(device=self.device)
    #         Y_valid = Y_valid.to(device=self.device)

    #     # Define model if not already defined
    #     if self.input_dim is None or self.output_dim is None:
    #         self._define_model(
    #             input_dim=vocab_size,
    #             output_dim=vocab_size,
    #             learning_rate=learning_rate,
    #             weight_decay=weight_decay,
    #             classification=classification
    #         )
    #     print(self)
        
    #     # Create DataLoader
    #     train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    #     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    #     if X_valid is not None and Y_valid is not None:
    #         valid_dataset = torch.utils.data.TensorDataset(X_valid, Y_valid)
    #         valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    #     # Loop variables
    #     history = {'train_loss': [], 'val_loss': []}
    #     patience_counter = 0
    #     best_val_loss = float('inf')

    #     # Training loop
    #     for epoch in range(epochs):
    #         self.train()
    #         epoch_loss = 0.0

    #         # For each batch
    #         for X_batch, Y_batch in train_loader:
    #             # Reset gradients
    #             self.optimizer.zero_grad()
                
    #             # Forward pass
    #             outputs = self.forward(X_batch, chunk_size=chunk_size, grad_checkpoint=True) # (B, T, O)

    #             # Compute loss & backpropagate
    #             loss = self._compute_loss(Y_batch, outputs)
    #             loss.backward()
    #             self.optimizer.step()
    #             epoch_loss += loss.item() * X_batch.size(0)

    #         # Compute average loss & store
    #         epoch_loss /= len(train_loader.dataset)
    #         history['train_loss'].append(epoch_loss)

    #         # Validation
    #         if X_valid is not None and Y_valid is not None:
    #             self.eval()
    #             with torch.no_grad():
                    
    #                 losses = []
    #                 for X_valid_batch, Y_valid_batch in valid_loader:
    #                     # Forward pass
    #                     val_outputs = self.forward(X_valid_batch, chunk_size=chunk_size) # (B, T, O)

    #                     # Compute validation loss
    #                     val = self._compute_loss(Y_valid_batch, val_outputs)
    #                     losses.append(val.item() * X_valid_batch.size(0))

    #                 val_loss = sum(losses) / len(valid_loader.dataset)
    #                 history['val_loss'].append(val_loss)
                
    #             # Early stopping check
    #             if val_loss < best_val_loss - min_delta:
    #                 best_val_loss = val_loss
    #                 patience_counter = 0
    #                 # Save model
    #                 if path is not None:
    #                     self.save(path)
    #             else:
    #                 patience_counter += 1
    #                 if patience_counter >= patience:
    #                     print(f"Stop {epoch+1}, Best val loss: {best_val_loss:.4f}, Path: {path}")
    #                     break

    #         # Print progress
    #         print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}", end='')
    #         if X_valid is not None and Y_valid is not None:
    #             print(f", Val Loss: {val_loss:.4f}")
    #         else:
    #             print()
        
    #     return history


    # def run_inference(self, X, batch_size=32, state=None, return_state=False):
    #     """Run inference on the model."""
    #     # Convert data to tensor
    #     # X = torch.tensor(X, dtype=self.dtype, device=self.device)
    #     X = X.to(device=self.device)
    #     state = state.to(device=self.device) if state is not None else None

    #     # Create DataLoader
    #     dataset = torch.utils.data.TensorDataset(X)
    #     data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    #     # Inference loop
    #     self.eval()
    #     outputs_list = []
    #     states_list = []
    #     with torch.no_grad():
    #         for (X_batch,) in data_loader:
    #             # Forward pass
    #             if return_state:
    #                 outputs_batch, state = self.forward(X_batch, state=state, return_state=True)
    #                 states_list.append(state)
    #             else:
    #                 outputs_batch = self.forward(X_batch, state=state, return_state=False)
    #             outputs_list.append(outputs_batch.cpu())

    #     # Concatenate outputs
    #     outputs = torch.cat(outputs_list, dim=0) # (N, T, O)
    #     states = torch.cat(states_list, dim=0) if return_state else None

    #     if return_state:
    #         return outputs.numpy(), states.numpy()
    #     return outputs.numpy()

    # def _compute_loss(self, Y, outputs):
    #     """
    #     Calcule la CrossEntropyLoss directement.
        
    #     Args:
    #         Y: Tenseur des indices cibles. Shape: (Batch, Time)
    #         outputs: Logits du modèle. Shape: (Batch, Time, Vocab_Size)
    #     """
        
    #     # 1. On aplatit les dimensions Batch et Time ensemble
    #     # outputs devient : (Batch * Time, Vocab_Size)
    #     # C'est une opération "gratuite" (vue mémoire), pas de copie de données -> Pas de OOM
    #     logits_flat = outputs.view(-1, outputs.size(-1))
        
    #     # 2. On aplatit les cibles de la même façon
    #     # Y devient : (Batch * Time)
    #     targets_flat = Y.view(-1)
        
    #     # 3. Calcul de la loss
    #     # PyTorch compare chaque vecteur de logits (taille 50257) avec l'index cible correspondant
    #     loss = self.criterion(logits_flat, targets_flat.long())
        
    #     return loss



































