"""
Conditional Variational Autoencoder (CVAE) for Single-Cell RNA Analysis

This module implements a Conditional Variational Autoencoder specifically designed for
single-cell RNA sequencing data. The CVAE can learn the underlying distribution of
gene expression patterns and generate synthetic cells conditioned on specific factors
like cell type, disease state, or experimental conditions.

Key Features:
- Conditional generation based on cell type, disease state, etc.
- Disentangled latent space representation
- Generation of synthetic cells for data augmentation
- Anomaly detection in gene expression patterns
- Interpolation between different cell states
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Optional, Dict, Any, Union
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Encoder(nn.Module):
    """
    Encoder network for the CVAE.
    
    Maps input gene expression and conditioning variables to the parameters
    of the latent space distribution (mean and log variance).
    """
    
    def __init__(self, 
                 input_dim: int,
                 condition_dim: int,
                 hidden_dims: List[int] = [512, 256, 128],
                 latent_dim: int = 64,
                 dropout: float = 0.2):
        """
        Initialize the encoder.
        
        Args:
            input_dim: Dimension of gene expression input
            condition_dim: Dimension of conditioning variables
            hidden_dims: List of hidden layer dimensions
            latent_dim: Dimension of latent space
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim
        
        # Input processing layers
        layers = []
        current_dim = input_dim + condition_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Latent space parameters
        self.fc_mu = nn.Linear(current_dim, latent_dim)
        self.fc_logvar = nn.Linear(current_dim, latent_dim)
        
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the encoder.
        
        Args:
            x: Gene expression input
            condition: Conditioning variables
            
        Returns:
            Tuple of (mu, logvar) for latent space
        """
        # Concatenate input and condition
        combined = torch.cat([x, condition], dim=1)
        
        # Extract features
        features = self.feature_extractor(combined)
        
        # Generate latent parameters
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)
        
        return mu, logvar

class Decoder(nn.Module):
    """
    Decoder network for the CVAE.
    
    Reconstructs gene expression from latent space samples and conditioning variables.
    """
    
    def __init__(self, 
                 latent_dim: int,
                 condition_dim: int,
                 output_dim: int,
                 hidden_dims: List[int] = [128, 256, 512],
                 dropout: float = 0.2):
        """
        Initialize the decoder.
        
        Args:
            latent_dim: Dimension of latent space
            condition_dim: Dimension of conditioning variables
            output_dim: Dimension of output (gene expression)
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.output_dim = output_dim
        
        # Input processing layers
        layers = []
        current_dim = latent_dim + condition_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim
        
        self.feature_generator = nn.Sequential(*layers)
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(current_dim, output_dim),
            nn.Sigmoid()  # Gene expression is typically non-negative
        )
        
    def forward(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the decoder.
        
        Args:
            z: Latent space samples
            condition: Conditioning variables
            
        Returns:
            Reconstructed gene expression
        """
        # Concatenate latent and condition
        combined = torch.cat([z, condition], dim=1)
        
        # Generate features
        features = self.feature_generator(combined)
        
        # Generate output
        output = self.output_layer(features)
        
        return output

class ConditionalVAE(nn.Module):
    """
    Conditional Variational Autoencoder for single-cell RNA analysis.
    
    This model can generate synthetic cells conditioned on specific factors
    and learn meaningful representations of gene expression patterns.
    """
    
    def __init__(self, 
                 input_dim: int,
                 condition_dim: int,
                 latent_dim: int = 64,
                 hidden_dims: List[int] = [512, 256, 128],
                 beta: float = 1.0,
                 dropout: float = 0.2):
        """
        Initialize the CVAE.
        
        Args:
            input_dim: Dimension of gene expression input
            condition_dim: Dimension of conditioning variables
            latent_dim: Dimension of latent space
            hidden_dims: List of hidden layer dimensions
            beta: Weight for KL divergence (controls disentanglement)
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim
        self.beta = beta
        
        # Encoder and decoder
        self.encoder = Encoder(input_dim, condition_dim, hidden_dims, latent_dim, dropout)
        self.decoder = Decoder(latent_dim, condition_dim, input_dim, hidden_dims[::-1], dropout)
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for sampling from latent space.
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            Sampled latent vectors
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the CVAE.
        
        Args:
            x: Gene expression input
            condition: Conditioning variables
            
        Returns:
            Dictionary containing reconstruction, mu, logvar, and z
        """
        # Encode
        mu, logvar = self.encoder(x, condition)
        
        # Sample from latent space
        z = self.reparameterize(mu, logvar)
        
        # Decode
        x_recon = self.decoder(z, condition)
        
        return {
            'reconstruction': x_recon,
            'mu': mu,
            'logvar': logvar,
            'z': z
        }
    
    def generate(self, condition: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        Generate synthetic cells given conditioning variables.
        
        Args:
            condition: Conditioning variables
            num_samples: Number of samples to generate
            
        Returns:
            Generated gene expression
        """
        self.eval()
        with torch.no_grad():
            # Sample from prior
            z = torch.randn(num_samples, self.latent_dim, device=condition.device)
            
            # Generate
            generated = self.decoder(z, condition)
            
        return generated
    
    def interpolate(self, 
                   condition: torch.Tensor,
                   z1: torch.Tensor,
                   z2: torch.Tensor,
                   steps: int = 10) -> torch.Tensor:
        """
        Interpolate between two latent vectors.
        
        Args:
            condition: Conditioning variables
            z1: First latent vector
            z2: Second latent vector
            steps: Number of interpolation steps
            
        Returns:
            Interpolated gene expression
        """
        self.eval()
        with torch.no_grad():
            interpolated = []
            for i in range(steps):
                alpha = i / (steps - 1)
                z_interp = alpha * z1 + (1 - alpha) * z2
                gen = self.decoder(z_interp, condition)
                interpolated.append(gen)
            
        return torch.stack(interpolated)

class CVAELoss(nn.Module):
    """
    Loss function for the CVAE.
    
    Combines reconstruction loss and KL divergence with beta weighting.
    """
    
    def __init__(self, beta: float = 1.0, reconstruction_loss: str = 'mse'):
        """
        Initialize the loss function.
        
        Args:
            beta: Weight for KL divergence
            reconstruction_loss: Type of reconstruction loss ('mse' or 'bce')
        """
        super().__init__()
        self.beta = beta
        self.reconstruction_loss = reconstruction_loss
        
    def forward(self, 
                x: torch.Tensor,
                x_recon: torch.Tensor,
                mu: torch.Tensor,
                logvar: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute the CVAE loss.
        
        Args:
            x: Original input
            x_recon: Reconstructed input
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            Dictionary containing total loss and components
        """
        # Reconstruction loss
        if self.reconstruction_loss == 'mse':
            recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        elif self.reconstruction_loss == 'bce':
            recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
        else:
            raise ValueError(f"Unknown reconstruction loss: {self.reconstruction_loss}")
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        total_loss = recon_loss + self.beta * kl_loss
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': recon_loss,
            'kl_loss': kl_loss
        }

class CVAETrainer:
    """
    Trainer class for the Conditional VAE.
    
    Handles training, validation, and generation of synthetic cells.
    """
    
    def __init__(self, 
                 model: ConditionalVAE,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-5):
        """
        Initialize the trainer.
        
        Args:
            model: CVAE model
            device: Device to run training on
            learning_rate: Learning rate for optimization
            weight_decay: Weight decay for regularization
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        self.criterion = CVAELoss(beta=model.beta)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5
        )
        
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary of losses
        """
        self.model.train()
        total_losses = {'total': 0, 'reconstruction': 0, 'kl': 0}
        
        for batch in train_loader:
            x, condition = batch
            x = x.to(self.device)
            condition = condition.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(x, condition)
            
            # Compute loss
            losses = self.criterion(x, output['reconstruction'], 
                                  output['mu'], output['logvar'])
            
            # Backward pass
            losses['total_loss'].backward()
            self.optimizer.step()
            
            # Accumulate losses
            for key in total_losses:
                total_losses[key] += losses[f'{key}_loss'].item()
        
        # Average losses
        num_batches = len(train_loader)
        return {key: value / num_batches for key, value in total_losses.items()}
    
    def validate(self, val_loader) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary of validation losses
        """
        self.model.eval()
        total_losses = {'total': 0, 'reconstruction': 0, 'kl': 0}
        
        with torch.no_grad():
            for batch in val_loader:
                x, condition = batch
                x = x.to(self.device)
                condition = condition.to(self.device)
                
                # Forward pass
                output = self.model(x, condition)
                
                # Compute loss
                losses = self.criterion(x, output['reconstruction'], 
                                      output['mu'], output['logvar'])
                
                # Accumulate losses
                for key in total_losses:
                    total_losses[key] += losses[f'{key}_loss'].item()
        
        # Average losses
        num_batches = len(val_loader)
        return {key: value / num_batches for key, value in total_losses.items()}
    
    def train(self, 
              train_loader,
              val_loader,
              epochs: int = 100,
              early_stopping_patience: int = 20) -> Dict[str, List[float]]:
        """
        Train the model with early stopping.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum number of epochs
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Training history
        """
        history = {
            'train_total_loss': [], 'train_reconstruction_loss': [], 'train_kl_loss': [],
            'val_total_loss': [], 'val_reconstruction_loss': [], 'val_kl_loss': []
        }
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            train_losses = self.train_epoch(train_loader)
            
            # Validation
            val_losses = self.validate(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step(val_losses['total'])
            
            # Early stopping
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_cvae.pth')
            else:
                patience_counter += 1
            
            # Record history
            for key in train_losses:
                history[f'train_{key}_loss'].append(train_losses[key])
                history[f'val_{key}_loss'].append(val_losses[key])
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: "
                           f"Train Loss: {train_losses['total']:.4f}, "
                           f"Val Loss: {val_losses['total']:.4f}")
            
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        return history

class SingleCellDataProcessor:
    """
    Data processor for single-cell RNA data to work with CVAE.
    
    Handles data preprocessing, conditioning variable creation, and
    data loading for the CVAE model.
    """
    
    def __init__(self, 
                 gene_columns: List[str],
                 condition_columns: List[str],
                 batch_size: int = 32):
        """
        Initialize the data processor.
        
        Args:
            gene_columns: List of gene expression column names
            condition_columns: List of conditioning variable column names
            batch_size: Batch size for data loading
        """
        self.gene_columns = gene_columns
        self.condition_columns = condition_columns
        self.batch_size = batch_size
        
        self.gene_scaler = StandardScaler()
        self.condition_scaler = StandardScaler()
        self.label_encoders = {}
        
    def preprocess_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the data for CVAE training.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Tuple of (gene_expression, conditions)
        """
        # Process gene expression
        gene_data = data[self.gene_columns].values
        gene_data_scaled = self.gene_scaler.fit_transform(gene_data)
        
        # Process conditioning variables
        condition_data = []
        for col in self.condition_columns:
            if col in data.columns:
                if data[col].dtype == 'object':
                    # Categorical variable
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                    encoded = self.label_encoders[col].fit_transform(data[col].fillna('Unknown'))
                    condition_data.append(encoded)
                else:
                    # Numerical variable
                    condition_data.append(data[col].fillna(0).values)
        
        condition_data = np.column_stack(condition_data)
        condition_data_scaled = self.condition_scaler.fit_transform(condition_data)
        
        return gene_data_scaled, condition_data_scaled
    
    def create_data_loader(self, 
                          gene_data: np.ndarray,
                          condition_data: np.ndarray,
                          shuffle: bool = True) -> torch.utils.data.DataLoader:
        """
        Create a PyTorch DataLoader for training.
        
        Args:
            gene_data: Preprocessed gene expression data
            condition_data: Preprocessed condition data
            shuffle: Whether to shuffle the data
            
        Returns:
            PyTorch DataLoader
        """
        # Convert to tensors
        gene_tensor = torch.FloatTensor(gene_data)
        condition_tensor = torch.FloatTensor(condition_data)
        
        # Create dataset
        dataset = torch.utils.data.TensorDataset(gene_tensor, condition_tensor)
        
        # Create data loader
        loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=shuffle
        )
        
        return loader

def create_cvae_model(input_dim: int,
                     condition_dim: int,
                     latent_dim: int = 64,
                     hidden_dims: List[int] = [512, 256, 128],
                     beta: float = 1.0) -> ConditionalVAE:
    """
    Factory function to create a CVAE model.
    
    Args:
        input_dim: Input feature dimension
        condition_dim: Condition feature dimension
        latent_dim: Latent space dimension
        hidden_dims: Hidden layer dimensions
        beta: Beta weight for KL divergence
        
    Returns:
        Configured CVAE model
    """
    return ConditionalVAE(
        input_dim=input_dim,
        condition_dim=condition_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        beta=beta
    )

# Example usage and testing
if __name__ == "__main__":
    # Example: Create and test the CVAE model
    logger.info("Testing Conditional VAE model...")
    
    # Create synthetic data
    n_cells = 1000
    n_genes = 50
    n_conditions = 3
    
    # Generate synthetic gene expression and conditions
    gene_expression = np.random.rand(n_cells, n_genes)
    conditions = np.random.rand(n_cells, n_conditions)
    
    # Create model
    model = create_cvae_model(n_genes, n_conditions, latent_dim=32)
    
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    logger.info("Conditional VAE model ready for training!")

    # Always generate synthetic spatial coordinates for GNN
    spatial_coords = np.random.rand(len(data), 2) * 1000 