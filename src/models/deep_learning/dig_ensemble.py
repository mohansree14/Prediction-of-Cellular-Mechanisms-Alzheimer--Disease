"""
Deeply Integrated Graph Ensemble (DIG-Ensemble)

This module implements a novel, end-to-end trainable hybrid model that combines
a Graph Neural Network (GNN) backbone with a differentiable decision forest.
The model overcomes limitations of simple GNN-RF pipelines through three key innovations:
1. Attention-gated feature fusion
2. Differentiable forest for joint optimization  
3. Residual connection for robust prediction

Key Components:
- Component A: GNN Backbone (GraphSAGE/GAT)
- Component B: Attention-Gated Feature Fusion Module
- Component C: Differentiable Forest Module
- Component D: Final Output Layer with Residual Connection

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import logging
from typing import Tuple, List, Optional, Dict, Any, Union
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GNNBackbone(nn.Module):
    """
    Component A: GNN Backbone
    
    Creates a GNN module that processes a graph and produces:
    - E_GNN: Final node embeddings (128-dimensional vector)
    - P_GNN: Preliminary node classification probability distribution
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 128,
                 output_dim: int = 10,
                 num_layers: int = 3,
                 dropout: float = 0.3,
                 gnn_type: str = 'gat'):
        """
        Initialize the GNN backbone.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (number of classes)
            num_layers: Number of GNN layers
            dropout: Dropout rate
            gnn_type: Type of GNN ('gat' or 'sage')
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.gnn_type = gnn_type
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        if gnn_type == 'gat':
            # Graph Attention Network layers
            for i in range(num_layers):
                if i == 0:
                    self.gnn_layers.append(GATConv(input_dim, hidden_dim, heads=4, dropout=dropout))
                    self.batch_norms.append(nn.BatchNorm1d(hidden_dim * 4))  # 4 heads
                elif i == num_layers - 1:
                    self.gnn_layers.append(GATConv(hidden_dim * 4, hidden_dim, heads=1, dropout=dropout))
                    self.batch_norms.append(nn.BatchNorm1d(hidden_dim))  # 1 head
                else:
                    self.gnn_layers.append(GATConv(hidden_dim * 4, hidden_dim, heads=4, dropout=dropout))
                    self.batch_norms.append(nn.BatchNorm1d(hidden_dim * 4))  # 4 heads
        else:
            # GraphSAGE layers
            for i in range(num_layers):
                if i == 0:
                    self.gnn_layers.append(SAGEConv(input_dim, hidden_dim))
                else:
                    self.gnn_layers.append(SAGEConv(hidden_dim, hidden_dim))
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Classification head for P_GNN
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the GNN backbone.
        
        Args:
            x: Node features
            edge_index: Graph connectivity
            
        Returns:
            E_GNN: Node embeddings
            P_GNN: Preliminary classification probabilities
        """
        # GNN message passing
        h = x
        for i, (gnn_layer, bn) in enumerate(zip(self.gnn_layers, self.batch_norms)):
            h = gnn_layer(h, edge_index)
            h = bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        # E_GNN: Final node embeddings
        E_GNN = h
        
        # P_GNN: Preliminary classification probabilities
        logits = self.classifier(E_GNN)
        P_GNN = F.softmax(logits, dim=-1)
        
        return E_GNN, P_GNN


class AttentionGatedFeatureFusion(nn.Module):
    """
    Component B: Attention-Gated Feature Fusion Module
    
    Intelligently merges original node features with GNN embeddings using
    an attention gate that learns the importance of each feature source.
    """
    
    def __init__(self, original_dim: int, gnn_dim: int, fusion_dim: int = 128):
        """
        Initialize the attention-gated feature fusion module.
        
        Args:
            original_dim: Dimension of original node features
            gnn_dim: Dimension of GNN embeddings
            fusion_dim: Dimension of fused features
        """
        super().__init__()
        self.original_dim = original_dim
        self.gnn_dim = gnn_dim
        self.fusion_dim = fusion_dim
        
        # Linear projection for original features to match GNN dimension
        self.original_projection = nn.Linear(original_dim, gnn_dim)
        
        # Attention gate network
        self.attention_net = nn.Sequential(
            nn.Linear(gnn_dim * 2, gnn_dim),
            nn.ReLU(),
            nn.Linear(gnn_dim, gnn_dim // 2),
            nn.ReLU(),
            nn.Linear(gnn_dim // 2, gnn_dim),
            nn.Sigmoid()
        )
        
        # Final fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(gnn_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
    def forward(self, F_orig: torch.Tensor, E_GNN: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the attention-gated feature fusion.
        
        Args:
            F_orig: Original node features
            E_GNN: GNN embeddings
            
        Returns:
            F_final: Fused feature vector
            g: Attention gate values (for interpretability)
        """
        # Project original features to match GNN dimension
        F_orig_proj = self.original_projection(F_orig)
        
        # Concatenate features for attention computation
        C = torch.cat([F_orig_proj, E_GNN], dim=-1)
        
        # Compute attention gate
        g = self.attention_net(C)
        
        # Fused feature vector
        F_final = (g * E_GNN) + ((1 - g) * F_orig_proj)
        
        # Final transformation
        F_final = self.fusion_layer(F_final)
        
        return F_final, g


class DifferentiableForest(nn.Module):
    """
    Component C: Differentiable Forest Module
    
    Implements a differentiable decision forest classifier that allows
    end-to-end training with gradient flow.
    """
    
    def __init__(self, input_dim: int, num_classes: int, num_trees: int = 10, tree_depth: int = 4):
        """
        Initialize the differentiable forest.
        
        Args:
            input_dim: Input feature dimension
            num_classes: Number of output classes
            num_trees: Number of trees in the forest
            tree_depth: Maximum depth of each tree
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_trees = num_trees
        self.tree_depth = tree_depth
        
        # Decision nodes (splitting functions)
        self.decision_nodes = nn.ModuleList([
            nn.Linear(input_dim, 1) for _ in range(num_trees * (2**tree_depth - 1))
        ])
        
        # Leaf nodes (prediction functions)
        self.leaf_nodes = nn.ModuleList([
            nn.Linear(input_dim, num_classes) for _ in range(num_trees * 2**tree_depth)
        ])
        
        # Tree structure parameters
        self.tree_weights = nn.Parameter(torch.ones(num_trees) / num_trees)
        
    def _get_tree_path(self, x: torch.Tensor, tree_idx: int) -> torch.Tensor:
        """
        Compute the path through a single tree.
        
        Args:
            x: Input features
            tree_idx: Index of the tree
            
        Returns:
            Path probabilities through the tree
        """
        batch_size = x.size(0)
        path_probs = torch.ones(batch_size, 1).to(x.device)
        
        for depth in range(self.tree_depth):
            # Get decision nodes for this depth
            start_idx = tree_idx * (2**self.tree_depth - 1) + (2**depth - 1)
            end_idx = start_idx + 2**depth
            
            current_paths = []
            for node_idx in range(start_idx, end_idx):
                # Decision function
                decision = torch.sigmoid(self.decision_nodes[node_idx](x))
                current_paths.append(decision)
                current_paths.append(1 - decision)
            
            # Update path probabilities
            path_probs = path_probs.repeat(1, 2)
            path_probs = path_probs * torch.cat(current_paths, dim=1)
        
        return path_probs
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the differentiable forest.
        
        Args:
            x: Input features
            
        Returns:
            P_Forest: Forest prediction probabilities
        """
        batch_size = x.size(0)
        forest_outputs = []
        
        for tree_idx in range(self.num_trees):
            # Get path through this tree
            path_probs = self._get_tree_path(x, tree_idx)
            
            # Get leaf predictions
            leaf_start_idx = tree_idx * 2**self.tree_depth
            leaf_end_idx = leaf_start_idx + 2**self.tree_depth
            
            tree_output = torch.zeros(batch_size, self.num_classes).to(x.device)
            
            for leaf_idx in range(leaf_start_idx, leaf_end_idx):
                leaf_pred = F.softmax(self.leaf_nodes[leaf_idx](x), dim=-1)
                path_idx = leaf_idx - leaf_start_idx
                tree_output += path_probs[:, path_idx:path_idx+1] * leaf_pred
            
            forest_outputs.append(tree_output)
        
        # Weighted combination of trees
        forest_outputs = torch.stack(forest_outputs, dim=1)
        tree_weights = F.softmax(self.tree_weights, dim=0)
        P_Forest = torch.sum(forest_outputs * tree_weights.view(1, -1, 1), dim=1)
        
        return P_Forest


class DIGEnsemble(nn.Module):
    """
    Deeply Integrated Graph Ensemble (DIG-Ensemble)
    
    Complete hybrid model that combines GNN backbone, attention-gated feature fusion,
    differentiable forest, and residual connection for end-to-end training.
    """
    
    def __init__(self, 
                 input_dim: int,
                 num_classes: int,
                 hidden_dim: int = 128,
                 gnn_layers: int = 3,
                 num_trees: int = 10,
                 tree_depth: int = 4,
                 gnn_type: str = 'gat',
                 dropout: float = 0.3):
        """
        Initialize the DIG-Ensemble model.
        
        Args:
            input_dim: Input feature dimension
            num_classes: Number of output classes
            hidden_dim: Hidden layer dimension
            gnn_layers: Number of GNN layers
            num_trees: Number of trees in the forest
            tree_depth: Maximum depth of each tree
            gnn_type: Type of GNN ('gat' or 'sage')
            dropout: Dropout rate
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # Component A: GNN Backbone
        self.gnn_backbone = GNNBackbone(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=num_classes,
            num_layers=gnn_layers,
            dropout=dropout,
            gnn_type=gnn_type
        )
        
        # Component B: Attention-Gated Feature Fusion
        self.feature_fusion = AttentionGatedFeatureFusion(
            original_dim=input_dim,
            gnn_dim=hidden_dim,
            fusion_dim=hidden_dim
        )
        
        # Component C: Differentiable Forest
        self.forest = DifferentiableForest(
            input_dim=hidden_dim,
            num_classes=num_classes,
            num_trees=num_trees,
            tree_depth=tree_depth
        )
        
        # Component D: Learnable alpha for residual connection
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete DIG-Ensemble model.
        
        Args:
            x: Node features
            edge_index: Graph connectivity
            
        Returns:
            Dictionary containing all intermediate outputs and final prediction
        """
        # Component A: GNN Backbone
        E_GNN, P_GNN = self.gnn_backbone(x, edge_index)
        
        # Component B: Attention-Gated Feature Fusion
        F_final, attention_gate = self.feature_fusion(x, E_GNN)
        
        # Component C: Differentiable Forest
        P_Forest = self.forest(F_final)
        
        # Component D: Final Output with Residual Connection
        alpha = torch.sigmoid(self.alpha)  # Ensure alpha is between 0 and 1
        P_Final = alpha * P_Forest + (1 - alpha) * P_GNN
        
        return {
            'E_GNN': E_GNN,
            'P_GNN': P_GNN,
            'F_final': F_final,
            'attention_gate': attention_gate,
            'P_Forest': P_Forest,
            'P_Final': P_Final,
            'alpha': alpha
        }
    
    def get_interpretability_info(self, x: torch.Tensor, edge_index: torch.Tensor) -> Dict[str, Any]:
        """
        Extract interpretability information from the model.
        
        Args:
            x: Node features
            edge_index: Graph connectivity
            
        Returns:
            Dictionary containing interpretability information
        """
        with torch.no_grad():
            outputs = self.forward(x, edge_index)
            
            return {
                'attention_gate_values': outputs['attention_gate'].cpu().numpy(),
                'alpha_value': outputs['alpha'].cpu().item(),
                'gnn_predictions': outputs['P_GNN'].cpu().numpy(),
                'forest_predictions': outputs['P_Forest'].cpu().numpy(),
                'final_predictions': outputs['P_Final'].cpu().numpy(),
                'gnn_embeddings': outputs['E_GNN'].cpu().numpy(),
                'fused_features': outputs['F_final'].cpu().numpy()
            }


class DIGEnsembleTrainer:
    """
    Trainer class for the DIG-Ensemble model.
    
    Handles training, validation, and evaluation of the complete hybrid model.
    """
    
    def __init__(self, 
                 model: DIGEnsemble,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-5,
                 class_weights: torch.Tensor = None):
        """
        Initialize the trainer.
        
        Args:
            model: DIG-Ensemble model
            device: Device to train on
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            class_weights: Class weights for handling imbalance
        """
        self.model = model.to(device)
        self.device = device
        
        # Use weighted loss if class weights are provided
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        else:
            self.criterion = nn.CrossEntropyLoss()
            
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            batch = batch.to(self.device)
            
            # Forward pass
            outputs = self.model(batch.x, batch.edge_index)
            loss = self.criterion(outputs['P_Final'], batch.y)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (validation loss, validation accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                
                # Forward pass
                outputs = self.model(batch.x, batch.edge_index)
                loss = self.criterion(outputs['P_Final'], batch.y)
                
                # Calculate accuracy
                pred = outputs['P_Final'].argmax(dim=1)
                correct += (pred == batch.y).sum().item()
                total += batch.y.size(0)
                
                total_loss += loss.item()
        
        val_loss = total_loss / len(val_loader)
        val_acc = correct / total
        
        return val_loss, val_acc
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              epochs: int = 100,
              early_stopping_patience: int = 20) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            early_stopping_patience: Early stopping patience
            
        Returns:
            Dictionary containing training history
        """
        best_val_loss = float('inf')
        patience_counter = 0
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': []
        }
        
        logger.info("Starting DIG-Ensemble training...")
        
        for epoch in range(epochs):
            # Training
            train_loss = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc = self.validate(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Record history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Logging
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_dig_ensemble.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_dig_ensemble.pth'))
        
        return history


class DIGEnsembleVisualizer:
    """
    Visualization utilities for the DIG-Ensemble model.
    
    Provides methods to visualize attention gates, feature importance,
    and model interpretability.
    """
    
    @staticmethod
    def plot_attention_gates(attention_gates: np.ndarray, save_path: str = None):
        """
        Plot attention gate values.
        
        Args:
            attention_gates: Attention gate values from the model
            save_path: Path to save the plot
        """
        plt.figure(figsize=(12, 6))
        
        # Plot attention gate distribution
        plt.subplot(1, 2, 1)
        plt.hist(attention_gates.flatten(), bins=50, alpha=0.7, color='skyblue')
        plt.title('Distribution of Attention Gate Values')
        plt.xlabel('Attention Gate Value')
        plt.ylabel('Frequency')
        plt.axvline(0.5, color='red', linestyle='--', label='Equal Weight')
        plt.legend()
        
        # Plot attention gate heatmap
        plt.subplot(1, 2, 2)
        sns.heatmap(attention_gates[:50, :], cmap='viridis', cbar_kws={'label': 'Attention Value'})
        plt.title('Attention Gate Heatmap (First 50 Nodes)')
        plt.xlabel('Feature Dimension')
        plt.ylabel('Node Index')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_model_comparison(gnn_preds: np.ndarray, forest_preds: np.ndarray, 
                            final_preds: np.ndarray, true_labels: np.ndarray,
                            save_path: str = None):
        """
        Plot comparison between GNN, Forest, and Final predictions.
        
        Args:
            gnn_preds: GNN predictions
            forest_preds: Forest predictions
            final_preds: Final ensemble predictions
            true_labels: True labels
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # GNN vs Forest predictions
        axes[0, 0].scatter(gnn_preds.argmax(axis=1), forest_preds.argmax(axis=1), alpha=0.6)
        axes[0, 0].set_xlabel('GNN Predictions')
        axes[0, 0].set_ylabel('Forest Predictions')
        axes[0, 0].set_title('GNN vs Forest Predictions')
        
        # Final vs True labels
        axes[0, 1].scatter(final_preds.argmax(axis=1), true_labels, alpha=0.6)
        axes[0, 1].set_xlabel('Final Predictions')
        axes[0, 1].set_ylabel('True Labels')
        axes[0, 1].set_title('Final Predictions vs True Labels')
        
        # Prediction confidence distributions
        axes[1, 0].hist(gnn_preds.max(axis=1), bins=30, alpha=0.7, label='GNN', color='blue')
        axes[1, 0].hist(forest_preds.max(axis=1), bins=30, alpha=0.7, label='Forest', color='green')
        axes[1, 0].hist(final_preds.max(axis=1), bins=30, alpha=0.7, label='Final', color='red')
        axes[1, 0].set_xlabel('Prediction Confidence')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Prediction Confidence Distributions')
        axes[1, 0].legend()
        
        # Alpha value over time (if available)
        axes[1, 1].text(0.5, 0.5, 'Alpha Value: Model Parameter\n(Shows GNN vs Forest Weight)', 
                       ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].set_title('Model Parameters')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def create_dig_ensemble_model(input_dim: int, 
                            num_classes: int,
                            hidden_dim: int = 128,
                            gnn_layers: int = 3,
                            num_trees: int = 10,
                            tree_depth: int = 4,
                            gnn_type: str = 'gat') -> DIGEnsemble:
    """
    Factory function to create a DIG-Ensemble model.
    
    Args:
        input_dim: Input feature dimension
        num_classes: Number of output classes
        hidden_dim: Hidden layer dimension
        gnn_layers: Number of GNN layers
        num_trees: Number of trees in the forest
        tree_depth: Maximum depth of each tree
        gnn_type: Type of GNN ('gat' or 'sage')
        
    Returns:
        Configured DIG-Ensemble model
    """
    model = DIGEnsemble(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        gnn_layers=gnn_layers,
        num_trees=num_trees,
        tree_depth=tree_depth,
        gnn_type=gnn_type
    )
    
    logger.info(f"Created DIG-Ensemble model with {sum(p.numel() for p in model.parameters())} parameters")
    return model


def evaluate_dig_ensemble(model: DIGEnsemble, 
                        test_loader: DataLoader,
                        device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> Dict[str, float]:
    """
    Evaluate the DIG-Ensemble model.
    
    Args:
        model: Trained DIG-Ensemble model
        test_loader: Test data loader
        device: Device to evaluate on
        
    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    model = model.to(device)
    
    all_predictions = []
    all_labels = []
    all_gnn_preds = []
    all_forest_preds = []
    all_attention_gates = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            
            outputs = model(batch.x, batch.edge_index)
            
            all_predictions.append(outputs['P_Final'].cpu())
            all_labels.append(batch.y.cpu())
            all_gnn_preds.append(outputs['P_GNN'].cpu())
            all_forest_preds.append(outputs['P_Forest'].cpu())
            all_attention_gates.append(outputs['attention_gate'].cpu())
    
    # Concatenate all batches
    predictions = torch.cat(all_predictions, dim=0)
    labels = torch.cat(all_labels, dim=0)
    gnn_preds = torch.cat(all_gnn_preds, dim=0)
    forest_preds = torch.cat(all_forest_preds, dim=0)
    attention_gates = torch.cat(all_attention_gates, dim=0)
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    
    pred_classes = predictions.argmax(dim=1).numpy()
    true_classes = labels.numpy()
    
    accuracy = accuracy_score(true_classes, pred_classes)
    f1_macro = f1_score(true_classes, pred_classes, average='macro')
    f1_weighted = f1_score(true_classes, pred_classes, average='weighted')
    
    # Component-wise accuracy
    gnn_accuracy = accuracy_score(true_classes, gnn_preds.argmax(dim=1).numpy())
    forest_accuracy = accuracy_score(true_classes, forest_preds.argmax(dim=1).numpy())
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'gnn_accuracy': gnn_accuracy,
        'forest_accuracy': forest_accuracy,
        'alpha_value': model.alpha.item(),
        'attention_gate_mean': attention_gates.mean().item(),
        'attention_gate_std': attention_gates.std().item()
    }
