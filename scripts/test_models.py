#!/usr/bin/env python3
"""
Test script for GNN and CVAE models

This script tests the basic functionality of both models to ensure they
can be imported and initialized correctly before running the full pipeline.
"""

import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import argparse
import yaml

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        from models.deep_learning.gnn_model import (
            SpatialGraphConstructor, 
            SpatialGNN, 
            create_spatial_gnn_model
        )
        print("✅ GNN model imports successful")
        
        from models.deep_learning.generative_model import (
            ConditionalVAE,
            create_cvae_model
        )
        print("✅ CVAE model imports successful")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_gnn_model():
    """Test GNN model creation and basic functionality."""
    print("\nTesting GNN model...")
    
    try:
        from models.deep_learning.gnn_model import (
            SpatialGraphConstructor, 
            create_spatial_gnn_model
        )
        
        # Create synthetic data
        n_cells = 100
        n_genes = 20
        n_classes = 5
        
        gene_expression = np.random.randn(n_cells, n_genes)
        spatial_coords = np.random.rand(n_cells, 2) * 100
        cell_types = np.random.randint(0, n_classes, n_cells)
        
        # Test graph construction
        graph_constructor = SpatialGraphConstructor(n_neighbors=5, radius=50.0)
        graph_data = graph_constructor.construct_graph(gene_expression, spatial_coords, cell_types)
        
        print(f"✅ Graph constructed: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")
        
        # Test model creation
        model = create_spatial_gnn_model(n_genes, n_classes, hidden_dim=64, num_layers=2)
        print(f"✅ GNN model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Test forward pass
        with torch.no_grad():
            output = model(graph_data)
            print(f"✅ Forward pass successful: output shape {output.shape}")
        
        return True
    except Exception as e:
        print(f"❌ GNN test failed: {e}")
        return False

def test_cvae_model():
    """Test CVAE model creation and basic functionality."""
    print("\nTesting CVAE model...")
    
    try:
        from models.deep_learning.generative_model import create_cvae_model
        
        # Create synthetic data
        n_cells = 100
        n_genes = 20
        n_conditions = 3
        
        gene_expression = torch.rand(n_cells, n_genes)
        conditions = torch.rand(n_cells, n_conditions)
        
        # Test model creation
        model = create_cvae_model(n_genes, n_conditions, latent_dim=16, hidden_dims=[64, 32])
        print(f"✅ CVAE model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Test forward pass
        with torch.no_grad():
            output = model(gene_expression, conditions)
            print(f"✅ Forward pass successful: reconstruction shape {output['reconstruction'].shape}")
        
        # Test generation
        synthetic = model.generate(conditions[:5], num_samples=5)
        print(f"✅ Generation successful: synthetic shape {synthetic.shape}")
        
        return True
    except Exception as e:
        print(f"❌ CVAE test failed: {e}")
        return False

def test_data_loading():
    """Test that the processed data can be loaded."""
    print("\nTesting data loading...")
    try:
        parser = argparse.ArgumentParser(description="Test Data Loading")
        parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config YAML file')
        args, _ = parser.parse_known_args()
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        data_path = config['data_path']
        if not Path(data_path).exists():
            print(f"⚠️  Processed data not found. Please run preprocessing first.")
            return False
        # Load a small sample
        sample_data = pd.read_csv(data_path, nrows=100)
        print(f"✅ Data loaded: {sample_data.shape}")
        print(f"✅ Columns: {len(sample_data.columns)}")
        return True
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("MODEL TESTING SUITE")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("GNN Model Test", test_gnn_model),
        ("CVAE Model Test", test_cvae_model),
        ("Data Loading Test", test_data_loading)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Models are ready for use.")
        print("You can now run: python scripts/04_run_hybrid_modeling.py")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        print("Make sure all dependencies are installed:")
        print("pip install -r requirements_deep_learning.txt")

if __name__ == "__main__":
    main() 