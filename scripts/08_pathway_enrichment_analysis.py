#!/usr/bin/env python3
"""
Pathway Enrichment Analysis Script

This script performs pathway enrichment analysis on top genes identified by
machine learning models to discover biological mechanisms relevant to Alzheimer's disease.

Author: AI Assistant
Date: 2024
"""

import os
import sys
import logging
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
import matplotlib.patheffects as PathEffects
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import argparse
import yaml
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# Configure logging
def setup_logging(results_dir: Path) -> logging.Logger:
    """Setup logging configuration."""
    log_file = results_dir / "pathway_enrichment_log.txt"
    logger = logging.getLogger("pathway_enrichment")
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()
    
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

def load_feature_importance_data(results_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Load feature importance data from various model outputs.
    
    Args:
        results_dir: Path to results directory
        
    Returns:
        Dictionary containing feature importance dataframes
    """
    logger = logging.getLogger(__name__)
    feature_data = {}
    
    # Look for feature importance files
    feature_files = list(results_dir.rglob("*feature_importance*.csv")) + \
                   list(results_dir.rglob("*feature_importance*.json"))
    
    for file_path in feature_files:
        try:
            if file_path.suffix == '.csv':
                df = pd.read_csv(file_path)
            elif file_path.suffix == '.json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
                df = pd.DataFrame(data)
            
            feature_data[file_path.stem] = df
            logger.info(f"Loaded feature importance from {file_path}")
            
        except Exception as e:
            logger.warning(f"Could not load {file_path}: {e}")
    
    return feature_data

def create_synthetic_pathway_data(gene_names: List[str]) -> Dict[str, List[str]]:
    """
    Create synthetic pathway data for demonstration.
    In a real scenario, this would be replaced with actual pathway databases.
    
    Args:
        gene_names: List of gene names
        
    Returns:
        Dictionary mapping pathway names to gene lists
    """
    # Alzheimer's disease related pathways (synthetic)
    pathways = {
        'Amyloid_beta_processing': ['APP', 'BACE1', 'PSEN1', 'PSEN2', 'APOE'],
        'Tau_phosphorylation': ['MAPT', 'GSK3B', 'CDK5', 'PPP2CA'],
        'Neuroinflammation': ['TREM2', 'CD33', 'IL1B', 'TNF', 'CX3CR1'],
        'Mitochondrial_dysfunction': ['COX1', 'COX2', 'ND1', 'ND2', 'ATP6'],
        'Synaptic_dysfunction': ['SYN1', 'SNAP25', 'VAMP2', 'GRIN1', 'GRIN2A'],
        'Oxidative_stress': ['SOD1', 'SOD2', 'CAT', 'GPX1', 'NQO1'],
        'Autophagy': ['BECN1', 'LC3B', 'ATG5', 'ATG7', 'SQSTM1'],
        'Cell_death': ['CASP3', 'CASP9', 'BAX', 'BCL2', 'TP53']
    }
    
    # Filter pathways to only include genes present in the data
    filtered_pathways = {}
    for pathway, genes in pathways.items():
        present_genes = [gene for gene in genes if gene in gene_names]
        if present_genes:
            filtered_pathways[pathway] = present_genes
    
    return filtered_pathways

def perform_pathway_enrichment(gene_list: List[str], 
                             background_genes: List[str],
                             pathway_data: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Perform pathway enrichment analysis using hypergeometric test.
    
    Args:
        gene_list: List of genes of interest (e.g., top genes from model)
        background_genes: List of all genes in the dataset
        pathway_data: Dictionary mapping pathway names to gene lists
        
    Returns:
        DataFrame with enrichment results
    """
    from scipy.stats import hypergeom
    import math
    
    results = []
    
    for pathway_name, pathway_genes in pathway_data.items():
        # Calculate overlap
        overlap = set(gene_list) & set(pathway_genes)
        overlap_size = len(overlap)
        
        if overlap_size == 0:
            continue
        
        # Hypergeometric test parameters
        N = len(background_genes)  # Total population
        K = len(pathway_genes)     # Successes in population
        n = len(gene_list)         # Sample size
        k = overlap_size           # Successes in sample
        
        # Calculate p-value
        p_value = hypergeom.sf(k-1, N, K, n)
        
        # Calculate enrichment ratio
        expected = (K * n) / N
        enrichment_ratio = k / expected if expected > 0 else 0
        
        # Calculate adjusted p-value (Bonferroni correction)
        adjusted_p_value = min(p_value * len(pathway_data), 1.0)
        
        # Calculate -log10(p-value) for visualization
        neg_log_p = -math.log10(p_value) if p_value > 0 else 0
        
        results.append({
            'Pathway': pathway_name,
            'Overlap_Size': overlap_size,
            'Pathway_Size': K,
            'Gene_List_Size': n,
            'Expected_Overlap': expected,
            'Enrichment_Ratio': enrichment_ratio,
            'P_Value': p_value,
            'Adjusted_P_Value': adjusted_p_value,
            'Neg_Log_P_Value': neg_log_p,
            'Overlap_Genes': ', '.join(overlap)
        })
    
    return pd.DataFrame(results).sort_values('P_Value')

def create_pathway_visualizations(enrichment_results: pd.DataFrame, 
                                results_dir: Path,
                                title: str = "Pathway Enrichment Analysis"):
    """
    Create pathway enrichment visualizations.
    
    Args:
        enrichment_results: DataFrame with enrichment results
        results_dir: Directory to save plots
        title: Plot title
    """
    logger = logging.getLogger(__name__)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Enrichment plot (bubble plot)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Filter significant pathways
    significant = enrichment_results[enrichment_results['Adjusted_P_Value'] < 0.05]
    
    if len(significant) > 0:
        scatter = ax.scatter(significant['Enrichment_Ratio'], 
                           significant['Neg_Log_P_Value'],
                           s=significant['Overlap_Size'] * 20,
                           alpha=0.7,
                           c=significant['Neg_Log_P_Value'],
                           cmap='viridis')
        
        # Add labels
        for idx, row in significant.iterrows():
            ax.annotate(row['Pathway'], 
                       (row['Enrichment_Ratio'], row['Neg_Log_P_Value']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.8)
        
        ax.set_xlabel('Enrichment Ratio')
        ax.set_ylabel('-log10(P-value)')
        ax.set_title(f'{title}\nSignificant Pathways (FDR < 0.05)')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('-log10(P-value)')
        
        plt.tight_layout()
        plt.savefig(results_dir / 'pathway_enrichment_bubble.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved pathway enrichment bubble plot")
    
    # 2. Bar plot of top pathways
    fig, ax = plt.subplots(figsize=(12, 8))
    
    top_pathways = enrichment_results.head(15)
    
    bars = ax.barh(range(len(top_pathways)), 
                   top_pathways['Neg_Log_P_Value'],
                   color=sns.color_palette("husl", len(top_pathways)))
    
    ax.set_yticks(range(len(top_pathways)))
    ax.set_yticklabels(top_pathways['Pathway'])
    ax.set_xlabel('-log10(P-value)')
    ax.set_title(f'{title}\nTop 15 Enriched Pathways')
    
    # Add significance threshold line
    threshold = -np.log10(0.05)
    ax.axvline(threshold, color='red', linestyle='--', alpha=0.7, label='P < 0.05')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(results_dir / 'pathway_enrichment_bar.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved pathway enrichment bar plot")
    
    # 3. Heatmap of gene-pathway overlap
    if len(enrichment_results) > 0:
        # Get all unique genes from significant pathways
        all_genes = set()
        for genes_str in enrichment_results['Overlap_Genes']:
            all_genes.update(genes_str.split(', '))
        all_genes = list(all_genes)  # Convert set to list for indexing
        
        # Create gene-pathway matrix
        gene_pathway_matrix = []
        pathway_names = []
        
        for _, row in enrichment_results.iterrows():
            pathway_genes = set(row['Overlap_Genes'].split(', '))
            pathway_names.append(row['Pathway'])
            
            row_data = [1 if gene in pathway_genes else 0 for gene in all_genes]
            gene_pathway_matrix.append(row_data)
        
        if gene_pathway_matrix:
            gene_pathway_matrix = np.array(gene_pathway_matrix)
            # Sort genes by their first pathway occurrence for grouping
            gene_to_first_pathway = {}
            for j, gene in enumerate(all_genes):
                for i, row in enumerate(gene_pathway_matrix):
                    if row[j] == 1:
                        gene_to_first_pathway[gene] = i
                        break
                else:
                    gene_to_first_pathway[gene] = len(pathway_names)
            sorted_genes = sorted(all_genes, key=lambda g: gene_to_first_pathway[g])
            gene_indices = [all_genes.index(g) for g in sorted_genes]
            gene_pathway_matrix_sorted = gene_pathway_matrix[:, gene_indices]

            # Add a professional heatmap with the 'viridis' colormap
            fig, ax = plt.subplots(figsize=(max(10, len(pathway_names) * 1.2), max(6, len(sorted_genes) * 0.7)))
            sns.set(font_scale=1.2)
            heatmap = sns.heatmap(
                gene_pathway_matrix_sorted.T,
                annot=True,
                fmt="d",
                cmap="viridis",
                linewidths=0.5,
                linecolor='white',
                xticklabels=pathway_names,
                yticklabels=sorted_genes,
                cbar=True
            )
            plt.title("Gene–Pathway Overlap Matrix for Alzheimer’s Mechanisms", fontsize=16)
            plt.xlabel("Pathways", fontsize=14)
            plt.ylabel("Genes", fontsize=14)
            plt.xticks(rotation=45, ha='right', fontsize=12)
            plt.yticks(fontsize=12, rotation=0)
            plt.tight_layout(rect=[0, 0.08, 1, 1])
            caption = ("Fig X: Heatmap showing gene-pathway overlaps. Each cell indicates the binary presence (1) "
                       "or absence (0) of a gene in a specific Alzheimer’s-related pathway.")
            plt.figtext(0.5, 0.01, caption, wrap=True, horizontalalignment='center', fontsize=11)
            plt.savefig(results_dir / 'pathway_gene_heatmap_viridis.png', dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved professional gene-pathway heatmap with viridis colormap")

def save_enrichment_results(enrichment_results: pd.DataFrame, 
                          results_dir: Path,
                          analysis_name: str = "pathway_enrichment"):
    """
    Save enrichment results to files.
    
    Args:
        enrichment_results: DataFrame with enrichment results
        results_dir: Directory to save results
        analysis_name: Name for the analysis
    """
    logger = logging.getLogger(__name__)
    
    # Save as CSV
    csv_path = results_dir / f'{analysis_name}_results.csv'
    enrichment_results.to_csv(csv_path, index=False)
    logger.info(f"Saved enrichment results to {csv_path}")
    
    # Save as JSON
    json_path = results_dir / f'{analysis_name}_results.json'
    enrichment_results.to_json(json_path, orient='records', indent=2)
    logger.info(f"Saved enrichment results to {json_path}")
    
    # Create summary report
    summary_path = results_dir / f'{analysis_name}_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("PATHWAY ENRICHMENT ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Pathways Analyzed: {len(enrichment_results)}\n")
        f.write(f"Significant Pathways (P < 0.05): {len(enrichment_results[enrichment_results['P_Value'] < 0.05])}\n")
        f.write(f"Significant Pathways (FDR < 0.05): {len(enrichment_results[enrichment_results['Adjusted_P_Value'] < 0.05])}\n\n")
        
        f.write("TOP 10 ENRICHED PATHWAYS:\n")
        f.write("-" * 30 + "\n")
        for idx, row in enrichment_results.head(10).iterrows():
            f.write(f"{row['Pathway']}:\n")
            f.write(f"  - Enrichment Ratio: {row['Enrichment_Ratio']:.3f}\n")
            f.write(f"  - P-value: {row['P_Value']:.2e}\n")
            f.write(f"  - Overlap Genes: {row['Overlap_Genes']}\n\n")
    
    logger.info(f"Saved summary report to {summary_path}")

def main():
    """Main function to run pathway enrichment analysis."""
    
    # Setup
    parser = argparse.ArgumentParser(description="Pathway Enrichment Analysis")
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config YAML file')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    results_dir = Path(config['pathway_enrichment_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(results_dir)
    logger.info("Starting pathway enrichment analysis")
    
    try:
        # Load feature importance data
        feature_data = load_feature_importance_data(Path(config['data_path']))
        
        if not feature_data:
            logger.warning("No feature importance data found. Creating synthetic data for demonstration.")
            # Create synthetic feature importance data
            synthetic_genes = ['APP', 'APOE', 'TREM2', 'CD33', 'MAPT', 'BACE1', 'PSEN1', 'IL1B', 'TNF', 'SOD1']
            feature_data['synthetic_importance'] = pd.DataFrame({
                'gene': synthetic_genes,
                'importance': np.random.rand(len(synthetic_genes))
            }).sort_values('importance', ascending=False)
        
        # Get gene names from feature importance data
        all_genes = set()
        for df in feature_data.values():
            if 'gene' in df.columns:
                all_genes.update(df['gene'].tolist())
            elif 'feature' in df.columns:
                all_genes.update(df['feature'].tolist())
        
        gene_names = list(all_genes)
        logger.info(f"Found {len(gene_names)} unique genes")
        
        # Create pathway data
        pathway_data = create_synthetic_pathway_data(gene_names)
        logger.info(f"Created {len(pathway_data)} pathways for analysis")
        
        # Perform enrichment analysis for each feature importance dataset
        for dataset_name, feature_df in feature_data.items():
            logger.info(f"Analyzing dataset: {dataset_name}")
            
            # Get top genes (assuming top 20% or top 50 genes)
            if 'gene' in feature_df.columns:
                gene_col = 'gene'
            elif 'feature' in feature_df.columns:
                gene_col = 'feature'
            else:
                continue
            
            # Sort by importance and get top genes
            if 'importance' in feature_df.columns:
                top_genes = feature_df.nlargest(min(50, len(feature_df)), 'importance')[gene_col].tolist()
            else:
                top_genes = feature_df[gene_col].head(50).tolist()
            
            logger.info(f"Using top {len(top_genes)} genes for enrichment analysis")
            
            # Perform enrichment
            enrichment_results = perform_pathway_enrichment(
                gene_list=top_genes,
                background_genes=gene_names,
                pathway_data=pathway_data
            )
            
            if len(enrichment_results) > 0:
                # Create visualizations
                dataset_dir = results_dir / dataset_name
                dataset_dir.mkdir(exist_ok=True)
                
                create_pathway_visualizations(
                    enrichment_results, 
                    dataset_dir,
                    title=f"Pathway Enrichment - {dataset_name}"
                )
                
                # Save results
                save_enrichment_results(
                    enrichment_results,
                    dataset_dir,
                    analysis_name=f"{dataset_name}_enrichment"
                )
                
                logger.info(f"Completed analysis for {dataset_name}")
            else:
                logger.warning(f"No significant pathways found for {dataset_name}")
        
        logger.info("Pathway enrichment analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Error in pathway enrichment analysis: {e}")
        raise

if __name__ == "__main__":
    main() 