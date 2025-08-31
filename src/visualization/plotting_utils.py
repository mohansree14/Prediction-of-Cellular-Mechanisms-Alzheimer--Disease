"""
Professional Plotting Utilities for Single-Cell RNA Sequencing Analysis
High-quality, publication-ready visualizations with clear legends and enhanced readability

Author: [Your Name]
Date: December 2024
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set professional plotting style
plt.style.use('default')
sns.set_palette("husl")

class ProfessionalPlottingUtils:
    """
    Professional plotting utilities for publication-ready visualizations
    """
    
    def __init__(self, config=None):
        """
        Initialize plotting utilities with professional settings
        
        Args:
            config: Configuration dictionary for plotting settings
        """
        self.config = config or {}
        self.setup_professional_style()
        
        # Professional color palette
        self.colors = {
            'primary': '#1f77b4',      # Blue
            'secondary': '#ff7f0e',    # Orange
            'tertiary': '#2ca02c',     # Green
            'quaternary': '#d62728',   # Red
            'quinary': '#9467bd',      # Purple
            'senary': '#8c564b',       # Brown
            'septenary': '#e377c2',    # Pink
            'octonary': '#7f7f7f',     # Gray
            'nonary': '#bcbd22',       # Olive
            'denary': '#17becf'        # Cyan
        }
        
        # Cell type colors for consistency
        self.cell_type_colors = {
            'Oligodendrocytes': '#1f77b4',
            'Astrocytes': '#ff7f0e',
            'Excitatory neurons': '#2ca02c',
            'Inhibitory neurons': '#d62728',
            'Oligodendrocyte precursor cells': '#9467bd',
            'Endothelial cells': '#8c564b',
            'Microglia': '#e377c2',
            'Pericytes': '#7f7f7f',
            'NK cells': '#bcbd22',
            'unknown': '#17becf'
        }
    
    def setup_professional_style(self):
        """Setup professional plotting style"""
        # Set font sizes for publication
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 11,
            'figure.titlesize': 18,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
            'lines.linewidth': 2,
            'axes.linewidth': 1.5,
            'grid.linewidth': 0.5,
            'grid.alpha': 0.3
        })
        
        # Set seaborn style
        sns.set_style("whitegrid", {
            'grid.linestyle': '--',
            'grid.alpha': 0.3,
            'axes.facecolor': 'white',
            'axes.edgecolor': 'black',
            'axes.linewidth': 1.5
        })
    
    def create_quality_control_dashboard(self, data, save_path="results/figures/quality_control/"):
        """
        Create professional quality control dashboard
        
        Args:
            data: DataFrame with scRNA-seq data
            save_path: Path to save the figure
        """
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Single-Cell RNA Sequencing Quality Control Dashboard', 
                    fontsize=20, fontweight='bold', y=0.95)
        
        # 1. RNA Count vs Feature Count Scatter Plot
        ax1 = axes[0, 0]
        scatter = ax1.scatter(data['nCount_RNA'], data['nFeature_RNA'], 
                            alpha=0.6, s=20, c=self.colors['primary'])
        ax1.set_xlabel('Number of RNA Molecules', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Number of Features', fontsize=14, fontweight='bold')
        ax1.set_title('RNA Count vs Feature Count', fontsize=16, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(data['nCount_RNA'], data['nFeature_RNA'], 1)
        p = np.poly1d(z)
        ax1.plot(data['nCount_RNA'], p(data['nCount_RNA']), 
                color=self.colors['secondary'], linewidth=3, linestyle='--', 
                label=f'Correlation: {np.corrcoef(data["nCount_RNA"], data["nFeature_RNA"])[0,1]:.3f}')
        ax1.legend(fontsize=12)
        
        # 2. RNA Count Distribution by Cell Type
        ax2 = axes[0, 1]
        cell_types = data['marker_cell_type'].value_counts().head(5).index
        for i, cell_type in enumerate(cell_types):
            cell_data = data[data['marker_cell_type'] == cell_type]['nCount_RNA']
            ax2.hist(cell_data, alpha=0.7, bins=30, 
                    color=list(self.cell_type_colors.values())[i % len(self.cell_type_colors)],
                    label=cell_type, edgecolor='black', linewidth=0.5)
        
        ax2.set_xlabel('RNA Count', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=14, fontweight='bold')
        ax2.set_title('RNA Count Distribution by Cell Type', fontsize=16, fontweight='bold')
        ax2.legend(fontsize=11, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.set_xlim(0, data['nCount_RNA'].quantile(0.95))
        
        # 3. Feature Count Distribution
        ax3 = axes[0, 2]
        ax3.hist(data['nFeature_RNA'], bins=50, alpha=0.8, 
                color=self.colors['tertiary'], edgecolor='black', linewidth=0.5)
        ax3.axvline(data['nFeature_RNA'].mean(), color=self.colors['quaternary'], 
                   linestyle='--', linewidth=3, 
                   label=f'Mean: {data["nFeature_RNA"].mean():.0f}')
        ax3.axvline(data['nFeature_RNA'].median(), color=self.colors['quinary'], 
                   linestyle='--', linewidth=3, 
                   label=f'Median: {data["nFeature_RNA"].median():.0f}')
        ax3.set_xlabel('Number of Features', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Frequency', fontsize=14, fontweight='bold')
        ax3.set_title('Feature Count Distribution', fontsize=16, fontweight='bold')
        ax3.legend(fontsize=12)
        
        # 4. Quality Metrics Summary
        ax4 = axes[1, 0]
        metrics = ['Low Quality\n(<500 RNA)', 'Medium Quality\n(500-2000 RNA)', 'High Quality\n(>2000 RNA)']
        low_quality = (data['nCount_RNA'] < 500).sum()
        medium_quality = ((data['nCount_RNA'] >= 500) & (data['nCount_RNA'] <= 2000)).sum()
        high_quality = (data['nCount_RNA'] > 2000).sum()
        counts = [low_quality, medium_quality, high_quality]
        
        bars = ax4.bar(metrics, counts, 
                      color=[self.colors['quaternary'], self.colors['secondary'], self.colors['tertiary']], 
                      alpha=0.8, edgecolor='black', linewidth=1)
        ax4.set_ylabel('Number of Cells', fontsize=14, fontweight='bold')
        ax4.set_title('Cell Quality Distribution', fontsize=16, fontweight='bold')
        
        # Add percentage labels
        total_cells = len(data)
        for bar, count in zip(bars, counts):
            percentage = (count / total_cells) * 100
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                    f'{percentage:.1f}%\n({count:,})', ha='center', va='bottom', 
                    fontsize=12, fontweight='bold')
        
        # 5. Clustering Distribution
        ax5 = axes[1, 1]
        cluster_counts = data['seurat_clusters'].value_counts().sort_index()
        bars = ax5.bar(range(len(cluster_counts)), cluster_counts.values, 
                      alpha=0.8, color=self.colors['primary'], edgecolor='black', linewidth=1)
        ax5.set_xlabel('Cluster ID', fontsize=14, fontweight='bold')
        ax5.set_ylabel('Number of Cells', fontsize=14, fontweight='bold')
        ax5.set_title('Cells per Cluster', fontsize=16, fontweight='bold')
        ax5.set_xticks(range(len(cluster_counts)))
        ax5.set_xticklabels(cluster_counts.index, fontsize=11)
        
        # Add value labels on bars
        for bar, count in zip(bars, cluster_counts.values):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(cluster_counts.values)*0.01,
                    f'{count:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 6. Cell Type Composition
        ax6 = axes[1, 2]
        cell_type_counts = data['marker_cell_type'].value_counts()
        colors = [self.cell_type_colors.get(ct, self.colors['octonary']) for ct in cell_type_counts.index]
        
        wedges, texts, autotexts = ax6.pie(cell_type_counts.values, labels=cell_type_counts.index, 
                                          colors=colors, autopct='%1.1f%%', startangle=90,
                                          textprops={'fontsize': 11, 'fontweight': 'bold'})
        ax6.set_title('Cell Type Composition', fontsize=16, fontweight='bold')
        
        # Enhance autotext appearance
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
        
        plt.tight_layout()
        
        # Save figure
        Path(save_path).mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{save_path}/quality_control_dashboard.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Quality control dashboard saved to {save_path}/quality_control_dashboard.png")
    
    def create_cell_type_comparison_plots(self, data, save_path="results/figures/cell_types/"):
        """
        Create a multi-panel cell type comparison figure (bar, pie, box, violin) and also save each plot as a separate image.
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Cell Type Analysis and Comparison', fontsize=20, fontweight='bold', y=0.95)

        # 1. Cell Type Distribution Comparison (Bar)
        ax1 = axes[0, 0]
        marker_counts = data['marker_cell_type'].value_counts()
        scina_counts = data['scina_cell_type'].value_counts()
        predicted_counts = data['predicted.id'].value_counts()
        all_cell_types = set(marker_counts.index) | set(scina_counts.index) | set(predicted_counts.index)
        x = np.arange(len(all_cell_types))
        width = 0.25
        marker_values = [marker_counts.get(cell_type, 0) for cell_type in all_cell_types]
        scina_values = [scina_counts.get(cell_type, 0) for cell_type in all_cell_types]
        predicted_values = [predicted_counts.get(cell_type, 0) for cell_type in all_cell_types]
        bars1 = ax1.bar(x - width, marker_values, width, label='Marker-based', alpha=0.8, color=self.colors['primary'], edgecolor='black', linewidth=1)
        bars2 = ax1.bar(x, scina_values, width, label='SCINA', alpha=0.8, color=self.colors['secondary'], edgecolor='black', linewidth=1)
        bars3 = ax1.bar(x + width, predicted_values, width, label='Predicted', alpha=0.8, color=self.colors['tertiary'], edgecolor='black', linewidth=1)
        ax1.set_xlabel('Cell Types', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Number of Cells', fontsize=14, fontweight='bold')
        ax1.set_title('Cell Type Distribution by Prediction Method', fontsize=16, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(all_cell_types, rotation=45, ha='right', fontsize=11)
        ax1.legend(fontsize=12, loc='upper right')
        ax1.grid(True, alpha=0.3)
        fig_bar = plt.figure(figsize=(10, 7))
        ax_bar = fig_bar.add_subplot(111)
        ax_bar.bar(x - width, marker_values, width, label='Marker-based', alpha=0.8, color=self.colors['primary'], edgecolor='black', linewidth=1)
        ax_bar.bar(x, scina_values, width, label='SCINA', alpha=0.8, color=self.colors['secondary'], edgecolor='black', linewidth=1)
        ax_bar.bar(x + width, predicted_values, width, label='Predicted', alpha=0.8, color=self.colors['tertiary'], edgecolor='black', linewidth=1)
        ax_bar.set_xlabel('Cell Types', fontsize=14, fontweight='bold')
        ax_bar.set_ylabel('Number of Cells', fontsize=14, fontweight='bold')
        ax_bar.set_title('Cell Type Distribution by Prediction Method', fontsize=16, fontweight='bold')
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(all_cell_types, rotation=45, ha='right', fontsize=11)
        ax_bar.legend(fontsize=12, loc='upper right')
        ax_bar.grid(True, alpha=0.3)
        fig_bar.tight_layout()
        fig_bar.savefig(f"{save_path}/cell_type_comparison_bar.png", dpi=300)
        plt.close(fig_bar)

        # 2. Agreement between Methods (Pie)
        ax2 = axes[0, 1]
        agreement_data = data[['marker_cell_type', 'predicted.id']].dropna()
        agreement = (agreement_data['marker_cell_type'] == agreement_data['predicted.id']).sum()
        disagreement = len(agreement_data) - agreement
        agreement_labels = ['Agreement', 'Disagreement']
        agreement_counts = [agreement, disagreement]
        colors = [self.colors['tertiary'], self.colors['quaternary']]
        wedges, texts, autotexts = ax2.pie(agreement_counts, labels=agreement_labels, colors=colors, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
        ax2.set_title('Agreement between Marker-based and Predicted Cell Types', fontsize=16, fontweight='bold')
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        fig_pie = plt.figure(figsize=(7, 7))
        ax_pie = fig_pie.add_subplot(111)
        wedges, texts, autotexts = ax_pie.pie(agreement_counts, labels=agreement_labels, colors=colors, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
        ax_pie.set_title('Agreement between Marker-based and Predicted Cell Types', fontsize=16, fontweight='bold')
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        fig_pie.tight_layout()
        fig_pie.savefig(f"{save_path}/cell_type_comparison_pie.png", dpi=300)
        plt.close(fig_pie)

        # 3. RNA Count by Cell Type (Box)
        ax3 = axes[1, 0]
        cell_types = data['marker_cell_type'].value_counts().head(6).index
        box_data = [data[data['marker_cell_type'] == ct]['nCount_RNA'] for ct in cell_types]
        box_colors = [self.cell_type_colors.get(ct, self.colors['octonary']) for ct in cell_types]
        bp = ax3.boxplot(box_data, labels=cell_types, patch_artist=True, medianprops=dict(color="black", linewidth=2), flierprops=dict(marker='o', markerfacecolor='red', markersize=4))
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax3.set_xlabel('Cell Types', fontsize=14, fontweight='bold')
        ax3.set_ylabel('RNA Count', fontsize=14, fontweight='bold')
        ax3.set_title('RNA Count Distribution by Cell Type', fontsize=16, fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        fig_box = plt.figure(figsize=(10, 7))
        ax_box = fig_box.add_subplot(111)
        bp2 = ax_box.boxplot(box_data, labels=cell_types, patch_artist=True, medianprops=dict(color="black", linewidth=2), flierprops=dict(marker='o', markerfacecolor='red', markersize=4))
        for patch, color in zip(bp2['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax_box.set_xlabel('Cell Types', fontsize=14, fontweight='bold')
        ax_box.set_ylabel('RNA Count', fontsize=14, fontweight='bold')
        ax_box.set_title('RNA Count Distribution by Cell Type', fontsize=16, fontweight='bold')
        ax_box.tick_params(axis='x', rotation=45)
        ax_box.grid(True, alpha=0.3)
        fig_box.tight_layout()
        fig_box.savefig(f"{save_path}/cell_type_comparison_box.png", dpi=300)
        plt.close(fig_box)

        # 4. Feature Count by Cell Type (Violin)
        ax4 = axes[1, 1]
        violin_data = [data[data['marker_cell_type'] == ct]['nFeature_RNA'] for ct in cell_types]
        vp = ax4.violinplot(violin_data, positions=range(len(cell_types)))
        for i, pc in enumerate(vp['bodies']):
            pc.set_facecolor(box_colors[i])
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            pc.set_linewidth(1)
        medians = [np.median(d) for d in violin_data]
        ax4.plot(range(len(cell_types)), medians, 'ko-', linewidth=2, markersize=6, label='Median')
        ax4.set_xlabel('Cell Types', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Feature Count', fontsize=14, fontweight='bold')
        ax4.set_title('Feature Count Distribution by Cell Type', fontsize=16, fontweight='bold')
        ax4.set_xticks(range(len(cell_types)))
        ax4.set_xticklabels(cell_types, rotation=45, ha='right', fontsize=11)
        ax4.legend(fontsize=12)
        ax4.grid(True, alpha=0.3)
        fig_violin = plt.figure(figsize=(10, 7))
        ax_violin = fig_violin.add_subplot(111)
        vp2 = ax_violin.violinplot(violin_data, positions=range(len(cell_types)))
        for i, pc in enumerate(vp2['bodies']):
            pc.set_facecolor(box_colors[i])
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            pc.set_linewidth(1)
        ax_violin.plot(range(len(cell_types)), medians, 'ko-', linewidth=2, markersize=6, label='Median')
        ax_violin.set_xlabel('Cell Types', fontsize=14, fontweight='bold')
        ax_violin.set_ylabel('Feature Count', fontsize=14, fontweight='bold')
        ax_violin.set_title('Feature Count Distribution by Cell Type', fontsize=16, fontweight='bold')
        ax_violin.set_xticks(range(len(cell_types)))
        ax_violin.set_xticklabels(cell_types, rotation=45, ha='right', fontsize=11)
        ax_violin.legend(fontsize=12)
        ax_violin.grid(True, alpha=0.3)
        fig_violin.tight_layout()
        fig_violin.savefig(f"{save_path}/cell_type_comparison_violin.png", dpi=300)
        plt.close(fig_violin)

        plt.tight_layout()
        fig.savefig(f"{save_path}/cell_type_comparison.png", dpi=300)
        plt.close(fig)
        print(f"✓ Cell type comparison multi-panel and individual plots saved to {save_path}")
    
    def create_clustering_analysis_plots(self, data, save_path="results/figures/clustering/"):
        """
        Create professional clustering analysis plots, saving each as a separate image.
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)

        # 1. Cluster Size Distribution (cleaned)
        plt.figure(figsize=(12, 7))
        cluster_counts = data['seurat_clusters'].value_counts().sort_index()
        bars = plt.bar(range(len(cluster_counts)), cluster_counts.values, 
                      alpha=0.85, color=self.colors['primary'], edgecolor='black', linewidth=1)
        plt.xlabel('Cluster ID', fontsize=16, fontweight='bold')
        plt.ylabel('Number of Cells', fontsize=16, fontweight='bold')
        plt.title('Cells per Cluster', fontsize=20, fontweight='bold', pad=20)
        # Show only every Nth label if too many clusters
        N = max(1, len(cluster_counts)//30)
        xticks = range(len(cluster_counts))
        plt.xticks(xticks, [str(c) if i % N == 0 else '' for i, c in enumerate(cluster_counts.index)], fontsize=12, rotation=45, ha='right')
        plt.yticks(fontsize=14)
        plt.grid(True, axis='y', alpha=0.3)
        # Add value labels only for bars above a threshold
        threshold = max(cluster_counts.values) * 0.15
        for bar, count in zip(bars, cluster_counts.values):
            if count > threshold:
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(cluster_counts.values)*0.01,
                         f'{count:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{save_path}/cells_per_cluster.png", dpi=300)
        plt.close()

        # 2. Cell Types within Clusters (Heatmap)
        plt.figure(figsize=(10, 6))
        top_clusters = cluster_counts.head(8).index
        top_cell_types = data['marker_cell_type'].value_counts().head(6).index
        heatmap_data = []
        for cluster in top_clusters:
            cluster_data = data[data['seurat_clusters'] == cluster]
            cluster_cell_types = cluster_data['marker_cell_type'].value_counts()
            row = [cluster_cell_types.get(cell_type, 0) for cell_type in top_cell_types]
            heatmap_data.append(row)
        heatmap_data = np.array(heatmap_data)
        im = plt.imshow(heatmap_data.T, cmap='YlOrRd', aspect='auto')
        plt.xlabel('Cluster ID', fontsize=14, fontweight='bold')
        plt.ylabel('Cell Type', fontsize=14, fontweight='bold')
        plt.title('Cell Type Composition by Cluster', fontsize=16, fontweight='bold')
        plt.xticks(range(len(top_clusters)), top_clusters, fontsize=11)
        plt.yticks(range(len(top_cell_types)), top_cell_types, fontsize=11)
        cbar = plt.colorbar(im)
        cbar.set_label('Number of Cells', fontsize=12, fontweight='bold')
        for i in range(len(top_clusters)):
            for j in range(len(top_cell_types)):
                plt.text(i, j, heatmap_data[i, j], ha="center", va="center", 
                         color="black", fontsize=10, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{save_path}/cell_type_composition_by_cluster.png", dpi=300)
        plt.close()

        # 3. RNA Count vs Feature Count by Cluster
        plt.figure(figsize=(10, 6))
        unique_clusters = data['seurat_clusters'].unique()
        colors_cluster = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))
        for i, cluster in enumerate(unique_clusters):
            cluster_data = data[data['seurat_clusters'] == cluster]
            plt.scatter(cluster_data['nCount_RNA'], cluster_data['nFeature_RNA'], 
                        alpha=0.6, s=20, c=[colors_cluster[i]], label=f'Cluster {cluster}')
        plt.xlabel('RNA Count', fontsize=14, fontweight='bold')
        plt.ylabel('Feature Count', fontsize=14, fontweight='bold')
        plt.title('RNA vs Feature Count by Cluster', fontsize=16, fontweight='bold')
        plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{save_path}/rna_vs_feature_count_by_cluster.png", dpi=300)
        plt.close()

        # 4. Cluster Quality Metrics
        plt.figure(figsize=(10, 6))
        cluster_metrics = []
        for cluster in unique_clusters:
            cluster_data = data[data['seurat_clusters'] == cluster]
            metrics = {
                'cluster': cluster,
                'mean_rna': cluster_data['nCount_RNA'].mean(),
                'mean_features': cluster_data['nFeature_RNA'].mean(),
                'cell_count': len(cluster_data)
            }
            cluster_metrics.append(metrics)
        metrics_df = pd.DataFrame(cluster_metrics)
        ax = plt.gca()
        ax2 = ax.twinx()
        bars1 = ax.bar(metrics_df['cluster'] - 0.2, metrics_df['mean_rna'], 
                       width=0.4, alpha=0.8, color=self.colors['primary'], 
                       label='Mean RNA Count', edgecolor='black', linewidth=1)
        bars2 = ax2.bar(metrics_df['cluster'] + 0.2, metrics_df['mean_features'], 
                        width=0.4, alpha=0.8, color=self.colors['secondary'], 
                        label='Mean Feature Count', edgecolor='black', linewidth=1)
        ax.set_xlabel('Cluster ID', fontsize=14, fontweight='bold')
        ax.set_ylabel('Mean RNA Count', fontsize=14, fontweight='bold', color=self.colors['primary'])
        ax2.set_ylabel('Mean Feature Count', fontsize=14, fontweight='bold', color=self.colors['secondary'])
        plt.title('Cluster Quality Metrics', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=11)
        plt.tight_layout()
        plt.savefig(f"{save_path}/cluster_quality_metrics.png", dpi=300)
        plt.close()

        print(f"✓ Clustering analysis plots saved to {save_path} as separate images.")
    
    def create_correlation_heatmap(self, data, save_path="results/figures/"):
        """
        Create professional correlation heatmap
        
        Args:
            data: DataFrame with numeric columns
            save_path: Path to save the figure
        """
        # Select numeric columns
        numeric_cols = ['nCount_RNA', 'nFeature_RNA', 'RNA_snn_res.0.8', 'seurat_clusters']
        numeric_data = data[numeric_cols].dropna()
        
        # Calculate correlation matrix
        correlation_matrix = numeric_data.corr()
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Create heatmap with professional styling
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": .8},
                   fmt='.3f', annot_kws={'fontsize': 12, 'fontweight': 'bold'})
        
        plt.title('Correlation Heatmap of Numeric Features', 
                 fontsize=18, fontweight='bold', pad=20)
        
        # Save figure
        Path(save_path).mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{save_path}/correlation_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Correlation heatmap saved to {save_path}/correlation_heatmap.png")
    
    def create_publication_figure(self, data, save_path="results/figures/publication/"):
        """
        Create a single, focused publication-ready figure: cell type distribution bar chart.
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(10, 7))
        cell_type_counts = data['marker_cell_type'].value_counts()
        colors = [self.cell_type_colors.get(ct, self.colors['octonary']) for ct in cell_type_counts.index]
        plt.bar(cell_type_counts.index, cell_type_counts.values, color=colors, edgecolor='black', alpha=0.85)
        plt.xlabel('Cell Type', fontsize=16, fontweight='bold')
        plt.ylabel('Number of Cells', fontsize=16, fontweight='bold')
        plt.title('Cell Type Distribution', fontsize=20, fontweight='bold', pad=20)
        plt.xticks(rotation=30, ha='right', fontsize=13)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{save_path}/cell_type_distribution_bar.png", dpi=300)
        plt.close()
        print(f"✓ Publication cell type bar chart saved to {save_path}/cell_type_distribution_bar.png")

    def create_dashboard(self, data, save_path="results/figures/dashboard/"):
        Path(save_path).mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        # ... create each subplot as axes[0,0], axes[0,1], etc. ...

        # For each subplot, also save as a separate image
        for i, ax in enumerate(axes.flat):
            fig_single, ax_single = plt.subplots(figsize=(10, 7))
            # Copy the content from ax to ax_single (re-plot with same data and style)
            # Example for a bar plot:
            # ax_single.bar(... same data ...)
            # ax_single.set_title(...), etc.
            fig_single.tight_layout()
            fig_single.savefig(f"{save_path}/dashboard_plot{i+1}.png", dpi=300)
            plt.close(fig_single)

        fig.tight_layout()
        fig.savefig(f"{save_path}/dashboard.png", dpi=300)
        plt.close(fig)

# Example usage
if __name__ == "__main__":
    # Initialize plotting utilities
    plotter = ProfessionalPlottingUtils()
    
    # Load data (example)
    data = pd.read_csv("data/raw/combined_data_fixed_sample.csv", nrows=100000)
    
    # Create visualizations
    plotter.create_quality_control_dashboard(data)
    plotter.create_cell_type_comparison_plots(data)
    plotter.create_clustering_analysis_plots(data)
    plotter.create_correlation_heatmap(data)
    plotter.create_publication_figure(data)
    plotter.create_dashboard(data) 