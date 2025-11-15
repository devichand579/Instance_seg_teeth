#!/usr/bin/env python3
"""
Tooth Size Correlation Analysis
Analyzes how different tooth sizes relate to each other through correlation matrices and relationship plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from scipy.cluster.hierarchy import dendrogram, linkage
import os

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_prepare_data():
    """Load and prepare the metadata CSV."""
    print("Loading metadata CSV...")
    df = pd.read_csv('metadata.csv')
    print(f"Loaded {len(df)} images")
    
    # Extract category from filename
    df['category'] = df['filename'].str.extract(r'(cate\d+)')
    
    return df

def extract_tooth_size_data(df):
    """Extract tooth size data for correlation analysis."""
    print("Extracting tooth size data...")
    
    # Define all FDI numbers
    fdi_numbers = [11, 12, 13, 14, 15, 16, 17, 18,
                   21, 22, 23, 24, 25, 26, 27, 28,
                   31, 32, 33, 34, 35, 36, 37, 38,
                   41, 42, 43, 44, 45, 46, 47, 48]
    
    # Create tooth size matrix
    size_data = []
    
    for idx, row in df.iterrows():
        tooth_sizes = {'filename': row['filename'], 'category': row['category']}
        
        for fdi in fdi_numbers:
            w_col = f'FDI_{fdi}_w'
            h_col = f'FDI_{fdi}_h'
            
            if w_col in df.columns and h_col in df.columns:
                w = row[w_col]
                h = row[h_col]
                
                if pd.notna(w) and w != '' and pd.notna(h) and h != '':
                    try:
                        width = float(w)
                        height = float(h)
                        area = width * height
                        
                        tooth_sizes[f'FDI_{fdi}_width'] = width
                        tooth_sizes[f'FDI_{fdi}_height'] = height
                        tooth_sizes[f'FDI_{fdi}_area'] = area
                    except (ValueError, TypeError):
                        tooth_sizes[f'FDI_{fdi}_width'] = np.nan
                        tooth_sizes[f'FDI_{fdi}_height'] = np.nan
                        tooth_sizes[f'FDI_{fdi}_area'] = np.nan
                else:
                    tooth_sizes[f'FDI_{fdi}_width'] = np.nan
                    tooth_sizes[f'FDI_{fdi}_height'] = np.nan
                    tooth_sizes[f'FDI_{fdi}_area'] = np.nan
        
        size_data.append(tooth_sizes)
    
    return pd.DataFrame(size_data)

def create_correlation_matrices(size_df):
    """Create comprehensive correlation matrices for tooth sizes."""
    print("Creating correlation matrices...")
    
    # Extract area columns for correlation analysis
    area_cols = [col for col in size_df.columns if col.endswith('_area')]
    area_data = size_df[area_cols].copy()
    
    # Clean column names for better visualization
    area_data.columns = [col.replace('FDI_', '').replace('_area', '') for col in area_data.columns]
    
    # Remove columns with too many missing values (>80% missing)
    threshold = len(area_data) * 0.2  # At least 20% data required
    area_data = area_data.dropna(axis=1, thresh=threshold)
    
    print(f"Using {len(area_data.columns)} teeth with sufficient data")
    
    # Calculate correlation matrix
    correlation_matrix = area_data.corr()
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # 1. Full correlation heatmap
    ax1 = axes[0, 0]
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8}, ax=ax1, fmt='.2f')
    ax1.set_title('Tooth Size Correlation Matrix (Area)', fontweight='bold', fontsize=14)
    ax1.set_xlabel('FDI Tooth Number')
    ax1.set_ylabel('FDI Tooth Number')
    
    # 2. Clustered correlation heatmap
    ax2 = axes[0, 1]
    
    # Perform hierarchical clustering
    linkage_matrix = linkage(1 - correlation_matrix.abs(), method='ward')
    cluster_order = dendrogram(linkage_matrix, no_plot=True)['leaves']
    
    # Reorder correlation matrix
    clustered_corr = correlation_matrix.iloc[cluster_order, cluster_order]
    
    sns.heatmap(clustered_corr, annot=True, cmap='RdYlBu_r', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8}, ax=ax2, fmt='.2f')
    ax2.set_title('Clustered Tooth Size Correlation Matrix', fontweight='bold', fontsize=14)
    ax2.set_xlabel('FDI Tooth Number (Clustered)')
    ax2.set_ylabel('FDI Tooth Number (Clustered)')
    
    # 3. Bilateral correlation analysis
    ax3 = axes[1, 0]
    
    # Define bilateral pairs
    bilateral_pairs = [
        (11, 21), (12, 22), (13, 23), (14, 24), (15, 25), (16, 26), (17, 27), (18, 28),
        (41, 31), (42, 32), (43, 33), (44, 34), (45, 35), (46, 36), (47, 37), (48, 38)
    ]
    
    bilateral_correlations = []
    bilateral_labels = []
    
    for right_fdi, left_fdi in bilateral_pairs:
        right_col = str(right_fdi)
        left_col = str(left_fdi)
        
        if right_col in area_data.columns and left_col in area_data.columns:
            # Get data for both teeth
            right_data = area_data[right_col].dropna()
            left_data = area_data[left_col].dropna()
            
            # Find common indices
            common_idx = right_data.index.intersection(left_data.index)
            
            if len(common_idx) > 10:  # Need at least 10 paired observations
                corr_coef, _ = pearsonr(right_data[common_idx], left_data[common_idx])
                bilateral_correlations.append(corr_coef)
                bilateral_labels.append(f'{right_fdi}-{left_fdi}')
    
    # Plot bilateral correlations
    bars = ax3.bar(range(len(bilateral_correlations)), bilateral_correlations, 
                   color='skyblue', alpha=0.8)
    ax3.set_title('Bilateral Tooth Size Correlations', fontweight='bold', fontsize=14)
    ax3.set_xlabel('Tooth Pairs (Right-Left)')
    ax3.set_ylabel('Correlation Coefficient')
    ax3.set_xticks(range(len(bilateral_labels)))
    ax3.set_xticklabels(bilateral_labels, rotation=45, fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    # 4. Tooth type correlation summary
    ax4 = axes[1, 1]
    
    # Define tooth types
    tooth_types = {
        'incisors': ['11', '12', '21', '22', '31', '32', '41', '42'],
        'canines': ['13', '23', '33', '43'],
        'premolars': ['14', '15', '24', '25', '34', '35', '44', '45'],
        'molars': ['16', '17', '18', '26', '27', '28', '36', '37', '38', '46', '47', '48']
    }
    
    # Calculate average correlations within and between tooth types
    type_correlations = {}
    
    for type1, teeth1 in tooth_types.items():
        for type2, teeth2 in tooth_types.items():
            # Get teeth present in data
            teeth1_present = [t for t in teeth1 if t in area_data.columns]
            teeth2_present = [t for t in teeth2 if t in area_data.columns]
            
            if teeth1_present and teeth2_present:
                correlations = []
                for t1 in teeth1_present:
                    for t2 in teeth2_present:
                        if t1 != t2:  # Don't correlate tooth with itself
                            corr_val = correlation_matrix.loc[t1, t2]
                            if not np.isnan(corr_val):
                                correlations.append(corr_val)
                
                if correlations:
                    avg_corr = np.mean(correlations)
                    type_correlations[f'{type1}-{type2}'] = avg_corr
    
    # Create heatmap for tooth type correlations
    type_names = list(tooth_types.keys())
    type_corr_matrix = np.zeros((len(type_names), len(type_names)))
    
    for i, type1 in enumerate(type_names):
        for j, type2 in enumerate(type_names):
            key = f'{type1}-{type2}'
            if key in type_correlations:
                type_corr_matrix[i, j] = type_correlations[key]
            else:
                type_corr_matrix[i, j] = np.nan
    
    type_corr_df = pd.DataFrame(type_corr_matrix, index=type_names, columns=type_names)
    
    sns.heatmap(type_corr_df, annot=True, cmap='RdYlBu_r', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8}, ax=ax4, fmt='.2f')
    ax4.set_title('Tooth Type Correlation Matrix', fontweight='bold', fontsize=14)
    ax4.set_xlabel('Tooth Type')
    ax4.set_ylabel('Tooth Type')
    
    plt.tight_layout()
    plt.savefig('tooth_size_correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: tooth_size_correlation_matrix.png")
    
    return correlation_matrix, rrea_dat

def create_detailed_correlation_analysis(correlation_matrix, area_data):
    """Create detailed correlation analysis plots."""
    print("Creating detailed correlation analysis...")
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # 1. Distribution of correlation coefficients
    ax1 = axes[0, 0]
    
    # Get upper triangle of correlation matrix (excluding diagonal)
    mask = np.triu(np.ones_like(correlation_matrix), k=1).astype(bool)
    correlations = correlation_matrix.values[mask]
    correlations = correlations[~np.isnan(correlations)]
    
    ax1.hist(correlations, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_title('Distribution of Tooth Size Correlations', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Correlation Coefficient')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)
    
    # Add statistics
    mean_corr = np.mean(correlations)
    std_corr = np.std(correlations)
    ax1.axvline(mean_corr, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_corr:.3f}')
    ax1.axvline(mean_corr + std_corr, color='orange', linestyle='--', alpha=0.8, label=f'+1σ: {mean_corr + std_corr:.3f}')
    ax1.axvline(mean_corr - std_corr, color='orange', linestyle='--', alpha=0.8, label=f'-1σ: {mean_corr - std_corr:.3f}')
    ax1.legend()
    
    # 2. Strongest correlations
    ax2 = axes[0, 1]
    
    # Find strongest positive and negative correlations
    corr_pairs = []
    for i in range(len(correlation_matrix)):
        for j in range(i+1, len(correlation_matrix)):
            corr_val = correlation_matrix.iloc[i, j]
            if not np.isnan(corr_val):
                tooth1 = correlation_matrix.index[i]
                tooth2 = correlation_matrix.columns[j]
                corr_pairs.append((f'{tooth1}-{tooth2}', corr_val))
    
    # Sort by absolute correlation value
    corr_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # Take top 15 strongest correlations
    top_correlations = corr_pairs[:15]
    
    pairs = [pair[0] for pair in top_correlations]
    values = [pair[1] for pair in top_correlations]
    colors = ['red' if v < 0 else 'blue' for v in values]
    
    bars = ax2.barh(range(len(pairs)), values, color=colors, alpha=0.7)
    ax2.set_title('Strongest Tooth Size Correlations (Top 15)', fontweight='bold', fontsize=14)
    ax2.set_xlabel('Correlation Coefficient')
    ax2.set_yticks(range(len(pairs)))
    ax2.set_yticklabels(pairs, fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax2.text(width + (0.01 if width > 0 else -0.01), bar.get_y() + bar.get_height()/2,
                f'{width:.2f}', ha='left' if width > 0 else 'right', va='center', fontsize=8)
    
    # 3. Correlation network visualization
    ax3 = axes[0, 2]
    
    # Create a simplified network of strong correlations
    strong_threshold = 0.7
    strong_correlations = [(pair[0], pair[1]) for pair in corr_pairs if abs(pair[1]) > strong_threshold]
    
    # Count connections for each tooth
    tooth_connections = {}
    for pair_name, corr_val in strong_correlations:
        tooth1, tooth2 = pair_name.split('-')
        tooth_connections[tooth1] = tooth_connections.get(tooth1, 0) + 1
        tooth_connections[tooth2] = tooth_connections.get(tooth2, 0) + 1
    
    if tooth_connections:
        teeth = list(tooth_connections.keys())
        connections = list(tooth_connections.values())
        
        bars = ax3.bar(teeth, connections, color='lightgreen', alpha=0.8)
        ax3.set_title(f'Teeth with Strong Correlations (r > {strong_threshold})', fontweight='bold', fontsize=14)
        ax3.set_xlabel('FDI Tooth Number')
        ax3.set_ylabel('Number of Strong Correlations')
        ax3.tick_params(axis='x', rotation=45)
        
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom', fontsize=10)
    else:
        ax3.text(0.5, 0.5, f'No correlations > {strong_threshold}', 
                transform=ax3.transAxes, ha='center', va='center', fontsize=14)
        ax3.set_title(f'Teeth with Strong Correlations (r > {strong_threshold})', fontweight='bold')
    
    # 4. Correlation by jaw position
    ax4 = axes[1, 0]
    
    # Define jaw positions
    upper_teeth = [str(i) for i in range(11, 29)]
    lower_teeth = [str(i) for i in range(31, 49)]
    
    # Calculate correlations within and between jaws
    upper_present = [t for t in upper_teeth if t in area_data.columns]
    lower_present = [t for t in lower_teeth if t in area_data.columns]
    
    correlations_within_upper = []
    correlations_within_lower = []
    correlations_between_jaws = []
    
    for i, t1 in enumerate(upper_present):
        for j, t2 in enumerate(upper_present):
            if i < j:
                corr_val = correlation_matrix.loc[t1, t2]
                if not np.isnan(corr_val):
                    correlations_within_upper.append(corr_val)
    
    for i, t1 in enumerate(lower_present):
        for j, t2 in enumerate(lower_present):
            if i < j:
                corr_val = correlation_matrix.loc[t1, t2]
                if not np.isnan(corr_val):
                    correlations_within_lower.append(corr_val)
    
    for t1 in upper_present:
        for t2 in lower_present:
            corr_val = correlation_matrix.loc[t1, t2]
            if not np.isnan(corr_val):
                correlations_between_jaws.append(corr_val)
    
    # Create box plot
    correlation_data = [correlations_within_upper, correlations_within_lower, correlations_between_jaws]
    correlation_labels = ['Within Upper Jaw', 'Within Lower Jaw', 'Between Jaws']
    
    box_plot = ax4.boxplot(correlation_data, labels=correlation_labels, patch_artist=True)
    colors = ['lightblue', 'lightcoral', 'lightgreen']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
    
    ax4.set_title('Tooth Size Correlations by Jaw Position', fontweight='bold', fontsize=14)
    ax4.set_ylabel('Correlation Coefficient')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # 5. Correlation vs distance analysis
    ax5 = axes[1, 1]
    
    # Define tooth positions (simplified linear arrangement)
    tooth_positions = {
        '18': 1, '17': 2, '16': 3, '15': 4, '14': 5, '13': 6, '12': 7, '11': 8,
        '21': 9, '22': 10, '23': 11, '24': 12, '25': 13, '26': 14, '27': 15, '28': 16,
        '38': 17, '37': 18, '36': 19, '35': 20, '34': 21, '33': 22, '32': 23, '31': 24,
        '41': 25, '42': 26, '43': 27, '44': 28, '45': 29, '46': 30, '47': 31, '48': 32
    }
    
    distances = []
    correlations_by_distance = []
    
    for pair_name, corr_val in corr_pairs:
        tooth1, tooth2 = pair_name.split('-')
        if tooth1 in tooth_positions and tooth2 in tooth_positions:
            distance = abs(tooth_positions[tooth1] - tooth_positions[tooth2])
            distances.append(distance)
            correlations_by_distance.append(abs(corr_val))  # Use absolute correlation
    
    if distances:
        ax5.scatter(distances, correlations_by_distance, alpha=0.6, s=30)
        ax5.set_title('Correlation vs Tooth Distance', fontweight='bold', fontsize=14)
        ax5.set_xlabel('Distance Between Teeth (positions)')
        ax5.set_ylabel('Absolute Correlation Coefficient')
        ax5.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(distances, correlations_by_distance, 1)
        p = np.poly1d(z)
        ax5.plot(sorted(distances), p(sorted(distances)), "r--", alpha=0.8, label=f'Trend: y={z[0]:.3f}x+{z[1]:.3f}')
        ax5.legend()
    
    # 6. Summary statistics table
    ax6 = axes[1, 2]
    ax6.axis('tight')
    ax6.axis('off')
    
    # Calculate summary statistics
    summary_stats = [
        ['Metric', 'Value'],
        ['Total Teeth Analyzed', f'{len(area_data.columns)}'],
        ['Total Correlations', f'{len(correlations)}'],
        ['Mean Correlation', f'{mean_corr:.3f}'],
        ['Std Correlation', f'{std_corr:.3f}'],
        ['Strongest Positive', f'{max(correlations):.3f}'],
        ['Strongest Negative', f'{min(correlations):.3f}'],
        ['Strong Correlations (>0.7)', f'{len([c for c in correlations if abs(c) > 0.7])}'],
        ['Moderate Correlations (0.3-0.7)', f'{len([c for c in correlations if 0.3 <= abs(c) <= 0.7])}'],
        ['Weak Correlations (<0.3)', f'{len([c for c in correlations if abs(c) < 0.3])}']
    ]
    
    table = ax6.table(cellText=summary_stats[1:],
                     colLabels=summary_stats[0],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)
    ax6.set_title('Correlation Analysis Summary', fontweight='bold', fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig('detailed_correlation_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(" Saved: detailed_correlation_analysis.png")

def main():
    """Main function to run tooth size correlation analysis."""
    print(" TOOTH SIZE CORRELATION ANALYSIS")
    print("=" * 50)
    
    # Create correlation_plots directory
    if not os.path.exists('correlation_plots'):
        os.makedirs('correlation_plots')
    
    # Load data
    df = load_and_prepare_data()
    
    # Extract tooth size data
    size_df = extract_tooth_size_data(df)
    print(f"Extracted tooth size data for {len(size_df)} images")
    
    
    # Create visualizations
    correlation_matrix, area_data = create_correlation_matrices(size_df)
    create_detailed_correlation_analysis(correlation_matrix, area_data)
    
    
    print("\n TOOTH SIZE CORRELATION ANALYSIS COMPLETED!")
    print("Generated files:")
    print("  • tooth_size_correlation_matrix.png")
    print("  • detailed_correlation_analysis.png") 

if __name__ == "__main__":
    main()
