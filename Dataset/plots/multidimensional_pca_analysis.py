

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import os

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_prepare_data():
    """Load and prepare the metadata CSV."""
    print("Loading metadata CSV...")
    df = pd.read_csv('enhanced_teeth_metadata.csv')
    print(f"Loaded {len(df)} images")
    
    # Extract category from filename
    df['category'] = df['filename'].str.extract(r'(cate\d+)')
    
    return df

def prepare_feature_matrix(df):
    """Prepare feature matrix for PCA analysis."""
    print("Preparing feature matrix...")
    
    # Define all FDI numbers
    fdi_numbers = [11, 12, 13, 14, 15, 16, 17, 18,
                   21, 22, 23, 24, 25, 26, 27, 28,
                   31, 32, 33, 34, 35, 36, 37, 38,
                   41, 42, 43, 44, 45, 46, 47, 48]
    
    features = []
    feature_names = []
    metadata = []
    
    for idx, row in df.iterrows():
        feature_vector = []
        valid_features = 0
        
        # Extract features for each tooth
        for fdi in fdi_numbers:
            x_col = f'FDI_{fdi}_x'
            y_col = f'FDI_{fdi}_y'
            w_col = f'FDI_{fdi}_w'
            h_col = f'FDI_{fdi}_h'
            density_col = f'FDI_{fdi}_pixel_density'
            
            # Check if tooth data exists
            if all(col in df.columns for col in [x_col, y_col, w_col, h_col]):
                x = row[x_col]
                y = row[y_col]
                w = row[w_col]
                h = row[h_col]
                density = row.get(density_col, np.nan)
                
                if all(pd.notna(val) and val != '' for val in [x, y, w, h]):
                    try:
                        x_val = float(x)
                        y_val = float(y)
                        w_val = float(w)
                        h_val = float(h)
                        
                        # Calculate derived features
                        area = w_val * h_val
                        aspect_ratio = w_val / h_val if h_val > 0 else 0
                        
                        # Add features
                        feature_vector.extend([x_val, y_val, w_val, h_val, area, aspect_ratio])
                        
                        # Add pixel density if available
                        if pd.notna(density) and density != '':
                            try:
                                density_val = float(density)
                                feature_vector.append(density_val)
                            except (ValueError, TypeError):
                                feature_vector.append(0.0)
                        else:
                            feature_vector.append(0.0)
                        
                        valid_features += 1
                    except (ValueError, TypeError):
                        # Add zeros for missing tooth
                        feature_vector.extend([0, 0, 0, 0, 0, 0, 0])
                else:
                    # Add zeros for missing tooth
                    feature_vector.extend([0, 0, 0, 0, 0, 0, 0])
        
        # Only include samples with sufficient valid features
        if valid_features >= 5:  # At least 5 teeth detected
            features.append(feature_vector)
            metadata.append({
                'filename': row['filename'],
                'category': row['category'],
                'valid_teeth': valid_features
            })
    
    # Create feature names
    for fdi in fdi_numbers:
        feature_names.extend([
            f'FDI_{fdi}_x', f'FDI_{fdi}_y', f'FDI_{fdi}_w', f'FDI_{fdi}_h',
            f'FDI_{fdi}_area', f'FDI_{fdi}_aspect_ratio', f'FDI_{fdi}_density'
        ])
    
    features_df = pd.DataFrame(features, columns=feature_names)
    metadata_df = pd.DataFrame(metadata)
    
    print(f"Created feature matrix: {features_df.shape}")
    print(f"Features per sample: {len(feature_names)}")
    print(f"Samples with sufficient data: {len(features_df)}")
    
    return features_df, metadata_df, feature_names

def perform_pca_analysis(features_df, metadata_df):
    """Perform comprehensive PCA analysis."""
    print("Performing PCA analysis...")
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_df)
    
    # Perform PCA
    pca = PCA()
    pca_result = pca.fit_transform(features_scaled)
    
    # Calculate cumulative explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # 1. Explained variance plot
    ax1 = axes[0, 0]
    
    n_components = min(20, len(pca.explained_variance_ratio_))
    components = range(1, n_components + 1)
    
    ax1.bar(components, pca.explained_variance_ratio_[:n_components], 
            alpha=0.7, color='skyblue', label='Individual')
    ax1.plot(components, cumulative_variance[:n_components], 
             'ro-', alpha=0.8, label='Cumulative')
    
    ax1.set_title('PCA Explained Variance', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add annotations for key thresholds
    for threshold in [0.8, 0.9, 0.95]:
        idx = np.where(cumulative_variance >= threshold)[0]
        if len(idx) > 0:
            first_idx = idx[0] + 1
            ax1.axhline(y=threshold, color='red', linestyle='--', alpha=0.5)
            ax1.text(first_idx, threshold + 0.02, f'{threshold*100}% at PC{first_idx}', 
                    fontsize=10, ha='center')
    
    # 2. PCA scatter plot (PC1 vs PC2)
    ax2 = axes[0, 1]
    
    categories = metadata_df['category'].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
    
    for i, category in enumerate(categories):
        mask = metadata_df['category'] == category
        ax2.scatter(pca_result[mask, 0], pca_result[mask, 1], 
                   c=[colors[i]], label=category, alpha=0.7, s=50)
    
    ax2.set_title('PCA: PC1 vs PC2 by Category', fontweight='bold', fontsize=14)
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # 3. PCA scatter plot (PC2 vs PC3)
    ax3 = axes[0, 2]
    
    for i, category in enumerate(categories):
        mask = metadata_df['category'] == category
        ax3.scatter(pca_result[mask, 1], pca_result[mask, 2], 
                   c=[colors[i]], label=category, alpha=0.7, s=50)
    
    ax3.set_title('PCA: PC2 vs PC3 by Category', fontweight='bold', fontsize=14)
    ax3.set_xlabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax3.set_ylabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%} variance)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Feature loadings heatmap
    ax4 = axes[1, 0]
    
    # Get top contributing features for first 10 PCs
    n_pcs = min(10, pca.components_.shape[0])
    n_features = min(50, pca.components_.shape[1])  # Show top 50 features
    
    # Get feature importance (absolute values)
    feature_importance = np.abs(pca.components_[:n_pcs, :])
    
    # Get indices of most important features
    top_feature_indices = np.argsort(np.sum(feature_importance, axis=0))[-n_features:]
    
    # Create heatmap of loadings
    loadings_subset = pca.components_[:n_pcs, top_feature_indices]
    feature_names_subset = [features_df.columns[i] for i in top_feature_indices]
    
    # Simplify feature names for display
    simplified_names = []
    for name in feature_names_subset:
        if 'FDI_' in name:
            parts = name.split('_')
            if len(parts) >= 3:
                simplified_names.append(f"{parts[1]}_{parts[2]}")
            else:
                simplified_names.append(name)
        else:
            simplified_names.append(name)
    
    sns.heatmap(loadings_subset, cmap='RdBu_r', center=0, 
                xticklabels=simplified_names, 
                yticklabels=[f'PC{i+1}' for i in range(n_pcs)],
                ax=ax4, cbar_kws={"shrink": .8})
    ax4.set_title('PCA Feature Loadings (Top Features)', fontweight='bold', fontsize=14)
    ax4.set_xlabel('Features')
    ax4.set_ylabel('Principal Components')
    plt.setp(ax4.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    
    # 5. Biplot (PC1 vs PC2 with feature vectors)
    ax5 = axes[1, 1]
    
    # Plot data points
    for i, category in enumerate(categories):
        mask = metadata_df['category'] == category
        ax5.scatter(pca_result[mask, 0], pca_result[mask, 1], 
                   c=[colors[i]], label=category, alpha=0.5, s=30)
    
    # Add feature vectors (for top contributing features)
    scale_factor = 3
    top_n_vectors = 10
    
    # Get top features by loading magnitude
    loading_magnitudes = np.sqrt(pca.components_[0]**2 + pca.components_[1]**2)
    top_vector_indices = np.argsort(loading_magnitudes)[-top_n_vectors:]
    
    for i in top_vector_indices:
        ax5.arrow(0, 0, 
                 pca.components_[0, i] * scale_factor,
                 pca.components_[1, i] * scale_factor,
                 head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.8)
        
        # Add feature label
        feature_name = features_df.columns[i]
        simplified_name = feature_name.split('_')[-1] if '_' in feature_name else feature_name
        ax5.text(pca.components_[0, i] * scale_factor * 1.1,
                pca.components_[1, i] * scale_factor * 1.1,
                simplified_name, fontsize=8, ha='center', va='center')
    
    ax5.set_title('PCA Biplot (PC1 vs PC2)', fontweight='bold', fontsize=14)
    ax5.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax5.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax5.grid(True, alpha=0.3)
    
    # 6. Summary statistics
    ax6 = axes[1, 2]
    ax6.axis('tight')
    ax6.axis('off')
    
    # Calculate summary statistics
    pc_80 = np.where(cumulative_variance >= 0.8)[0][0] + 1 if len(np.where(cumulative_variance >= 0.8)[0]) > 0 else len(cumulative_variance)
    pc_90 = np.where(cumulative_variance >= 0.9)[0][0] + 1 if len(np.where(cumulative_variance >= 0.9)[0]) > 0 else len(cumulative_variance)
    pc_95 = np.where(cumulative_variance >= 0.95)[0][0] + 1 if len(np.where(cumulative_variance >= 0.95)[0]) > 0 else len(cumulative_variance)
    
    summary_stats = [
        ['Metric', 'Value'],
        ['Total Features', f'{features_df.shape[1]}'],
        ['Total Samples', f'{features_df.shape[0]}'],
        ['PC1 Variance', f'{pca.explained_variance_ratio_[0]:.1%}'],
        ['PC2 Variance', f'{pca.explained_variance_ratio_[1]:.1%}'],
        ['PC3 Variance', f'{pca.explained_variance_ratio_[2]:.1%}'],
        ['PCs for 80% Variance', f'{pc_80}'],
        ['PCs for 90% Variance', f'{pc_90}'],
        ['PCs for 95% Variance', f'{pc_95}'],
        ['Categories', f'{len(categories)}']
    ]
    
    table = ax6.table(cellText=summary_stats[1:],
                     colLabels=summary_stats[0],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)
    ax6.set_title('PCA Analysis Summary', fontweight='bold', fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig('pca_multidimensional_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(" Saved: pca_multidimensional_analysis.png")
    
    return pca, pca_result, scaler

def create_advanced_pca_plots(pca, pca_result, metadata_df, features_df):
    """Create advanced PCA visualization plots."""
    print("Creating advanced PCA visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # 1. 3D PCA plot
    ax1 = axes[0, 0]
    
    # Since we can't do true 3D in this context, create a pseudo-3D view
    categories = metadata_df['category'].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
    
    # Create a projection that combines PC1, PC2, PC3
    projection = pca_result[:, 0] + 0.5 * pca_result[:, 2]  # Combine PC1 and PC3
    
    for i, category in enumerate(categories):
        mask = metadata_df['category'] == category
        ax1.scatter(projection[mask], pca_result[mask, 1], 
                   c=[colors[i]], label=category, alpha=0.7, s=50)
    
    ax1.set_title('PCA Projection (PC1+0.5*PC3 vs PC2)', fontweight='bold', fontsize=14)
    ax1.set_xlabel('PC1 + 0.5*PC3 Projection')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. PCA by number of valid teeth
    ax2 = axes[0, 1]
    
    valid_teeth_groups = metadata_df['valid_teeth'].value_counts().index[:5]  # Top 5 groups
    
    for teeth_count in valid_teeth_groups:
        mask = metadata_df['valid_teeth'] == teeth_count
        ax2.scatter(pca_result[mask, 0], pca_result[mask, 1], 
                   label=f'{teeth_count} teeth', alpha=0.7, s=50)
    
    ax2.set_title('PCA by Number of Valid Teeth', fontweight='bold', fontsize=14)
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Feature contribution analysis
    ax3 = axes[0, 2]
    
    # Analyze which types of features contribute most to each PC
    feature_types = ['x', 'y', 'w', 'h', 'area', 'aspect_ratio', 'density']
    pc_contributions = {ftype: [] for ftype in feature_types}
    
    for i, feature_name in enumerate(features_df.columns):
        for ftype in feature_types:
            if ftype in feature_name:
                # Sum absolute contributions across first 5 PCs
                contribution = np.sum(np.abs(pca.components_[:5, i]))
                pc_contributions[ftype].append(contribution)
                break
    
    # Calculate average contributions
    avg_contributions = {ftype: np.mean(contribs) if contribs else 0 
                        for ftype, contribs in pc_contributions.items()}
    
    feature_types_clean = list(avg_contributions.keys())
    contributions = list(avg_contributions.values())
    
    bars = ax3.bar(feature_types_clean, contributions, color='lightcoral', alpha=0.8)
    ax3.set_title('Average Feature Type Contributions (PC1-5)', fontweight='bold', fontsize=14)
    ax3.set_xlabel('Feature Type')
    ax3.set_ylabel('Average Absolute Contribution')
    ax3.tick_params(axis='x', rotation=45)
    
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 4. Outlier detection using PCA
    ax4 = axes[1, 0]
    
    # Calculate Mahalanobis distance in PCA space (using first 10 PCs)
    n_pcs = min(10, pca_result.shape[1])
    pca_subset = pca_result[:, :n_pcs]
    
    # Calculate distances from center
    center = np.mean(pca_subset, axis=0)
    distances = np.sqrt(np.sum((pca_subset - center)**2, axis=1))
    
    # Identify outliers (top 5%)
    outlier_threshold = np.percentile(distances, 95)
    outliers = distances > outlier_threshold
    
    # Plot all points
    ax4.scatter(pca_result[~outliers, 0], pca_result[~outliers, 1], 
               c='blue', alpha=0.6, s=30, label='Normal')
    ax4.scatter(pca_result[outliers, 0], pca_result[outliers, 1], 
               c='red', alpha=0.8, s=60, label='Outliers')
    
    ax4.set_title('PCA Outlier Detection', fontweight='bold', fontsize=14)
    ax4.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax4.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. PC evolution plot
    ax5 = axes[1, 1]
    
    # Show how the first few PCs change across categories
    categories_sorted = sorted(metadata_df['category'].unique())
    pc_means = []
    
    for category in categories_sorted:
        mask = metadata_df['category'] == category
        if np.sum(mask) > 0:
            pc_mean = np.mean(pca_result[mask, :3], axis=0)  # First 3 PCs
            pc_means.append(pc_mean)
        else:
            pc_means.append([0, 0, 0])
    
    pc_means = np.array(pc_means)
    
    x_pos = range(len(categories_sorted))
    width = 0.25
    
    ax5.bar([x - width for x in x_pos], pc_means[:, 0], width, label='PC1', alpha=0.8)
    ax5.bar(x_pos, pc_means[:, 1], width, label='PC2', alpha=0.8)
    ax5.bar([x + width for x in x_pos], pc_means[:, 2], width, label='PC3', alpha=0.8)
    
    ax5.set_title('Average PC Values by Category', fontweight='bold', fontsize=14)
    ax5.set_xlabel('Category')
    ax5.set_ylabel('Average PC Value')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(categories_sorted, rotation=45)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Reconstruction error analysis
    ax6 = axes[1, 2]
    
    # Calculate reconstruction error for different numbers of components
    n_components_range = range(1, min(21, pca_result.shape[1]), 2)
    reconstruction_errors = []
    
    for n_comp in n_components_range:
        # Reconstruct using n components
        pca_subset = pca_result[:, :n_comp]
        # Pad with zeros for missing components
        pca_padded = np.zeros((pca_subset.shape[0], pca.components_.shape[0]))
        pca_padded[:, :n_comp] = pca_subset
        reconstructed = pca.inverse_transform(pca_padded)
        
        # Calculate mean squared error
        original_scaled = StandardScaler().fit_transform(features_df)
        mse = np.mean((original_scaled - reconstructed)**2)
        reconstruction_errors.append(mse)
    
    ax6.plot(n_components_range, reconstruction_errors, 'bo-', alpha=0.8)
    ax6.set_title('PCA Reconstruction Error', fontweight='bold', fontsize=14)
    ax6.set_xlabel('Number of Principal Components')
    ax6.set_ylabel('Mean Squared Error')
    ax6.grid(True, alpha=0.3)
    
    # Add annotation for elbow point
    if len(reconstruction_errors) > 2:
        # Find elbow using second derivative
        second_deriv = np.diff(reconstruction_errors, 2)
        if len(second_deriv) > 0:
            elbow_idx = np.argmax(second_deriv) + 2
            if elbow_idx < len(n_components_range):
                elbow_point = n_components_range[elbow_idx]
                ax6.axvline(x=elbow_point, color='red', linestyle='--', alpha=0.7)
                ax6.text(elbow_point + 0.5, max(reconstruction_errors) * 0.8, 
                        f'Elbow: {elbow_point}', rotation=90, fontsize=10)
    
    plt.tight_layout()
    plt.savefig('advanced_pca_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: advanced_pca_analysis.png")


def main():
    """Main function to run multi-dimensional PCA analysis."""
    print(" MULTI-DIMENSIONAL PCA ANALYSIS")
    print("=" * 50)
    
    # Create correlation_plots directory
    if not os.path.exists('correlation_plots'):
        os.makedirs('correlation_plots')
    
    # Load data
    df = load_and_prepare_data()
    
    # Prepare feature matrix
    features_df, metadata_df, feature_names = prepare_feature_matrix(df)
    
    # Change to plots directory
    original_dir = os.getcwd()
    os.chdir('correlation_plots')
    
    # Perform PCA analysis
    pca, pca_result, scaler = perform_pca_analysis(features_df, metadata_df)
    
    # Create advanced PCA plots
    create_advanced_pca_plots(pca, pca_result, metadata_df, features_df)
    

    
    # Return to original directory
    os.chdir(original_dir)
    
    print("\n MULTI-DIMENSIONAL PCA ANALYSIS COMPLETED!")
    print("Generated files:")
    print("  • pca_multidimensional_analysis.png")
    print("  • advanced_pca_analysis.png")

if __name__ == "__main__":
    main()
