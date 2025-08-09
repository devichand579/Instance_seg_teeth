#!/usr/bin/env python3
"""
Tooth Position Heatmaps Analysis
Creates heatmaps showing where each tooth type typically appears in X-ray images.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_prepare_data():
    """Load and prepare the  metadata CSV."""
    print("Loading metadata CSV...")
    df = pd.read_csv('metadata.csv')
    print(f"Loaded {len(df)} images")
    
    # Define tooth types
    tooth_types = {
        'incisors': [11, 12, 21, 22, 31, 32, 41, 42],
        'canines': [13, 23, 33, 43],
        'premolars': [14, 15, 24, 25, 34, 35, 44, 45],
        'molars': [16, 17, 18, 26, 27, 28, 36, 37, 38, 46, 47, 48]
    }
    
    return df, tooth_types

def extract_tooth_positions(df):
    """Extract tooth position data for spatial analysis."""
    print("Extracting tooth position data...")
    
    position_data = []
    fdi_numbers = [11, 12, 13, 14, 15, 16, 17, 18,
                   21, 22, 23, 24, 25, 26, 27, 28,
                   31, 32, 33, 34, 35, 36, 37, 38,
                   41, 42, 43, 44, 45, 46, 47, 48]
    
    for idx, row in df.iterrows():
        for fdi in fdi_numbers:
            x_col = f'FDI_{fdi}_x'
            y_col = f'FDI_{fdi}_y'
            w_col = f'FDI_{fdi}_w'
            h_col = f'FDI_{fdi}_h'
            
            if all(col in df.columns for col in [x_col, y_col, w_col, h_col]):
                x = row[x_col]
                y = row[y_col]
                w = row[w_col]
                h = row[h_col]
                
                if pd.notna(x) and x != '' and pd.notna(y) and y != '':
                    try:
                        center_x = float(x) + float(w) / 2
                        center_y = float(y) + float(h) / 2
                        
                        position_data.append({
                            'fdi': fdi,
                            'center_x': center_x,
                            'center_y': center_y,
                            'area': float(w) * float(h)
                        })
                    except (ValueError, TypeError):
                        continue
    
    return pd.DataFrame(position_data)

def create_tooth_position_heatmaps(position_df, tooth_types):
    """Create heatmaps showing where each tooth type typically appears."""
    print("Creating tooth position heatmaps...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Define image dimensions (640x640 for numbering images)
    img_width, img_height = 640, 640
    
    # Create grid for heatmap
    grid_size = 32
    x_bins = np.linspace(0, img_width, grid_size)
    y_bins = np.linspace(0, img_height, grid_size)
    
    for idx, (tooth_type, fdi_list) in enumerate(tooth_types.items()):
        ax = axes[idx // 2, idx % 2]
        
        # Filter data for this tooth type
        tooth_data = position_df[position_df['fdi'].isin(fdi_list)]
        
        if len(tooth_data) > 0:
            # Create 2D histogram
            heatmap, _, _ = np.histogram2d(
                tooth_data['center_x'], 
                tooth_data['center_y'],
                bins=[x_bins, y_bins]
            )
            
            # Plot heatmap
            im = ax.imshow(heatmap.T, origin='lower', extent=[0, img_width, 0, img_height],
                          cmap='YlOrRd', alpha=0.8)
            
            # Add colorbar
            plt.colorbar(im, ax=ax, label='Frequency')
            
            ax.set_title(f'{tooth_type.title()} Position Heatmap', fontweight='bold', fontsize=14)
            ax.set_xlabel('X Position (pixels)')
            ax.set_ylabel('Y Position (pixels)')
            
            # Invert Y axis to match image coordinates
            ax.invert_yaxis()
            
            # Add sample points for reference
            sample_data = tooth_data.sample(min(100, len(tooth_data)))
            ax.scatter(sample_data['center_x'], sample_data['center_y'], 
                      alpha=0.3, s=10, color='blue', label='Sample positions')
            ax.legend()
            
            # Add statistics text
            stats_text = f'Count: {len(tooth_data)}\nAvg X: {tooth_data["center_x"].mean():.0f}\nAvg Y: {tooth_data["center_y"].mean():.0f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   bbox=dict(boxstyle="round", facecolor='white', alpha=0.8),
                   verticalalignment='top', fontsize=10)
        else:
            ax.text(0.5, 0.5, f'No data for {tooth_type}', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=16)
            ax.set_title(f'{tooth_type.title()} Position Heatmap', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('tooth_position_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: tooth_position_heatmaps.png")



def main():
    """Main function to run tooth position heatmap analysis."""
    print(" TOOTH POSITION HEATMAP ANALYSIS")

    
    # Load data
    df, tooth_types = load_and_prepare_data()
    
    # Extract position data
    position_df = extract_tooth_positions(df)
    print(f"Extracted {len(position_df)} tooth position records")
    
    # Change to plots directory
    original_dir = os.getcwd()
    os.chdir('anatomical_plots')
    
    # Create visualizations
    create_tooth_position_heatmaps(position_df, tooth_types)
    
    # Return to original directory
    os.chdir(original_dir)
    
    print("\n TOOTH POSITION HEATMAP ANALYSIS COMPLETED!")
    print("Generated files:")
    print("  • tooth_position_heatmaps.png")

if __name__ == "__main__":
    main()
