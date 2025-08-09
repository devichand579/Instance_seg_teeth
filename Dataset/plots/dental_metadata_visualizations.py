#!/usr/bin/env python3
"""
Comprehensive visualization script for dental metadata analysis.
Creates statistical and distribution plots across 425 dental X-ray images.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import os

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_prepare_data():
    """Load and prepare the metadata CSV for analysis."""
    print("Loading metadata CSV...")
    
    df = pd.read_csv('metadata.csv')
    print(f"Loaded {len(df)} images")
    
    # Extract category from filename
    df['category'] = df['filename'].str.extract(r'(cate\d+)')
    
    # FDI tooth numbers organized by type
    tooth_types = {
        'incisors': [11, 12, 21, 22, 31, 32, 41, 42],
        'canines': [13, 23, 33, 43],
        'premolars': [14, 15, 24, 25, 34, 35, 44, 45],
        'molars': [16, 17, 18, 26, 27, 28, 36, 37, 38, 46, 47, 48]
    }
    
    # Jaw quadrants
    quadrants = {
        'upper_right': [11, 12, 13, 14, 15, 16, 17, 18],
        'upper_left': [21, 22, 23, 24, 25, 26, 27, 28],
        'lower_left': [31, 32, 33, 34, 35, 36, 37, 38],
        'lower_right': [41, 42, 43, 44, 45, 46, 47, 48]
    }
    
    return df, tooth_types, quadrants



def analyze_pixel_densities_by_tooth_type(df, tooth_types):
    """Analyze pixel density distributions by tooth type."""
    print("Analyzing pixel densities by tooth type...")
    
    density_data = []
    
    for tooth_type, fdi_list in tooth_types.items():
        for fdi in fdi_list:
            density_col = f'FDI_{fdi}_pixel_density'
            if density_col in df.columns:
                densities = df[density_col].dropna()
                densities = densities[densities != '']
                
                for density in densities:
                    try:
                        density_data.append({
                            'tooth_type': tooth_type,
                            'fdi': fdi,
                            'pixel_density': float(density)
                        })
                    except (ValueError, TypeError):
                        continue
    
    return pd.DataFrame(density_data)

def plot_pixel_density_distributions(density_df):
    """Create box plots of pixel density by tooth type."""
    print("Creating pixel density distribution plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Box plot by tooth type
    ax1 = axes[0, 0]
    sns.boxplot(data=density_df, x='tooth_type', y='pixel_density', ax=ax1)
    ax1.set_title('Pixel Density Distribution by Tooth Type', fontweight='bold')
    ax1.set_xlabel('Tooth Type')
    ax1.set_ylabel('Pixel Density')
    ax1.tick_params(axis='x', rotation=45)
    
    # Violin plot for more detailed distribution
    ax2 = axes[0, 1]
    sns.violinplot(data=density_df, x='tooth_type', y='pixel_density', ax=ax2)
    ax2.set_title('Pixel Density Distribution (Detailed)', fontweight='bold')
    ax2.set_xlabel('Tooth Type')
    ax2.set_ylabel('Pixel Density')
    ax2.tick_params(axis='x', rotation=45)
    
    # Histogram of all densities
    ax3 = axes[1, 0]
    ax3.hist(density_df['pixel_density'], bins=50, alpha=0.7, color='green', edgecolor='black')
    ax3.set_title('Overall Pixel Density Distribution', fontweight='bold')
    ax3.set_xlabel('Pixel Density')
    ax3.set_ylabel('Frequency')
    ax3.grid(True, alpha=0.3)
    
    # Average density by tooth type
    ax4 = axes[1, 1]
    avg_density = density_df.groupby('tooth_type')['pixel_density'].mean().sort_values(ascending=False)
    bars = ax4.bar(avg_density.index, avg_density.values, color='orange', alpha=0.8)
    ax4.set_title('Average Pixel Density by Tooth Type', fontweight='bold')
    ax4.set_xlabel('Tooth Type')
    ax4.set_ylabel('Average Pixel Density')
    ax4.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.0001,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('pixel_density_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: pixel_density_distributions.png")

def analyze_tooth_sizes(df):
    """Analyze tooth size distributions."""
    print("Analyzing tooth size distributions...")
    
    size_data = []
    fdi_numbers = [11, 12, 13, 14, 15, 16, 17, 18,
                   21, 22, 23, 24, 25, 26, 27, 28,
                   31, 32, 33, 34, 35, 36, 37, 38,
                   41, 42, 43, 44, 45, 46, 47, 48]
    
    for fdi in fdi_numbers:
        w_col = f'FDI_{fdi}_w'
        h_col = f'FDI_{fdi}_h'
        
        if w_col in df.columns and h_col in df.columns:
            widths = df[w_col].dropna()
            heights = df[h_col].dropna()
            
            # Remove empty strings and convert to numeric
            widths = widths[widths != '']
            heights = heights[heights != '']
            
            for w, h in zip(widths, heights):
                try:
                    width = float(w)
                    height = float(h)
                    area = width * height
                    
                    size_data.append({
                        'fdi': fdi,
                        'width': width,
                        'height': height,
                        'area': area
                    })
                except (ValueError, TypeError):
                    continue
    
    return pd.DataFrame(size_data)

def plot_tooth_size_histograms(size_df, tooth_types):
    """Create histograms of tooth size distributions."""
    print("Creating tooth size distribution plots...")
    
    # Add tooth type to size data
    size_df['tooth_type'] = size_df['fdi'].apply(
        lambda x: next((ttype for ttype, fdis in tooth_types.items() if x in fdis), 'unknown')
    )
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Width distribution by tooth type
    ax1 = axes[0, 0]
    for tooth_type in tooth_types.keys():
        data = size_df[size_df['tooth_type'] == tooth_type]['width']
        ax1.hist(data, alpha=0.6, label=tooth_type, bins=20)
    ax1.set_title('Tooth Width Distribution by Type', fontweight='bold')
    ax1.set_xlabel('Width (pixels)')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Height distribution by tooth type
    ax2 = axes[0, 1]
    for tooth_type in tooth_types.keys():
        data = size_df[size_df['tooth_type'] == tooth_type]['height']
        ax2.hist(data, alpha=0.6, label=tooth_type, bins=20)
    ax2.set_title('Tooth Height Distribution by Type', fontweight='bold')
    ax2.set_xlabel('Height (pixels)')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Area distribution by tooth type
    ax3 = axes[0, 2]
    for tooth_type in tooth_types.keys():
        data = size_df[size_df['tooth_type'] == tooth_type]['area']
        ax3.hist(data, alpha=0.6, label=tooth_type, bins=20)
    ax3.set_title('Tooth Area Distribution by Type', fontweight='bold')
    ax3.set_xlabel('Area (pixels²)')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Box plots for size comparison
    ax4 = axes[1, 0]
    sns.boxplot(data=size_df, x='tooth_type', y='width', ax=ax4)
    ax4.set_title('Width Comparison by Tooth Type', fontweight='bold')
    ax4.tick_params(axis='x', rotation=45)
    
    ax5 = axes[1, 1]
    sns.boxplot(data=size_df, x='tooth_type', y='height', ax=ax5)
    ax5.set_title('Height Comparison by Tooth Type', fontweight='bold')
    ax5.tick_params(axis='x', rotation=45)
    
    ax6 = axes[1, 2]
    sns.boxplot(data=size_df, x='tooth_type', y='area', ax=ax6)
    ax6.set_title('Area Comparison by Tooth Type', fontweight='bold')
    ax6.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('tooth_size_histograms.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(" Saved: tooth_size_histograms.png")



def main():
    """Main function to run all visualizations."""
    print(" DENTAL METADATA VISUALIZATION SUITE")
    print("=" * 50)
    
    # Load data first
    df, tooth_types = load_and_prepare_data()
     
    # 1. Pixel Density Distributions
    density_df = analyze_pixel_densities_by_tooth_type(df, tooth_types)
    plot_pixel_density_distributions(density_df)
    
    # 2. Tooth Size Histograms
    size_df = analyze_tooth_sizes(df)
    plot_tooth_size_histograms(size_df, tooth_types)
 
    
    print("\n ALL VISUALIZATIONS COMPLETED!")
    print("\nGenerated files:")
    print("  • pixel_density_distributions.png") 
    print("  • tooth_size_histograms.png")

if __name__ == "__main__":
    main()
