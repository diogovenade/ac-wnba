from IPython.display import display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def handle_outliers(players_df):
    # Data Quality Check: Handle extreme outliers before imputation
    print("=== PRE-IMPUTATION DATA QUALITY CHECK ===\n")

    # Check for unrealistic height values (WNBA players typically 60-80 inches)
    unrealistic_heights = (players_df['height'] < 60) & (players_df['height'] > 0)
    if unrealistic_heights.sum() > 0:
        print(f"Found {unrealistic_heights.sum()} unrealistic height values (<60 inches)")
        print("Converting to zero for proper imputation:")
        display(players_df.loc[unrealistic_heights, ['bioID', 'height', 'weight', 'pos']])
        players_df.loc[unrealistic_heights, 'height'] = 0
        print("✓ Converted to zero for imputation\n")

    # Check for unrealistic weight values (WNBA players typically 115-260 lbs)
    unrealistic_weights = (players_df['weight'] < 100) & (players_df['weight'] > 0)
    if unrealistic_weights.sum() > 0:
        print(f"Found {unrealistic_weights.sum()} unrealistic weight values (<100 lbs)")
        print("Converting to zero for proper imputation:")
        display(players_df.loc[unrealistic_weights, ['bioID', 'height', 'weight', 'pos']])
        players_df.loc[unrealistic_weights, 'weight'] = 0
        print("✓ Converted to zero for imputation\n")

    print("Pre-imputation quality check complete!")
    return players_df


def detect_outliers_zscore(players_df, threshold=3):
    # Select only height and weight columns with non-zero values
    numerical_cols = ['height', 'weight']
    df_valid = players_df[numerical_cols].replace(0, np.nan).dropna()
    
    # Calculate Z-scores
    z_scores = np.abs(stats.zscore(df_valid))
    
    # Find outliers (where any column has |z| > threshold)
    outlier_mask = (z_scores > threshold).any(axis=1)
    outliers = players_df.loc[df_valid[outlier_mask].index]
    
    print(f"Z-Score Method (threshold={threshold}):")
    print(f"Total outliers detected: {len(outliers)}")
    print(f"Percentage of dataset: {len(outliers)/len(players_df)*100:.2f}%\n")
    
    if len(outliers) > 0:
        print("Outlier Summary Statistics:")
        display(outliers[['bioID', 'pos', 'height', 'weight']].describe())
        print("\nSample Outliers:")
        display(outliers[['bioID', 'pos', 'height', 'weight']].head(10))
    
    return outliers


def detect_outliers_iqr(players_df, multiplier=1.5):
    numerical_cols = ['height', 'weight']
    df_valid = players_df[numerical_cols].replace(0, np.nan).dropna()
    
    outlier_indices = set()
    
    for col in numerical_cols:
        Q1 = df_valid[col].quantile(0.25)
        Q3 = df_valid[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        col_outliers = df_valid[(df_valid[col] < lower_bound) | (df_valid[col] > upper_bound)]
        outlier_indices.update(col_outliers.index)
        
        print(f"{col.upper()}:")
        print(f"  Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
        print(f"  Lower bound: {lower_bound:.2f}, Upper bound: {upper_bound:.2f}")
        print(f"  Outliers: {len(col_outliers)}\n")
    
    outliers = players_df.loc[list(outlier_indices)]
    
    print(f"IQR Method (multiplier={multiplier}):")
    print(f"Total unique outliers detected: {len(outliers)}")
    print(f"Percentage of dataset: {len(outliers)/len(players_df)*100:.2f}%\n")
    
    if len(outliers) > 0:
        print("Sample Outliers:")
        display(outliers[['bioID', 'pos', 'height', 'weight']].head(10))
    
    return outliers


def visualize_outliers(players_df):
    # Filter out zero/missing values for visualization
    df_valid = players_df[['height', 'weight']].replace(0, np.nan).dropna()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Box plot for height
    axes[0, 0].boxplot(df_valid['height'], vert=True)
    axes[0, 0].set_ylabel('Height (inches)')
    axes[0, 0].set_title('Height Distribution - Box Plot')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Box plot for weight
    axes[0, 1].boxplot(df_valid['weight'], vert=True)
    axes[0, 1].set_ylabel('Weight (lbs)')
    axes[0, 1].set_title('Weight Distribution - Box Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Histogram with Z-score overlay for height
    z_height = np.abs(stats.zscore(df_valid['height']))
    axes[1, 0].hist(df_valid['height'], bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[1, 0].axvline(df_valid['height'].mean(), color='red', linestyle='--', label='Mean')
    axes[1, 0].axvline(df_valid['height'].mean() + 3*df_valid['height'].std(), 
                       color='orange', linestyle='--', label='±3σ')
    axes[1, 0].axvline(df_valid['height'].mean() - 3*df_valid['height'].std(), 
                       color='orange', linestyle='--')
    axes[1, 0].set_xlabel('Height (inches)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Height Distribution with Z-Score Boundaries')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Histogram with Z-score overlay for weight
    z_weight = np.abs(stats.zscore(df_valid['weight']))
    axes[1, 1].hist(df_valid['weight'], bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[1, 1].axvline(df_valid['weight'].mean(), color='red', linestyle='--', label='Mean')
    axes[1, 1].axvline(df_valid['weight'].mean() + 3*df_valid['weight'].std(), 
                       color='orange', linestyle='--', label='±3σ')
    axes[1, 1].axvline(df_valid['weight'].mean() - 3*df_valid['weight'].std(), 
                       color='orange', linestyle='--')
    axes[1, 1].set_xlabel('Weight (lbs)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Weight Distribution with Z-Score Boundaries')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Scatter plot: Height vs Weight with outliers highlighted
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate outliers using Z-score
    z_scores = np.abs(stats.zscore(df_valid))
    outlier_mask = (z_scores > 3).any(axis=1)
    
    # Plot normal points
    ax.scatter(df_valid[~outlier_mask]['height'], 
               df_valid[~outlier_mask]['weight'],
               alpha=0.5, s=30, c='blue', label='Normal')
    
    # Plot outliers
    if outlier_mask.sum() > 0:
        ax.scatter(df_valid[outlier_mask]['height'], 
                   df_valid[outlier_mask]['weight'],
                   alpha=0.8, s=100, c='red', marker='x', label='Outliers (Z>3)')
    
    ax.set_xlabel('Height (inches)')
    ax.set_ylabel('Weight (lbs)')
    ax.set_title('Height vs Weight - Outlier Detection')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"✓ Visualizations complete")

# ==========================================
# TEAMS DATASET OUTLIER DETECTION
# ==========================================

def detect_outliers_zscore_teams(teams_df, threshold=3):
    # Select only numeric columns
    numerical_cols = teams_df.select_dtypes(include=['number']).columns.tolist()
    df_valid = teams_df[numerical_cols].dropna()
    
    if len(df_valid) == 0:
        print("No valid numeric data to analyze")
        return pd.DataFrame()
    
    # Calculate Z-scores
    z_scores = np.abs(stats.zscore(df_valid))
    
    # Find outliers (where any column has |z| > threshold)
    outlier_mask = (z_scores > threshold).any(axis=1)
    outliers = teams_df.loc[df_valid[outlier_mask].index]
    
    print(f"Z-Score Method (threshold={threshold}):")
    print(f"Total outliers detected: {len(outliers)}")
    print(f"Percentage of dataset: {len(outliers)/len(teams_df)*100:.2f}%\n")
    
    if len(outliers) > 0:
        print("Sample Outliers:")
        display_cols = ['year', 'tmID', 'name', 'won', 'lost', 'attend'] if all(c in outliers.columns for c in ['year', 'tmID', 'name', 'won', 'lost', 'attend']) else outliers.columns[:6]
        display(outliers[display_cols].head(10))
        
        # Show which columns have outliers
        outlier_counts = (z_scores > threshold).sum(axis=0)
        outlier_counts_series = pd.Series(outlier_counts, index=numerical_cols)
        outlier_cols = outlier_counts_series[outlier_counts_series > 0].sort_values(ascending=False)
        print(f"\nColumns with outliers (count):")
        print(outlier_cols.head(10))
    
    return outliers


def detect_outliers_iqr_teams(teams_df, multiplier=1.5):
    numerical_cols = teams_df.select_dtypes(include=['number']).columns.tolist()
    df_valid = teams_df[numerical_cols].dropna()
    
    if len(df_valid) == 0:
        print("No valid numeric data to analyze")
        return pd.DataFrame()
    
    outlier_indices = set()
    outlier_col_counts = {}
    
    for col in numerical_cols:
        Q1 = df_valid[col].quantile(0.25)
        Q3 = df_valid[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        col_outliers = df_valid[(df_valid[col] < lower_bound) | (df_valid[col] > upper_bound)]
        if len(col_outliers) > 0:
            outlier_indices.update(col_outliers.index)
            outlier_col_counts[col] = len(col_outliers)
    
    outliers = teams_df.loc[list(outlier_indices)]
    
    print(f"IQR Method (multiplier={multiplier}):")
    print(f"Total unique outliers detected: {len(outliers)}")
    print(f"Percentage of dataset: {len(outliers)/len(teams_df)*100:.2f}%\n")
    
    if len(outliers) > 0:
        print("Columns with most outliers:")
        for col, count in sorted(outlier_col_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {col}: {count}")
        
        print("\nSample Outliers:")
        display_cols = ['year', 'tmID', 'name', 'won', 'lost', 'attend'] if all(c in outliers.columns for c in ['year', 'tmID', 'name', 'won', 'lost', 'attend']) else outliers.columns[:6]
        display(outliers[display_cols].head(10))
    
    return outliers


def visualize_outliers_teams(teams_df):
    # Select key numeric columns for visualization
    key_cols = []
    potential_cols = ['won', 'lost', 'attend', 'o_pts', 'd_pts', 'o_reb', 'd_reb']
    
    for col in potential_cols:
        if col in teams_df.columns:
            key_cols.append(col)
    
    if len(key_cols) == 0:
        print("No suitable numeric columns for visualization")
        return
    
    # Limit to first 4 columns for visualization
    viz_cols = key_cols[:4]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, col in enumerate(viz_cols):
        df_valid = teams_df[col].dropna()
        
        if len(df_valid) == 0:
            continue
        
        # Calculate Z-scores
        z_scores = np.abs(stats.zscore(df_valid))
        mean_val = df_valid.mean()
        std_val = df_valid.std()
        
        # Box plot
        axes[idx].boxplot(df_valid, vert=True)
        axes[idx].set_ylabel(col)
        axes[idx].set_title(f'{col} Distribution - Box Plot')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Scatter plot: Wins vs Attendance (if both exist)
    if 'won' in teams_df.columns and 'attend' in teams_df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        df_valid = teams_df[['won', 'attend']].dropna()
        
        if len(df_valid) > 0:
            # Calculate outliers using Z-score
            z_scores = np.abs(stats.zscore(df_valid))
            outlier_mask = (z_scores > 3).any(axis=1)
            
            # Plot normal points
            ax.scatter(df_valid[~outlier_mask]['won'], 
                       df_valid[~outlier_mask]['attend'],
                       alpha=0.5, s=50, c='blue', label='Normal')
            
            # Plot outliers
            if outlier_mask.sum() > 0:
                ax.scatter(df_valid[outlier_mask]['won'], 
                           df_valid[outlier_mask]['attend'],
                           alpha=0.8, s=150, c='red', marker='x', label='Outliers (Z>3)')
            
            ax.set_xlabel('Wins')
            ax.set_ylabel('Attendance')
            ax.set_title('Wins vs Attendance - Outlier Detection')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
    
    print(f"✓ Visualizations complete")

def detect_outliers_zscore_players_teams(players_teams_df, threshold=3):
    """Detect outliers in players_teams dataset using Z-score method."""
    # Select key performance statistics
    performance_cols = ['points', 'rebounds', 'assists', 'steals', 'blocks', 'minutes',
                       'fgAttempted', 'fgMade', 'ftAttempted', 'ftMade', 'turnovers']
    
    # Filter to only existing columns
    numerical_cols = [col for col in performance_cols if col in players_teams_df.columns]
    df_valid = players_teams_df[numerical_cols].dropna()
    
    if len(df_valid) == 0:
        print("No valid numeric data to analyze")
        return pd.DataFrame()
    
    # Calculate Z-scores
    z_scores = np.abs(stats.zscore(df_valid))
    
    # Find outliers (where any column has |z| > threshold)
    outlier_mask = (z_scores > threshold).any(axis=1)
    outliers = players_teams_df.loc[df_valid[outlier_mask].index]
    
    print(f"Z-Score Method (threshold={threshold}):")
    print(f"Total outliers detected: {len(outliers)}")
    print(f"Percentage of dataset: {len(outliers)/len(players_teams_df)*100:.2f}%\n")
    
    if len(outliers) > 0:
        print("Sample Outliers:")
        display_cols = ['playerID', 'year', 'tmID', 'GP', 'points', 'rebounds', 'assists']
        display_cols = [col for col in display_cols if col in outliers.columns]
        display(outliers[display_cols].head(10))
        
        # Show which columns have outliers
        outlier_counts = (z_scores > threshold).sum(axis=0)
        outlier_counts_series = pd.Series(outlier_counts, index=numerical_cols)
        outlier_cols = outlier_counts_series[outlier_counts_series > 0].sort_values(ascending=False)
        print(f"\nColumns with outliers (count):")
        print(outlier_cols.head(10))
    
    return outliers


def detect_outliers_iqr_players_teams(players_teams_df, multiplier=1.5):
    """Detect outliers in players_teams dataset using IQR method."""
    performance_cols = ['points', 'rebounds', 'assists', 'steals', 'blocks', 'minutes',
                       'fgAttempted', 'fgMade', 'ftAttempted', 'ftMade', 'turnovers']
    
    numerical_cols = [col for col in performance_cols if col in players_teams_df.columns]
    df_valid = players_teams_df[numerical_cols].dropna()
    
    if len(df_valid) == 0:
        print("No valid numeric data to analyze")
        return pd.DataFrame()
    
    outlier_indices = set()
    outlier_col_counts = {}
    
    for col in numerical_cols:
        Q1 = df_valid[col].quantile(0.25)
        Q3 = df_valid[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        col_outliers = df_valid[(df_valid[col] < lower_bound) | (df_valid[col] > upper_bound)]
        if len(col_outliers) > 0:
            outlier_indices.update(col_outliers.index)
            outlier_col_counts[col] = len(col_outliers)
    
    outliers = players_teams_df.loc[list(outlier_indices)]
    
    print(f"IQR Method (multiplier={multiplier}):")
    print(f"Total unique outliers detected: {len(outliers)}")
    print(f"Percentage of dataset: {len(outliers)/len(players_teams_df)*100:.2f}%\n")
    
    if len(outliers) > 0:
        print("Columns with most outliers:")
        for col, count in sorted(outlier_col_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {col}: {count}")
        
        print("\nSample Outliers:")
        display_cols = ['playerID', 'year', 'tmID', 'GP', 'points', 'rebounds', 'assists']
        display_cols = [col for col in display_cols if col in outliers.columns]
        display(outliers[display_cols].head(10))
    
    return outliers


def visualize_outliers_players_teams(players_teams_df):
    """Visualize outliers in players_teams dataset."""
    # Select key stats for visualization
    key_cols = ['points', 'rebounds', 'assists', 'steals', 'blocks', 'minutes']
    viz_cols = [col for col in key_cols if col in players_teams_df.columns][:4]
    
    if len(viz_cols) == 0:
        print("No suitable numeric columns for visualization")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, col in enumerate(viz_cols):
        df_valid = players_teams_df[col].dropna()
        
        if len(df_valid) == 0:
            continue
        
        # Box plot
        axes[idx].boxplot(df_valid, vert=True)
        axes[idx].set_ylabel(col)
        axes[idx].set_title(f'{col} Distribution - Box Plot')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Scatter plot: Points vs Minutes (if both exist)
    if 'points' in players_teams_df.columns and 'minutes' in players_teams_df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        df_valid = players_teams_df[['points', 'minutes']].dropna()
        
        if len(df_valid) > 0:
            # Calculate outliers using Z-score
            z_scores = np.abs(stats.zscore(df_valid))
            outlier_mask = (z_scores > 3).any(axis=1)
            
            # Plot normal points
            ax.scatter(df_valid[~outlier_mask]['minutes'], 
                       df_valid[~outlier_mask]['points'],
                       alpha=0.5, s=30, c='blue', label='Normal')
            
            # Plot outliers
            if outlier_mask.sum() > 0:
                ax.scatter(df_valid[outlier_mask]['minutes'], 
                           df_valid[outlier_mask]['points'],
                           alpha=0.8, s=100, c='red', marker='x', label='Outliers (Z>3)')
            
            ax.set_xlabel('Minutes Played')
            ax.set_ylabel('Points')
            ax.set_title('Points vs Minutes - Outlier Detection')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
    
    print(f"✓ Visualizations complete")
