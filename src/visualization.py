"""
Visualization utilities for gene expression analysis.

This module provides plotting functions for:
- PCA visualizations (2D and 3D)
- Box plots with statistical tests
- Gene distribution scatter plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import scipy.stats as stats
import os


def results_dist_plot(df, target_genes, plot_name, include_text=False, log_base=2):
    """
    Create a mean-variance scatter plot highlighting target genes.
    
    Plots all genes as background points and highlights specified target
    genes with red stars. Useful for visualizing where significant genes
    fall in the mean-variance space.
    
    Args:
        df (pd.DataFrame): Gene expression matrix (samples x genes)
        target_genes (list): Gene symbols to highlight
        plot_name (str): Output path for the plot (without extension)
        include_text (bool): Whether to annotate top 10 genes
        log_base (int): Log base for transformation (2 or 10, None for linear)
        
    Returns:
        None (saves plot to file)
    """
    from .feature_selection import get_symbol_from_id
    
    # Convert Ensembl IDs to symbols if needed
    gene_list = list(df.columns)[1:] if 'sample' in df.columns else list(df.columns)
    symbol_list = get_symbol_from_id(gene_list)
    
    # Update column names
    if 'sample' in df.columns:
        df_plot = df.copy()
        df_plot.columns = ['sample'] + symbol_list
        df_plot = df_plot.drop(columns=['sample'])
    else:
        df_plot = df.copy()
        df_plot.columns = symbol_list
    
    # Calculate mean and variance for each gene
    means_list = df_plot.mean()
    vars_list = df_plot.var()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlabel('mean', fontsize=30)
    ax.set_ylabel('variance', fontsize=30)
    plt.xticks([])
    plt.yticks([])
    
    plot_title = os.path.basename(plot_name)
    fig.suptitle(plot_title, fontsize=40)
    
    # Plot all genes
    if log_base is None:
        plt.scatter(x=means_list, y=vars_list, alpha=0.5)
    elif log_base == 10:
        plt.scatter(
            x=np.log10([i + 1 for i in means_list]),
            y=np.log10([j + 1 for j in vars_list]),
            alpha=0.5
        )
    elif log_base == 2:
        plt.scatter(
            x=np.log2([i + 1 for i in means_list]),
            y=np.log2([j + 1 for j in vars_list]),
            alpha=0.5
        )
    
    # Highlight target genes
    x_genes = []
    y_genes = []
    means_vars_dict = {}
    
    for gene in target_genes:
        if gene not in df_plot.columns:
            continue
        
        m = float(df_plot[gene].mean())
        v = float(df_plot[gene].var())
        
        if log_base is None:
            x_genes.append(m)
            y_genes.append(v)
            means_vars_dict[gene] = [m, v]
        elif log_base == 10:
            x_genes.append(np.log10(1 + m))
            y_genes.append(np.log10(1 + v))
            means_vars_dict[gene] = [np.log10(m + 1), np.log10(v + 1)]
        elif log_base == 2:
            x_genes.append(np.log2(1 + m))
            y_genes.append(np.log2(1 + v))
            means_vars_dict[gene] = [np.log2(m + 1), np.log2(v + 1)]
    
    plt.scatter(x=x_genes, y=y_genes, marker='*', color='red', s=100)
    
    # Add text annotations for top genes
    if include_text and means_vars_dict:
        sorted_items = sorted(
            means_vars_dict.items(), 
            key=lambda item: item[1][0], 
            reverse=True
        )
        top10 = sorted_items[:10]
        print('Top 10 genes:', top10)
        
        for gene, coords in top10:
            plt.annotate(gene, (float(coords[0]), float(coords[1])), fontsize=14)
    
    plt.savefig(plot_name + '.png', dpi=300)
    plt.close()


def plot_2d_pca(X, y, class_1_label, class_2_label, class_1_color, class_2_color):
    """
    Create a 2D PCA scatter plot colored by class.
    
    Args:
        X (pd.DataFrame): Feature matrix (samples x genes)
        y (np.ndarray): Binary class labels (0 or 1)
        class_1_label (str): Label for class 0
        class_2_label (str): Label for class 1
        class_1_color (str): Color for class 0 points
        class_2_color (str): Color for class 1 points
        
    Returns:
        matplotlib.pyplot: Plot object for saving
    """
    target_names = np.array([class_1_label, class_2_label])
    
    # Prepare data
    if 'sample' in list(X.columns):
        X_t = X.drop(columns=['sample']).to_numpy()
    else:
        X_t = X.to_numpy() if hasattr(X, 'to_numpy') else np.array(X)
    
    # Run PCA
    pca = PCA(n_components=2, random_state=0)
    X_r = pca.fit_transform(X_t)
    
    print(f"  PCA explained variance ratio: {pca.explained_variance_ratio_}")
    
    # Create plot
    fig = plt.figure(figsize=(8, 6))
    colors = [class_1_color, class_2_color]
    lw = 2
    
    for color, i, target_name in zip(colors, [0, 1], target_names):
        plt.scatter(
            X_r[y == i, 0], X_r[y == i, 1],
            color=color, alpha=0.8, lw=lw, label=target_name
        )
    
    plt.legend(loc="best", shadow=False, scatterpoints=1, fontsize=12)
    plt.xlabel('PC1', fontsize=14)
    plt.ylabel('PC2', fontsize=14)
    plt.xticks([])
    plt.yticks([])
    
    return plt


def plot_3d_pca(X, y, class_1_label, class_2_label, class_1_color, class_2_color):
    """
    Create a 3D PCA scatter plot colored by class.
    
    Args:
        X (pd.DataFrame): Feature matrix (samples x genes)
        y (np.ndarray): Binary class labels (0 or 1)
        class_1_label (str): Label for class 0
        class_2_label (str): Label for class 1
        class_1_color (str): Color for class 0 points
        class_2_color (str): Color for class 1 points
        
    Returns:
        matplotlib.pyplot: Plot object for saving
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    # Prepare data
    if 'sample' in list(X.columns):
        X_t = X.drop(columns=['sample']).to_numpy()
    else:
        X_t = X.to_numpy() if hasattr(X, 'to_numpy') else np.array(X)
    
    # Run PCA
    pca = PCA(n_components=3, random_state=0)
    X_pca = pca.fit_transform(X_t)
    
    Xax = X_pca[:, 0]
    Yax = X_pca[:, 1]
    Zax = X_pca[:, 2]
    
    cdict = {0: class_1_color, 1: class_2_color}
    labl = {0: class_1_label, 1: class_2_label}
    marker = {0: 'o', 1: 'o'}
    alpha = {0: 0.3, 1: 0.5}
    
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    
    fig.patch.set_facecolor('white')
    
    for l in np.unique(y):
        ix = np.where(y == l)
        ax.scatter(
            Xax[ix], Yax[ix], Zax[ix],
            c=cdict[l], s=40, label=labl[l],
            marker=marker[l], alpha=alpha[l]
        )
    
    ax.set_xlabel("PC1", fontsize=16)
    ax.set_ylabel("PC2", fontsize=16)
    ax.set_zlabel("PC3", fontsize=16)
    ax.set_box_aspect(None, zoom=0.75)
    ax.legend(fontsize=12)
    
    return plt


def plotbox_and_stats(data_, sample_key, field, treatment,
                      condition_1_value=None, condition_1_name='flight',
                      condition_2_name='nonflight', exclude_samples=None):
    """
    Create box plot comparing two conditions with statistical test.
    
    Performs independent t-test between conditions and visualizes
    the distributions.
    
    Args:
        data_ (pd.DataFrame): Data with samples and phenotype values
        sample_key (str): Column name for sample identifiers
        field (str): Column name for the variable to plot
        treatment (str): Column name for treatment/condition grouping
        condition_1_value: Value in treatment column for condition 1
        condition_1_name (str): Display name for condition 1
        condition_2_name (str): Display name for condition 2
        exclude_samples (list): Sample names to exclude
        
    Returns:
        matplotlib.pyplot: Plot object for saving
    """
    if exclude_samples is None:
        exclude_samples = []
    
    print(f'Field: {field}')
    print(f'Excluding samples: {exclude_samples}')
    
    condition_1 = f'{field}_{condition_1_name}'
    condition_2 = f'{field}_{condition_2_name}'
    
    value_dict = {condition_1: [], condition_2: []}
    results = {field: {}}
    
    for i in range(len(data_)):
        if data_.iloc[i][sample_key] in exclude_samples:
            continue
        elif treatment is None:
            # Infer from sample name prefix
            if data_.iloc[i][sample_key].startswith('F'):
                value_dict[condition_1].append(data_.iloc[i][field])
            else:
                value_dict[condition_2].append(data_.iloc[i][field])
        else:
            if data_.iloc[i][treatment] == condition_1_value:
                value_dict[condition_1].append(data_.iloc[i][field])
            else:
                value_dict[condition_2].append(data_.iloc[i][field])
    
    # Statistical test
    if len(value_dict[condition_1]) > 0 and len(value_dict[condition_2]) > 0:
        ttest_result = stats.ttest_ind(
            value_dict[condition_1],
            value_dict[condition_2],
            equal_var=False
        )
        results[field]['t-test p-value'] = float(f'{ttest_result.pvalue:.5f}')
    
    print(f'Results: {results}')
    print(f'n condition 1 = {len(value_dict[condition_1])}')
    print(f'n condition 2 = {len(value_dict[condition_2])}')
    
    # Create box plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot(value_dict.values(), showfliers=False)
    ax.set_xticklabels(value_dict.keys())
    plt.xticks(rotation=30, ha='right')
    plt.ylabel(field, fontsize=12)
    plt.tight_layout()
    
    return plt


def plot_gene_heatmap(X, y, gene_list=None, n_genes=50, cmap='RdBu_r'):
    """
    Create a heatmap of top variable genes.
    
    Args:
        X (pd.DataFrame): Expression matrix (samples x genes)
        y (np.ndarray): Class labels for sample ordering
        gene_list (list): Specific genes to include (optional)
        n_genes (int): Number of top variable genes if gene_list not provided
        cmap (str): Colormap name
        
    Returns:
        matplotlib.pyplot: Plot object
    """
    import seaborn as sns
    
    # Prepare data
    if 'sample' in X.columns:
        X_plot = X.drop(columns=['sample'])
    else:
        X_plot = X.copy()
    
    # Select genes
    if gene_list is not None:
        genes_to_plot = [g for g in gene_list if g in X_plot.columns]
        X_subset = X_plot[genes_to_plot]
    else:
        # Select top variable genes
        variances = X_plot.var()
        top_genes = variances.nlargest(n_genes).index
        X_subset = X_plot[top_genes]
    
    # Sort samples by class
    sort_idx = np.argsort(y)
    X_sorted = X_subset.iloc[sort_idx]
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(X_sorted.T, cmap=cmap, xticklabels=False, yticklabels=True, ax=ax)
    plt.xlabel('Samples', fontsize=12)
    plt.ylabel('Genes', fontsize=12)
    plt.tight_layout()
    
    return plt
