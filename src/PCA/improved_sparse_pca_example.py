# Improved Sparse PCA Approach for Ballpushing Analysis

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, SparsePCA
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import explained_variance_score
import matplotlib.pyplot as plt
import seaborn as sns

def compare_pca_methods(data, feature_names, n_components=10):
    """
    Compare regular PCA vs Sparse PCA with different sparsity levels
    """
    results = {}
    
    # 1. Regular PCA (baseline)
    pca = PCA(n_components=n_components)
    pca_scores = pca.fit_transform(data)
    pca_loadings = pca.components_
    
    results['regular_pca'] = {
        'scores': pca_scores,
        'loadings': pca_loadings,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'n_nonzero_per_component': [len(feature_names) for _ in range(n_components)]  # All features used
    }
    
    # 2. Sparse PCA with different alpha values
    alpha_values = [0.01, 0.1, 0.5, 1.0, 2.0]  # Higher = more sparse
    
    for alpha in alpha_values:
        sparse_pca = SparsePCA(
            n_components=n_components,
            alpha=alpha,           # L1 penalty (sparsity)
            ridge_alpha=0.01,      # L2 penalty (regularization)
            max_iter=1000,         # Increase for convergence
            random_state=42
        )
        
        sparse_scores = sparse_pca.fit_transform(data)
        sparse_loadings = sparse_pca.components_
        
        # Calculate explained variance (Sparse PCA doesn't provide this directly)
        explained_var_ratio = []
        for i in range(n_components):
            scores_i = sparse_scores[:, i].reshape(-1, 1)
            reconstructed = scores_i @ sparse_loadings[i:i+1]
            var_explained = explained_variance_score(data, reconstructed)
            explained_var_ratio.append(max(0, var_explained))  # Ensure non-negative
        
        # Count non-zero loadings per component
        n_nonzero = [(sparse_loadings[i] != 0).sum() for i in range(n_components)]
        
        results[f'sparse_alpha_{alpha}'] = {
            'scores': sparse_scores,
            'loadings': sparse_loadings,
            'explained_variance_ratio': np.array(explained_var_ratio),
            'n_nonzero_per_component': n_nonzero,
            'alpha': alpha
        }
    
    return results

def analyze_sparsity_trade_offs(results, feature_names):
    """
    Analyze the trade-off between sparsity and explained variance
    """
    analysis = []
    
    for method_name, result in results.items():
        explained_var = result['explained_variance_ratio']
        n_nonzero = result['n_nonzero_per_component']
        
        analysis.append({
            'method': method_name,
            'total_explained_variance': np.sum(explained_var),
            'mean_features_per_component': np.mean(n_nonzero),
            'sparsity_ratio': 1 - (np.mean(n_nonzero) / len(feature_names)),
            'first_3_components_variance': np.sum(explained_var[:3])
        })
    
    return pd.DataFrame(analysis)

def plot_sparsity_analysis(results, feature_names):
    """
    Create visualization of sparsity vs explained variance trade-off
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Explained variance vs sparsity
    methods = []
    total_variance = []
    sparsity_ratios = []
    
    for method_name, result in results.items():
        if method_name != 'regular_pca':  # Skip regular PCA for this plot
            methods.append(method_name)
            total_variance.append(np.sum(result['explained_variance_ratio']))
            n_nonzero = np.mean(result['n_nonzero_per_component'])
            sparsity_ratios.append(1 - (n_nonzero / len(feature_names)))
    
    axes[0, 0].scatter(sparsity_ratios, total_variance, s=100, alpha=0.7)
    for i, method in enumerate(methods):
        axes[0, 0].annotate(method, (sparsity_ratios[i], total_variance[i]))
    axes[0, 0].set_xlabel('Sparsity Ratio (1 - features_used/total_features)')
    axes[0, 0].set_ylabel('Total Explained Variance')
    axes[0, 0].set_title('Sparsity vs Explained Variance Trade-off')
    
    # Plot 2: Component-wise explained variance
    for method_name, result in results.items():
        if 'alpha' in method_name or method_name == 'regular_pca':
            axes[0, 1].plot(range(1, len(result['explained_variance_ratio'])+1), 
                           result['explained_variance_ratio'], 
                           marker='o', label=method_name)
    axes[0, 1].set_xlabel('Component Number')
    axes[0, 1].set_ylabel('Explained Variance Ratio')
    axes[0, 1].set_title('Explained Variance by Component')
    axes[0, 1].legend()
    
    # Plot 3: Number of features per component
    for method_name, result in results.items():
        if 'alpha' in method_name or method_name == 'regular_pca':
            axes[1, 0].plot(range(1, len(result['n_nonzero_per_component'])+1), 
                           result['n_nonzero_per_component'], 
                           marker='s', label=method_name)
    axes[1, 0].set_xlabel('Component Number')
    axes[1, 0].set_ylabel('Number of Non-zero Features')
    axes[1, 0].set_title('Feature Usage by Component')
    axes[1, 0].legend()
    
    # Plot 4: Heatmap of best sparse loadings
    best_sparse = None
    best_balance = 0
    for method_name, result in results.items():
        if 'alpha' in method_name:
            variance = np.sum(result['explained_variance_ratio'])
            sparsity = 1 - (np.mean(result['n_nonzero_per_component']) / len(feature_names))
            balance = variance * sparsity  # Simple balance metric
            if balance > best_balance:
                best_balance = balance
                best_sparse = result
    
    if best_sparse is not None:
        # Show first 3 components of best sparse method
        loadings_subset = best_sparse['loadings'][:3, :]
        sns.heatmap(loadings_subset, 
                   xticklabels=feature_names, 
                   yticklabels=[f'PC{i+1}' for i in range(3)],
                   cmap='RdBu_r', center=0, ax=axes[1, 1])
        axes[1, 1].set_title('Best Sparse PCA Loadings (First 3 Components)')
        plt.setp(axes[1, 1].get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('/home/matthias/ballpushing_utils/outputs/pca_outputs/sparse_pca_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def select_optimal_sparse_pca(results, feature_names, variance_threshold=0.7):
    """
    Select optimal sparse PCA based on variance explained vs sparsity trade-off
    """
    candidates = []
    
    for method_name, result in results.items():
        if 'alpha' in method_name:
            total_var = np.sum(result['explained_variance_ratio'])
            if total_var >= variance_threshold:  # Must explain at least 70% variance
                sparsity = 1 - (np.mean(result['n_nonzero_per_component']) / len(feature_names))
                candidates.append({
                    'method': method_name,
                    'alpha': result['alpha'],
                    'total_variance': total_var,
                    'sparsity': sparsity,
                    'balance_score': total_var * sparsity
                })
    
    if not candidates:
        print(f"No sparse PCA method meets variance threshold of {variance_threshold}")
        return None
    
    # Select method with highest balance score
    best_method = max(candidates, key=lambda x: x['balance_score'])
    return best_method

# Example usage function that could replace your current approach
def improved_sparse_pca_analysis(data, feature_names, target_variance=0.95):
    """
    Complete improved sparse PCA analysis
    """
    print("=== Sparse PCA Analysis ===")
    
    # Compare different methods
    results = compare_pca_methods(data, feature_names)
    
    # Analyze trade-offs
    analysis_df = analyze_sparsity_trade_offs(results, feature_names)
    print("\nSparsity vs Variance Analysis:")
    print(analysis_df.round(4))
    
    # Create visualizations
    plot_sparsity_analysis(results, feature_names)
    
    # Select optimal method
    optimal = select_optimal_sparse_pca(results, feature_names, variance_threshold=0.7)
    if optimal:
        print(f"\nOptimal Sparse PCA: {optimal['method']}")
        print(f"Alpha: {optimal['alpha']}")
        print(f"Total Variance Explained: {optimal['total_variance']:.3f}")
        print(f"Sparsity Ratio: {optimal['sparsity']:.3f}")
        
        # Return the optimal results for downstream use
        optimal_result = results[optimal['method']]
        return optimal_result, analysis_df
    else:
        print("\nUsing regular PCA as fallback")
        return results['regular_pca'], analysis_df

# Recommendation for your statistical tests
def create_statistical_comparison_data(regular_pca_result, sparse_pca_result, metadata):
    """
    Create comparison datasets for statistical testing
    """
    # Create DataFrames for both methods
    regular_pca_df = pd.DataFrame(
        regular_pca_result['scores'], 
        columns=[f"PCA{i+1}" for i in range(regular_pca_result['scores'].shape[1])]
    )
    
    sparse_pca_df = pd.DataFrame(
        sparse_pca_result['scores'], 
        columns=[f"SparsePCA{i+1}" for i in range(sparse_pca_result['scores'].shape[1])]
    )
    
    # Combine with metadata
    regular_with_meta = pd.concat([metadata.reset_index(drop=True), regular_pca_df], axis=1)
    sparse_with_meta = pd.concat([metadata.reset_index(drop=True), sparse_pca_df], axis=1)
    
    return regular_with_meta, sparse_with_meta
