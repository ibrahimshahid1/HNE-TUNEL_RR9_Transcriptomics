"""
Machine learning regression models for gene expression analysis.

This module provides implementations of various regression models
with hyperparameter tuning, cross-validation, and feature importance
extraction for predicting phenotype values from gene expression.

Models included:
- ElasticNet
- Support Vector Regression (SVR)
- Ridge Regression
- Lasso Regression  
- Linear Regression
"""

import numpy as np
import pandas as pd
from itertools import islice
from sklearn.model_selection import (
    train_test_split, cross_validate, GridSearchCV, LeavePOut
)
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.feature_selection import RFE

from .preprocessing import full_transform
from .feature_selection import permutation_feature_importance, get_symbol_from_id


def run_elasticnet(y, X_array, X_orig, n_genes, score, xform_list, cv, seed, 
                   test_size=0.30, gtf_path=None):
    """
    Train ElasticNet regression model with hyperparameter tuning.
    
    ElasticNet combines L1 and L2 regularization, providing a balance
    between Lasso and Ridge regression.
    
    Args:
        y (np.ndarray): Target values
        X_array (np.ndarray): Feature matrix (not used, kept for API consistency)
        X_orig (pd.DataFrame): Original feature matrix (samples x genes)
        n_genes (int): Number of top genes to extract
        score (str): Scoring metric for optimization (e.g., 'r2')
        xform_list (list): Transformations to apply (e.g., ['tpm', 'std'])
        cv (int): Number of cross-validation folds
        seed (int): Random seed
        test_size (float): Fraction of data for testing
        gtf_path (str): Path to GTF file for TPM normalization
        
    Returns:
        tuple: (gene_dict, best_estimator, perfs_dict, pos_coefs_dict, neg_coefs_dict)
    """
    from sklearn.linear_model import ElasticNet
    
    perfs = {}
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_orig, y, test_size=test_size, random_state=seed
    )
    
    # Transform training data
    X_train = full_transform(X_train, xform_list, gtf_path)
    
    # Find best alpha via grid search
    param_grid = {'alpha': np.logspace(1, 10, 10)}
    elastic = ElasticNet(random_state=seed)
    grid_search = GridSearchCV(elastic, param_grid, cv=cv, scoring=score)
    grid_search.fit(X_train, y_train)
    best_alpha = grid_search.best_params_['alpha']
    print(f'  ElasticNet best alpha: {best_alpha}')
    
    # Train model with cross-validation
    reg = ElasticNet(alpha=best_alpha, l1_ratio=0.5, random_state=seed)
    output = cross_validate(
        reg, X_train, y_train, 
        cv=LeavePOut(p=2), 
        scoring=score, 
        return_estimator=True
    )
    
    best_estimator = output['estimator'][np.argmax(output['test_score'])]
    scores = output['test_score']
    avg_score = np.median(list(scores))
    print(f'  ElasticNet avg train score: {avg_score:.4f}')
    perfs['avg_train'] = avg_score
    
    # Transform and evaluate on test data
    X_test = full_transform(X_test, xform_list, gtf_path)
    y_pred = best_estimator.predict(X_test)
    
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'  ElasticNet test RMSE: {rmse:.2f}, R²: {r2:.2f}')
    perfs['test_rmse'] = rmse
    perfs['test_r2'] = r2
    
    # Extract feature importance
    elastic_genes = {}
    genes_list = list(X_orig.columns)
    
    # Permutation Feature Importance
    elastic_genes['pfi'] = get_symbol_from_id(
        permutation_feature_importance(
            best_estimator, X_test, y_test,
            genes=genes_list, scoring=score, n=n_genes, random_state=seed
        )
    )
    print(f'  ElasticNet PFI genes: {elastic_genes["pfi"][:5]}...')
    
    # Recursive Feature Elimination
    selector = RFE(best_estimator, n_features_to_select=n_genes, step=0.25)
    selector.fit(X_test, y_test)
    indices = selector.get_support(indices=True)
    elastic_genes['rfe'] = get_symbol_from_id([genes_list[i] for i in indices])
    
    # Coefficient-based feature importance
    coefs_dict = dict(zip(genes_list, best_estimator.coef_))
    
    # Positive coefficients (upregulated)
    pos_coefs = {k: v for k, v in coefs_dict.items() if v > 0}
    pos_sorted = dict(sorted(pos_coefs.items(), key=lambda x: x[1], reverse=True))
    pos_top = list(islice(pos_sorted, n_genes))
    elastic_genes['pos'] = get_symbol_from_id(pos_top)
    pos_symbols_coefs = {g: c for g, c in zip(elastic_genes['pos'], list(pos_sorted.values())[:n_genes])}
    
    # Negative coefficients (downregulated)
    neg_coefs = {k: v for k, v in coefs_dict.items() if v < 0}
    neg_sorted = dict(sorted(neg_coefs.items(), key=lambda x: x[1]))
    neg_top = list(islice(neg_sorted, n_genes))
    elastic_genes['neg'] = get_symbol_from_id(neg_top)
    neg_symbols_coefs = {g: c for g, c in zip(elastic_genes['neg'], list(neg_sorted.values())[:n_genes])}
    
    return elastic_genes, best_estimator, perfs, pos_symbols_coefs, neg_symbols_coefs


def run_svm(y, X_array, X_orig, n_genes, score, xform_list, cv, seed,
            test_size=0.30, gtf_path=None):
    """
    Train Support Vector Regression (SVR) model.
    
    Uses linear kernel SVR with C parameter tuning.
    
    Args:
        y (np.ndarray): Target values
        X_array (np.ndarray): Feature matrix (not used)
        X_orig (pd.DataFrame): Original feature matrix
        n_genes (int): Number of top genes to extract
        score (str): Scoring metric
        xform_list (list): Transformations to apply
        cv (int): Number of CV folds
        seed (int): Random seed
        test_size (float): Test set fraction
        gtf_path (str): Path to GTF file
        
    Returns:
        tuple: (gene_dict, best_estimator, perfs_dict, pos_coefs_dict, neg_coefs_dict)
    """
    from sklearn.svm import SVR
    
    perfs = {}
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_orig, y, test_size=test_size, random_state=seed
    )
    
    # Transform training data
    X_train = full_transform(X_train, xform_list, gtf_path)
    
    # Find best C via grid search
    svr = SVR(kernel='linear')
    param_grid = {'C': [0.1, 1, 10, 100, 1000]}
    grid_search = GridSearchCV(svr, param_grid, cv=cv, scoring=score)
    grid_search.fit(X_train, y_train)
    best_C = grid_search.best_params_['C']
    print(f'  SVR best C: {best_C}')
    
    # Train model with cross-validation
    reg = SVR(kernel="linear", C=best_C, gamma="auto")
    output = cross_validate(
        reg, X_train, y_train,
        cv=LeavePOut(p=2),
        scoring=score,
        return_estimator=True
    )
    
    best_estimator = output['estimator'][np.argmax(output['test_score'])]
    scores = output['test_score']
    avg_score = np.median(list(scores))
    print(f'  SVR avg train score: {avg_score:.4f}')
    perfs['avg_train'] = avg_score
    
    # Transform and evaluate on test data
    X_test = full_transform(X_test, xform_list, gtf_path)
    y_pred = best_estimator.predict(X_test)
    
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'  SVR test RMSE: {rmse:.2f}, R²: {r2:.2f}')
    perfs['test_rmse'] = rmse
    perfs['test_r2'] = r2
    
    # Extract feature importance
    svm_genes = {}
    genes_list = list(X_orig.columns)
    
    # PFI
    svm_genes['pfi'] = get_symbol_from_id(
        permutation_feature_importance(
            best_estimator, X_test, y_test,
            genes=genes_list, scoring=score, n=n_genes, random_state=seed
        )
    )
    
    # RFE
    selector = RFE(best_estimator, n_features_to_select=n_genes, step=0.25)
    selector.fit(X_test, y_test)
    indices = selector.get_support(indices=True)
    svm_genes['rfe'] = get_symbol_from_id([genes_list[i] for i in indices])
    
    # Coefficient-based importance
    coefs = list(best_estimator.coef_)[0]
    features = pd.Series(coefs, index=genes_list)
    
    pos_coefs = {k: v for k, v in features.items() if v > 0}
    pos_sorted = dict(sorted(pos_coefs.items(), key=lambda x: x[1], reverse=True))
    pos_top = list(islice(pos_sorted, n_genes))
    svm_genes['pos'] = get_symbol_from_id(pos_top)
    pos_symbols_coefs = {g: c for g, c in zip(svm_genes['pos'], list(pos_sorted.values())[:n_genes])}
    
    neg_coefs = {k: v for k, v in features.items() if v < 0}
    neg_sorted = dict(sorted(neg_coefs.items(), key=lambda x: x[1]))
    neg_top = list(islice(neg_sorted, n_genes))
    svm_genes['neg'] = get_symbol_from_id(neg_top)
    neg_symbols_coefs = {g: c for g, c in zip(svm_genes['neg'], list(neg_sorted.values())[:n_genes])}
    
    return svm_genes, best_estimator, perfs, pos_symbols_coefs, neg_symbols_coefs


def run_ridge_regression(y, X_array, X_orig, n_genes, score, xform_list, cv, seed,
                         test_size=0.30, gtf_path=None):
    """
    Train Ridge Regression model (L2 regularization).
    
    Args:
        y (np.ndarray): Target values
        X_array (np.ndarray): Feature matrix (not used)
        X_orig (pd.DataFrame): Original feature matrix
        n_genes (int): Number of top genes to extract
        score (str): Scoring metric
        xform_list (list): Transformations to apply
        cv (int): Number of CV folds
        seed (int): Random seed
        test_size (float): Test set fraction
        gtf_path (str): Path to GTF file
        
    Returns:
        tuple: (gene_dict, best_estimator, perfs_dict, pos_coefs_dict, neg_coefs_dict)
    """
    from sklearn.linear_model import Ridge
    
    perfs = {}
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_orig, y, test_size=test_size, random_state=seed
    )
    
    X_train = full_transform(X_train, xform_list, gtf_path)
    
    # Find best alpha
    param_grid = {'alpha': np.logspace(1, 10, 10)}
    ridge = Ridge(random_state=seed)
    grid_search = GridSearchCV(ridge, param_grid, cv=cv, scoring=score)
    grid_search.fit(X_train, y_train)
    best_alpha = grid_search.best_params_['alpha']
    print(f'  Ridge best alpha: {best_alpha}')
    
    reg = Ridge(alpha=best_alpha, random_state=seed)
    output = cross_validate(
        reg, X_train, y_train,
        cv=LeavePOut(p=2),
        scoring=score,
        return_estimator=True
    )
    
    best_estimator = output['estimator'][np.argmax(output['test_score'])]
    avg_score = np.median(output['test_score'])
    print(f'  Ridge avg train score: {avg_score:.4f}')
    perfs['avg_train'] = avg_score
    
    X_test = full_transform(X_test, xform_list, gtf_path)
    y_pred = best_estimator.predict(X_test)
    
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'  Ridge test RMSE: {rmse:.2f}, R²: {r2:.2f}')
    perfs['test_rmse'] = rmse
    perfs['test_r2'] = r2
    
    # Feature importance
    ridge_genes = {}
    genes_list = list(X_orig.columns)
    
    ridge_genes['pfi'] = get_symbol_from_id(
        permutation_feature_importance(
            best_estimator, X_test, y_test,
            genes=genes_list, scoring=score, n=n_genes, random_state=seed
        )
    )
    
    selector = RFE(best_estimator, n_features_to_select=n_genes, step=0.25)
    selector.fit(X_test, y_test)
    indices = selector.get_support(indices=True)
    ridge_genes['rfe'] = get_symbol_from_id([genes_list[i] for i in indices])
    
    coefs_dict = dict(zip(genes_list, best_estimator.coef_))
    
    pos_coefs = {k: v for k, v in coefs_dict.items() if v > 0}
    pos_sorted = dict(sorted(pos_coefs.items(), key=lambda x: x[1], reverse=True))
    pos_top = list(islice(pos_sorted, n_genes))
    ridge_genes['pos'] = get_symbol_from_id(pos_top)
    pos_symbols_coefs = {g: c for g, c in zip(ridge_genes['pos'], list(pos_sorted.values())[:n_genes])}
    
    neg_coefs = {k: v for k, v in coefs_dict.items() if v < 0}
    neg_sorted = dict(sorted(neg_coefs.items(), key=lambda x: x[1]))
    neg_top = list(islice(neg_sorted, n_genes))
    ridge_genes['neg'] = get_symbol_from_id(neg_top)
    neg_symbols_coefs = {g: c for g, c in zip(ridge_genes['neg'], list(neg_sorted.values())[:n_genes])}
    
    return ridge_genes, best_estimator, perfs, pos_symbols_coefs, neg_symbols_coefs


def run_lasso_regression(y, X_array, X_orig, n_genes, score, xform_list, cv, seed,
                         test_size=0.30, gtf_path=None):
    """
    Train Lasso Regression model (L1 regularization).
    
    Lasso performs feature selection by shrinking some coefficients to zero.
    
    Args:
        y (np.ndarray): Target values
        X_array (np.ndarray): Feature matrix (not used)
        X_orig (pd.DataFrame): Original feature matrix
        n_genes (int): Number of top genes to extract
        score (str): Scoring metric
        xform_list (list): Transformations to apply
        cv (int): Number of CV folds
        seed (int): Random seed
        test_size (float): Test set fraction
        gtf_path (str): Path to GTF file
        
    Returns:
        tuple: (gene_dict, best_estimator, perfs_dict, pos_coefs_dict, neg_coefs_dict)
    """
    from sklearn.linear_model import Lasso
    
    perfs = {}
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_orig, y, test_size=test_size, random_state=seed
    )
    
    X_train = full_transform(X_train, xform_list, gtf_path)
    
    param_grid = {'alpha': np.logspace(1, 10, 10)}
    lasso = Lasso(random_state=seed)
    grid_search = GridSearchCV(lasso, param_grid, cv=cv, scoring=score)
    grid_search.fit(X_train, y_train)
    best_alpha = grid_search.best_params_['alpha']
    print(f'  Lasso best alpha: {best_alpha}')
    
    reg = Lasso(alpha=best_alpha, random_state=seed)
    output = cross_validate(
        reg, X_train, y_train,
        cv=LeavePOut(p=2),
        scoring=score,
        return_estimator=True
    )
    
    best_estimator = output['estimator'][np.argmax(output['test_score'])]
    avg_score = np.median(output['test_score'])
    print(f'  Lasso avg train score: {avg_score:.4f}')
    perfs['avg_train'] = avg_score
    
    X_test = full_transform(X_test, xform_list, gtf_path)
    y_pred = best_estimator.predict(X_test)
    
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'  Lasso test RMSE: {rmse:.2f}, R²: {r2:.2f}')
    perfs['test_rmse'] = rmse
    perfs['test_r2'] = r2
    
    lasso_genes = {}
    genes_list = list(X_orig.columns)
    
    lasso_genes['pfi'] = get_symbol_from_id(
        permutation_feature_importance(
            best_estimator, X_test, y_test,
            genes=genes_list, scoring=score, n=n_genes, random_state=seed
        )
    )
    
    selector = RFE(best_estimator, n_features_to_select=n_genes, step=0.25)
    selector.fit(X_test, y_test)
    indices = selector.get_support(indices=True)
    lasso_genes['rfe'] = get_symbol_from_id([genes_list[i] for i in indices])
    
    coefs_dict = dict(zip(genes_list, best_estimator.coef_))
    
    pos_coefs = {k: v for k, v in coefs_dict.items() if v > 0}
    pos_sorted = dict(sorted(pos_coefs.items(), key=lambda x: x[1], reverse=True))
    pos_top = list(islice(pos_sorted, n_genes))
    lasso_genes['pos'] = get_symbol_from_id(pos_top)
    pos_symbols_coefs = {g: c for g, c in zip(lasso_genes['pos'], list(pos_sorted.values())[:n_genes])}
    
    neg_coefs = {k: v for k, v in coefs_dict.items() if v < 0}
    neg_sorted = dict(sorted(neg_coefs.items(), key=lambda x: x[1]))
    neg_top = list(islice(neg_sorted, n_genes))
    lasso_genes['neg'] = get_symbol_from_id(neg_top)
    neg_symbols_coefs = {g: c for g, c in zip(lasso_genes['neg'], list(neg_sorted.values())[:n_genes])}
    
    return lasso_genes, best_estimator, perfs, pos_symbols_coefs, neg_symbols_coefs


def run_linear_regression(y, X_array, X_orig, n_genes, score, xform_list, cv, seed,
                          test_size=0.30, gtf_path=None):
    """
    Train standard Linear Regression model (OLS).
    
    No regularization - standard ordinary least squares.
    
    Args:
        y (np.ndarray): Target values
        X_array (np.ndarray): Feature matrix (not used)
        X_orig (pd.DataFrame): Original feature matrix
        n_genes (int): Number of top genes to extract
        score (str): Scoring metric
        xform_list (list): Transformations to apply
        cv (int): Number of CV folds
        seed (int): Random seed
        test_size (float): Test set fraction
        gtf_path (str): Path to GTF file
        
    Returns:
        tuple: (gene_dict, best_estimator, perfs_dict, pos_coefs_dict, neg_coefs_dict)
    """
    from sklearn.linear_model import LinearRegression
    
    perfs = {}
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_orig, y, test_size=test_size, random_state=seed
    )
    
    X_train = full_transform(X_train, xform_list, gtf_path)
    
    reg = LinearRegression(positive=False)
    output = cross_validate(
        reg, X_train, y_train,
        cv=LeavePOut(p=2),
        scoring=score,
        return_estimator=True
    )
    
    best_estimator = output['estimator'][np.argmax(output['test_score'])]
    avg_score = np.median(output['test_score'])
    print(f'  LinearRegression avg train score: {avg_score:.4f}')
    perfs['avg_train'] = avg_score
    
    X_test = full_transform(X_test, xform_list, gtf_path)
    y_pred = best_estimator.predict(X_test)
    
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'  LinearRegression test RMSE: {rmse:.2f}, R²: {r2:.2f}')
    perfs['test_rmse'] = rmse
    perfs['test_r2'] = r2
    
    linear_genes = {}
    genes_list = list(X_orig.columns)
    
    linear_genes['pfi'] = get_symbol_from_id(
        permutation_feature_importance(
            best_estimator, X_test, y_test,
            genes=genes_list, scoring=score, n=n_genes, random_state=seed
        )
    )
    
    selector = RFE(best_estimator, n_features_to_select=n_genes, step=0.25)
    selector.fit(X_test, y_test)
    indices = selector.get_support(indices=True)
    linear_genes['rfe'] = get_symbol_from_id([genes_list[i] for i in indices])
    
    coefs_dict = dict(zip(genes_list, best_estimator.coef_))
    
    pos_coefs = {k: v for k, v in coefs_dict.items() if v > 0}
    pos_sorted = dict(sorted(pos_coefs.items(), key=lambda x: x[1], reverse=True))
    pos_top = list(islice(pos_sorted, n_genes))
    linear_genes['pos'] = get_symbol_from_id(pos_top)
    pos_symbols_coefs = {g: c for g, c in zip(linear_genes['pos'], list(pos_sorted.values())[:n_genes])}
    
    neg_coefs = {k: v for k, v in coefs_dict.items() if v < 0}
    neg_sorted = dict(sorted(neg_coefs.items(), key=lambda x: x[1]))
    neg_top = list(islice(neg_sorted, n_genes))
    linear_genes['neg'] = get_symbol_from_id(neg_top)
    neg_symbols_coefs = {g: c for g, c in zip(linear_genes['neg'], list(neg_sorted.values())[:n_genes])}
    
    return linear_genes, best_estimator, perfs, pos_symbols_coefs, neg_symbols_coefs
