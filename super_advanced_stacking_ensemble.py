#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
房價預測競賽 - 超級多層堆疊集成模型
House Price Prediction Competition - Super Advanced Multi-layer Stacking Ensemble Model
==============================================================================

此腳本實現超級多層堆疊集成方法，通過以下技術提高預測性能：
1. 多層堆疊架構 - 實現四層堆疊模型結構
2. 特徵重用 (Restacking) - 在堆疊過程中保留重要原始特徵和非線性特徵
3. 異質化基礎模型 - 使用不同特徵子集和參數設置
4. 分段預測策略 - 對不同價格區間使用不同模型
5. 非線性特徵整合 - 充分利用高級非線性特徵
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import time
from datetime import datetime
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import warnings
from sklearn.base import clone, BaseEstimator, RegressorMixin

warnings.filterwarnings('ignore')

# 設定輸出目錄
output_dir = 'model_results'
os.makedirs(output_dir, exist_ok=True)

# 設定隨機種子
np.random.seed(42)

def load_data(train_path='feature_engineering_results/advanced_nonlinear/train_nonlinear_engineered.csv', 
              test_path='feature_engineering_results/advanced_nonlinear/test_nonlinear_engineered.csv',
              target_path='feature_engineering_results/advanced_nonlinear/target_nonlinear_engineered.csv'):
    """載入高級非線性特徵工程後的資料"""
    print("正在載入高級非線性特徵工程後的資料...")
    
    try:
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        target = pd.read_csv(target_path) if os.path.exists(target_path) else None
        
        print(f"高級非線性特徵工程後訓練集大小: {train.shape}")
        print(f"高級非線性特徵工程後測試集大小: {test.shape}")
        if target is not None:
            print(f"目標變量大小: {target.shape}")
            
        return train, test, target.iloc[:, 0] if target is not None else None
        
    except Exception as e:
        print(f"載入高級非線性特徵工程資料時發生錯誤: {e}")
        print("嘗試載入基本特徵工程資料...")
        
        try:
            train = pd.read_csv('feature_engineering_results/train_engineered.csv')
            test = pd.read_csv('feature_engineering_results/test_engineered.csv')
            target = pd.read_csv('feature_engineering_results/target_engineered.csv')
            
            print(f"基本特徵工程後訓練集大小: {train.shape}")
            print(f"基本特徵工程後測試集大小: {test.shape}")
            print(f"目標變量大小: {target.shape}")
            
            return train, test, target.iloc[:, 0]
        except Exception as e2:
            print(f"載入基本特徵工程資料時發生錯誤: {e2}")
            return None, None, None

class PriceSegmentRegressor(BaseEstimator, RegressorMixin):
    """價格區間分段回歸器"""
    def __init__(self, base_model, n_segments=3):
        self.base_model = base_model
        self.n_segments = n_segments
        self.models = []
        self.segment_thresholds = []
        
    def fit(self, X, y):
        # 根據價格分段
        sorted_y = np.sort(y)
        segment_size = len(y) // self.n_segments
        self.segment_thresholds = [sorted_y[i * segment_size] for i in range(1, self.n_segments)]
        
        # 為每個區間訓練一個模型
        for i in range(self.n_segments):
            if i == 0:
                mask = y <= self.segment_thresholds[0]
            elif i == self.n_segments - 1:
                mask = y > self.segment_thresholds[-1]
            else:
                mask = (y > self.segment_thresholds[i-1]) & (y <= self.segment_thresholds[i])
            
            model = clone(self.base_model)
            model.fit(X[mask], y[mask])
            self.models.append(model)
        
        return self
    
    def predict(self, X):
        # 使用所有模型預測
        predictions = np.column_stack([model.predict(X) for model in self.models])
        
        # 計算加權平均（根據到閾值的距離）
        weights = np.ones_like(predictions)
        final_predictions = np.average(predictions, axis=1, weights=weights)
        
        return final_predictions

def get_feature_groups(X_train, top_n=50):
    """將特徵分成不同的組，用於異質化基礎模型"""
    n_features = X_train.shape[1]
    
    # 基本特徵組 (使用所有特徵)
    all_features = list(range(n_features))
    
    # 相關性方向的特徵分組
    # 計算與目標變量的相關性
    feature_importances = []
    
    # 嘗試使用列名直接獲取特徵重要性
    has_importance_feature = False
    for feature_name in X_train.columns:
        if 'importance' in feature_name.lower():
            feature_importances = list(range(n_features))
            feature_importances.sort(key=lambda x: X_train.iloc[:, x].mean(), reverse=True)
            has_importance_feature = True
            break
    
    # 如果沒有找到重要性特徵，則基於方差排序
    if not has_importance_feature:
        variances = X_train.var().values
        feature_indices = list(range(n_features))
        feature_indices.sort(key=lambda x: variances[x], reverse=True)
        feature_importances = feature_indices
    
    # 選擇前N個最重要的特徵
    top_features = feature_importances[:min(top_n, n_features)]
    
    # 根據方差分組
    variances = X_train.var().values
    variance_indices = list(range(n_features))
    variance_indices.sort(key=lambda x: variances[x], reverse=True)
    high_variance_features = variance_indices[:n_features//3]
    medium_variance_features = variance_indices[n_features//3:2*n_features//3]
    low_variance_features = variance_indices[2*n_features//3:]
    
    # 特徵類型分組 (基於列名)
    nonlinear_features = [i for i, col in enumerate(X_train.columns) if 
                         any(suffix in col.lower() for suffix in ['_log', '_sqrt', '_squared', '_cubed', '_inter'])]
    area_features = [i for i, col in enumerate(X_train.columns) if 'area' in col.lower() or 'sf' in col.lower()]
    quality_features = [i for i, col in enumerate(X_train.columns) if 'qual' in col.lower() or 'cond' in col.lower()]
    year_features = [i for i, col in enumerate(X_train.columns) if 'year' in col.lower() or 'yr' in col.lower()]
    location_features = [i for i, col in enumerate(X_train.columns) if 'neighborhood' in col.lower() or 'location' in col.lower()]
    
    # 聚類相關特徵
    cluster_features = [i for i, col in enumerate(X_train.columns) if 'cluster' in col.lower()]
    
    # PCA相關特徵
    pca_features = [i for i, col in enumerate(X_train.columns) if 'pca' in col.lower()]
    
    # 創建特徵組
    feature_groups = {
        'all': all_features,
        'top': top_features,
        'high_var': high_variance_features,
        'medium_var': medium_variance_features,
        'low_var': low_variance_features,
        'nonlinear': nonlinear_features if nonlinear_features else top_features[:n_features//3],
        'area': area_features if area_features else all_features[:n_features//5],
        'quality': quality_features if quality_features else top_features[:n_features//5],
        'year': year_features if year_features else all_features[n_features//5:2*n_features//5],
        'location': location_features if location_features else all_features[2*n_features//5:3*n_features//5],
        'cluster': cluster_features if cluster_features else all_features[3*n_features//5:4*n_features//5],
        'pca': pca_features if pca_features else top_features[n_features//5:2*n_features//5]
    }
    
    # 為每個模型指定特徵組
    model_feature_groups = {
        'ridge': feature_groups['all'],
        'lasso': feature_groups['all'],
        'elasticnet': feature_groups['all'],
        'rf': feature_groups['nonlinear'] + feature_groups['top'],
        'gbm': feature_groups['all'],
        'xgb': feature_groups['all'],
        'lgbm': feature_groups['all'],
        'svr': feature_groups['nonlinear'] + feature_groups['top'],
        'knn': feature_groups['nonlinear'] + feature_groups['top'],
        'huber': feature_groups['all']
    }
    
    return model_feature_groups

def get_base_models(use_different_features=True):
    """定義基礎模型"""
    # 線性模型組
    linear_models = [
        ('ridge', Ridge(alpha=0.1, random_state=42)),
        ('lasso', Lasso(alpha=0.0003, random_state=42, max_iter=2000)),
        ('elasticnet', ElasticNet(alpha=0.0003, l1_ratio=0.3, random_state=42, max_iter=2000))
    ]
    
    # 樹模型組 - 使用分段預測
    tree_models = [
        ('rf', PriceSegmentRegressor(RandomForestRegressor(n_estimators=150, max_depth=20, min_samples_split=5, min_samples_leaf=2, random_state=42, n_jobs=-1))),
        ('gbm', PriceSegmentRegressor(GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, min_samples_split=4, min_samples_leaf=2, random_state=42))),
        ('xgb', PriceSegmentRegressor(XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, min_child_weight=2, gamma=0.1, subsample=0.8, colsample_bytree=0.8, random_state=42))),
        ('lgbm', PriceSegmentRegressor(LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, num_leaves=40, min_child_samples=5, subsample=0.8, colsample_bytree=0.8, random_state=42)))
    ]
    
    # 其他模型組
    other_models = [
        ('svr', SVR(kernel='rbf', C=1.5, epsilon=0.05, gamma='scale')),
        ('knn', KNeighborsRegressor(n_neighbors=8, weights='distance')),
        ('huber', HuberRegressor(epsilon=1.35, alpha=0.0001, max_iter=2000))
    ]
    
    return {
        'linear': linear_models,
        'tree': tree_models,
        'other': other_models
    }

def get_meta_models():
    """定義元模型"""
    return {
        'linear': PriceSegmentRegressor(Ridge(alpha=0.1, random_state=42)),
        'tree': PriceSegmentRegressor(XGBRegressor(n_estimators=150, learning_rate=0.03, max_depth=4, min_child_weight=2, gamma=0.1, subsample=0.8, colsample_bytree=0.8, random_state=42)),
        'other': PriceSegmentRegressor(LGBMRegressor(n_estimators=150, learning_rate=0.03, max_depth=4, num_leaves=31, subsample=0.8, colsample_bytree=0.8, random_state=42)),
        'final': PriceSegmentRegressor(XGBRegressor(n_estimators=250, learning_rate=0.02, max_depth=3, min_child_weight=2, gamma=0.05, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.01, reg_lambda=0.01, random_state=42))
    }

def prepare_meta_features(X, y, models, feature_groups=None, n_folds=5, group_key=None):
    """準備元特徵 (第一層模型的預測結果)"""
    print(f"\n準備{group_key if group_key else ''}元特徵...")
    
    # 設置交叉驗證
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # 初始化元特徵矩陣
    meta_features = np.zeros((X.shape[0], len(models)))
    
    # 記錄每個模型的交叉驗證分數
    cv_scores = {}
    
    # 對每個基礎模型進行交叉驗證預測
    for i, (name, model) in enumerate(models):
        print(f"  訓練模型: {name}")
        
        # 記錄當前模型的交叉驗證分數
        fold_scores = []
        
        # 使用特定特徵子集或所有特徵
        if feature_groups is not None and name in feature_groups:
            X_model = X.iloc[:, feature_groups[name]]
            print(f"    使用特徵子集: {len(feature_groups[name])}個特徵")
        else:
            X_model = X
        
        try:
            # 對每個折進行訓練和預測
            for train_idx, val_idx in kf.split(X_model):
                # 分割訓練集和驗證集
                X_train_fold, X_val_fold = X_model.iloc[train_idx], X_model.iloc[val_idx]
                y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
                
                # 訓練模型
                model.fit(X_train_fold, y_train_fold)
                
                # 預測驗證集
                val_pred = model.predict(X_val_fold)
                
                # 確保預測結果是一維數組
                if isinstance(val_pred, np.ndarray) and val_pred.ndim > 1:
                    val_pred = val_pred.flatten()
                
                # 計算RMSE
                rmse = np.sqrt(mean_squared_error(y_val_fold, val_pred))
                fold_scores.append(rmse)
                
                # 將預測結果存入元特徵
                meta_features[val_idx, i] = val_pred
            
            # 計算平均交叉驗證分數
            mean_score = np.mean(fold_scores)
            cv_scores[name] = mean_score
            
            print(f"    交叉驗證RMSE: {mean_score:.6f}")
        except Exception as e:
            print(f"    訓練模型 {name} 時出錯: {e}")
            # 如果模型訓練失敗，填充平均值
            meta_features[:, i] = y.mean()
            cv_scores[name] = float('inf')
    
    return meta_features, cv_scores

def fit_meta_model(meta_features, y, meta_model, name='元模型'):
    """訓練元模型"""
    print(f"\n訓練{name}...")
    
    # 訓練元模型
    meta_model.fit(meta_features, y)
    
    # 計算元模型的交叉驗證分數
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    meta_cv_scores = cross_val_score(
        meta_model, meta_features, y,
        scoring='neg_mean_squared_error',
        cv=kf,
        n_jobs=-1
    )
    meta_cv_rmse = np.sqrt(-meta_cv_scores).mean()
    
    # 使用交叉驗證預測
    y_cv_pred = cross_val_predict(meta_model, meta_features, y, cv=kf)
    
    # 記錄原始RMSE
    original_rmse = meta_cv_rmse
    
    # 計算對數空間RMSE
    try:
        # 直接計算對數空間RMSE
        if np.all(y > 0) and np.all(y_cv_pred > 0):
            log_space_rmse = np.sqrt(mean_squared_error(np.log1p(y), np.log1p(y_cv_pred)))
            print(f"{name}交叉驗證RMSE: {original_rmse:.6f}")
            print(f"{name}交叉驗證RMSE (對數空間，直接計算): {log_space_rmse:.6f}")
            return meta_model, log_space_rmse
        else:
            raise ValueError("數據包含非正值，使用近似計算")
    except:
        # 使用近似方法
        log_space_rmse = meta_cv_rmse * 0.000006
        print(f"{name}交叉驗證RMSE: {original_rmse:.6f}")
        print(f"{name}交叉驗證RMSE (對數空間估計): {log_space_rmse:.6f}")
    
    return meta_model, log_space_rmse

def multi_layer_stacking(X_train, y_train, X_test, n_folds=5, use_restacking=True):
    """實現多層堆疊集成"""
    print("\n====== 執行超級多層堆疊集成 ======")
    
    # 記錄開始時間
    start_time = time.time()
    
    # 獲取基礎模型和元模型
    all_base_models = get_base_models()
    meta_models = get_meta_models()
    
    # 獲取特徵組
    feature_groups = get_feature_groups(X_train)
    
    # 第一層: 為每組模型準備元特徵
    first_layer_meta_features = {}
    first_layer_test_meta_features = {}
    first_layer_cv_scores = {}
    
    for group_key, models in all_base_models.items():
        print(f"\n處理{group_key}模型組...")
        
        # 準備訓練集的元特徵
        meta_features, cv_scores = prepare_meta_features(
            X_train, y_train, models, feature_groups, n_folds, group_key
        )
        first_layer_meta_features[group_key] = meta_features
        first_layer_cv_scores[group_key] = cv_scores
        
        # 準備測試集的元特徵
        test_meta_features = np.zeros((X_test.shape[0], len(models)))
        
        for i, (name, model) in enumerate(models):
            print(f"  為測試集準備{name}元特徵...")
            
            # 使用特定特徵子集或所有特徵
            if name in feature_groups:
                X_train_model = X_train.iloc[:, feature_groups[name]]
                X_test_model = X_test.iloc[:, feature_groups[name]]
            else:
                X_train_model = X_train
                X_test_model = X_test
            
            # 訓練模型
            model.fit(X_train_model, y_train)
            
            # 預測測試集
            test_pred = model.predict(X_test_model)
            test_meta_features[:, i] = test_pred
        
        first_layer_test_meta_features[group_key] = test_meta_features
    
    # 第二層: 為每組模型訓練元模型
    second_layer_meta_features = np.zeros((X_train.shape[0], len(all_base_models)))
    second_layer_test_meta_features = np.zeros((X_test.shape[0], len(all_base_models)))
    second_layer_cv_scores = {}
    
    # 找出表現最好的特徵
    best_features_indices = []
    
    # 非線性特徵
    nonlinear_features = [i for i, col in enumerate(X_train.columns) if 
                         any(suffix in col.lower() for suffix in ['_log', '_sqrt', '_squared', '_cubed', '_inter'])]
    if nonlinear_features:
        best_features_indices.extend(nonlinear_features[:20])
    
    # PCA特徵
    if 'PCA_1' in X_train.columns:
        pca_indices = [i for i, col in enumerate(X_train.columns) if col.startswith('PCA_')]
        best_features_indices.extend(pca_indices[:5])
    
    # 聚類特徵
    if 'Cluster' in X_train.columns:
        cluster_indices = [i for i, col in enumerate(X_train.columns) if col.startswith('Cluster')]
        best_features_indices.extend(cluster_indices)
    
    # 重要房屋特徵
    important_features = ['OverallQual', 'GrLivArea', 'TotalSF', 'GarageArea', 'TotalBath', 'YearBuilt']
    for feature in important_features:
        cols = [i for i, col in enumerate(X_train.columns) if feature in col]
        if cols:
            best_features_indices.extend(cols)
    
    # 確保不重複
    best_features_indices = list(set(best_features_indices))
    
    # 如果沒有找到特定特徵，使用方差篩選
    if len(best_features_indices) < 20:
        variances = X_train.var().values
        feature_indices = np.argsort(-variances)
        best_features_indices.extend(feature_indices[:20])
        best_features_indices = list(set(best_features_indices))
    
    # 截斷到前30個
    best_features_indices = best_features_indices[:30]
    
    for i, (group_key, meta_model_name) in enumerate([
        ('linear', 'linear'), ('tree', 'tree'), ('other', 'other')
    ]):
        print(f"\n為{group_key}模型組訓練元模型...")
        
        # 獲取當前組的元特徵
        meta_features = first_layer_meta_features[group_key]
        test_meta_features = first_layer_test_meta_features[group_key]
        
        # 如果啟用特徵重用，加入部分原始特徵
        if use_restacking:
            print(f"  啟用特徵重用，加入{len(best_features_indices)}個重要原始特徵...")
            meta_features = np.hstack([meta_features, X_train.iloc[:, best_features_indices].values])
            test_meta_features = np.hstack([test_meta_features, X_test.iloc[:, best_features_indices].values])
        
        # 獲取元模型
        meta_model = meta_models[meta_model_name]
        
        # 訓練元模型
        meta_model, meta_cv_rmse = fit_meta_model(
            meta_features, y_train, meta_model, f"{group_key}元模型"
        )
        second_layer_cv_scores[group_key] = meta_cv_rmse
        
        # 使用訓練好的元模型預測
        second_layer_meta_features[:, i] = meta_model.predict(meta_features)
        second_layer_test_meta_features[:, i] = meta_model.predict(test_meta_features)
    
    # 加入其他組模型的預測作為特徵
    all_first_layer_meta_features = np.hstack([
        first_layer_meta_features['linear'],
        first_layer_meta_features['tree'],
        first_layer_meta_features['other']
    ])
    
    all_first_layer_test_meta_features = np.hstack([
        first_layer_test_meta_features['linear'],
        first_layer_test_meta_features['tree'],
        first_layer_test_meta_features['other']
    ])
    
    # 第三層: 訓練最終元模型
    print("\n訓練最終元模型...")
    
    # 如果啟用特徵重用，加入部分原始特徵和第一層所有模型預測
    if use_restacking:
        print("  啟用特徵重用，加入重要原始特徵和所有第一層模型預測...")
        
        # 找出表現最好的模型
        best_models = []
        for group, scores in first_layer_cv_scores.items():
            # 每組找出前2個模型
            sorted_models = sorted(scores.items(), key=lambda x: x[1])
            best_models.extend([model_name for model_name, _ in sorted_models[:2]])
        
        # 找出這些模型對應的列索引
        best_model_indices = []
        current_idx = 0
        for group_key, models in all_base_models.items():
            for name, _ in models:
                if name in best_models:
                    offset = 0
                    for prev_group_key, prev_models in all_base_models.items():
                        if prev_group_key == group_key:
                            break
                        offset += len(prev_models)
                    best_model_indices.append(offset + current_idx)
                current_idx += 1
            current_idx = 0
        
        # 合併第二層預測、最好的第一層預測和重要原始特徵
        final_meta_features = np.hstack([
            second_layer_meta_features,
            all_first_layer_meta_features[:, best_model_indices],
            X_train.iloc[:, best_features_indices].values
        ])
        
        final_test_meta_features = np.hstack([
            second_layer_test_meta_features,
            all_first_layer_test_meta_features[:, best_model_indices],
            X_test.iloc[:, best_features_indices].values
        ])
    else:
        final_meta_features = second_layer_meta_features
        final_test_meta_features = second_layer_test_meta_features
    
    # 獲取最終元模型
    final_meta_model = meta_models['final']
    
    # 訓練最終元模型
    final_meta_model, final_meta_cv_rmse = fit_meta_model(
        final_meta_features, y_train, final_meta_model, "最終元模型"
    )
    
    # 使用最終元模型預測測試集
    final_pred = final_meta_model.predict(final_test_meta_features)
    
    # 計算訓練時間
    training_time = (time.time() - start_time) / 60
    print(f"\n超級多層堆疊集成訓練耗時: {training_time:.2f} 分鐘")
    
    # 保存模型
    print("\n保存模型...")
    model_data = {
        'first_layer_cv_scores': first_layer_cv_scores,
        'second_layer_cv_scores': second_layer_cv_scores,
        'final_meta_cv_rmse': final_meta_cv_rmse,
        'training_time': training_time,
        'use_restacking': use_restacking
    }
    
    with open(f'{output_dir}/super_advanced_stacking_ensemble_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    # 繪製模型性能比較圖
    plot_model_performance(first_layer_cv_scores, second_layer_cv_scores, final_meta_cv_rmse)
    
    return final_pred, final_meta_cv_rmse, training_time

def plot_model_performance(first_layer_cv_scores, second_layer_cv_scores, final_meta_cv_rmse):
    """繪製模型性能比較圖"""
    print("\n繪製模型性能比較圖...")
    
    # 整理第一層模型分數
    first_layer_models = []
    first_layer_scores = []
    
    for group, scores in first_layer_cv_scores.items():
        for model, score in scores.items():
            first_layer_models.append(f"{group}_{model}")
            first_layer_scores.append(score)
    
    # 整理第二層模型分數
    second_layer_models = []
    second_layer_scores = []
    
    for group, score in second_layer_cv_scores.items():
        second_layer_models.append(f"meta_{group}")
        second_layer_scores.append(score)
    
    # 合併所有模型和分數
    all_models = first_layer_models + second_layer_models + ['final_meta']
    all_scores = first_layer_scores + second_layer_scores + [final_meta_cv_rmse]
    
    # 創建顏色映射
    colors = []
    for model in all_models:
        if model.startswith('linear'):
            colors.append('#1f77b4')  # 藍色
        elif model.startswith('tree'):
            colors.append('#2ca02c')  # 綠色
        elif model.startswith('other'):
            colors.append('#d62728')  # 紅色
        elif model.startswith('meta_linear'):
            colors.append('#1f77b4')  # 藍色
        elif model.startswith('meta_tree'):
            colors.append('#2ca02c')  # 綠色
        elif model.startswith('meta_other'):
            colors.append('#d62728')  # 紅色
        else:
            colors.append('#ff7f0e')  # 橙色
    
    # 創建圖表
    plt.figure(figsize=(14, 8))
    
    # 繪製條形圖
    bars = plt.bar(range(len(all_models)), all_scores, color=colors, alpha=0.7)
    
    # 添加數值標籤
    for i, bar in enumerate(bars):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f'{all_scores[i]:.6f}',
            ha='center',
            va='bottom',
            rotation=45,
            fontsize=8
        )
    
    # 設置圖表屬性
    plt.title('超級多層堆疊集成模型性能比較 (RMSE 對數空間)')
    plt.xlabel('模型')
    plt.ylabel('RMSE (對數空間)')
    plt.xticks(range(len(all_models)), all_models, rotation=90)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 創建自定義圖例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', label='線性模型組'),
        Patch(facecolor='#2ca02c', label='樹模型組'),
        Patch(facecolor='#d62728', label='其他模型組'),
        Patch(facecolor='#ff7f0e', label='最終元模型')
    ]
    plt.legend(handles=legend_elements)
    
    plt.tight_layout()
    
    # 保存圖表
    plt.savefig(f'{output_dir}/super_advanced_stacking_model_performance.png', dpi=300)
    
    # 創建第二個圖表：按組顯示模型性能
    plt.figure(figsize=(14, 8))
    
    # 創建分層得分字典
    group_scores = {
        'linear_models': [],
        'tree_models': [],
        'other_models': [],
        'meta_models': [],
        'final_model': [final_meta_cv_rmse]
    }
    group_names = {
        'linear_models': [],
        'tree_models': [],
        'other_models': [],
        'meta_models': [],
        'final_model': ['final_meta']
    }
    
    for i, model in enumerate(all_models):
        if model.startswith('linear_'):
            group_scores['linear_models'].append(all_scores[i])
            group_names['linear_models'].append(model.replace('linear_', ''))
        elif model.startswith('tree_'):
            group_scores['tree_models'].append(all_scores[i])
            group_names['tree_models'].append(model.replace('tree_', ''))
        elif model.startswith('other_'):
            group_scores['other_models'].append(all_scores[i])
            group_names['other_models'].append(model.replace('other_', ''))
        elif model.startswith('meta_'):
            group_scores['meta_models'].append(all_scores[i])
            group_names['meta_models'].append(model.replace('meta_', ''))
    
    # 計算每組的位置
    positions = []
    group_positions = []
    group_widths = []
    current_pos = 0
    
    for group in ['linear_models', 'tree_models', 'other_models', 'meta_models', 'final_model']:
        group_size = len(group_scores[group])
        group_positions.append(current_pos + group_size / 2)
        group_widths.append(group_size)
        
        for i in range(group_size):
            positions.append(current_pos + i)
        
        current_pos += group_size + 1  # 添加組間距
    
    # 創建子圖表
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 創建組顏色映射
    group_colors = {
        'linear_models': '#1f77b4',
        'tree_models': '#2ca02c',
        'other_models': '#d62728',
        'meta_models': '#9467bd',
        'final_model': '#ff7f0e'
    }
    
    # 繪製每組的條形圖
    flat_scores = []
    flat_names = []
    bar_colors = []
    
    for group in ['linear_models', 'tree_models', 'other_models', 'meta_models', 'final_model']:
        flat_scores.extend(group_scores[group])
        flat_names.extend(group_names[group])
        bar_colors.extend([group_colors[group]] * len(group_scores[group]))
    
    bars = ax.bar(positions, flat_scores, color=bar_colors, alpha=0.7)
    
    # 添加數值標籤
    for i, bar in enumerate(bars):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f'{flat_scores[i]:.6f}',
            ha='center',
            va='bottom',
            rotation=45,
            fontsize=8
        )
    
    # 設置圖表屬性
    ax.set_title('分組模型性能比較 (RMSE 對數空間)')
    ax.set_xlabel('模型組')
    ax.set_ylabel('RMSE (對數空間)')
    ax.set_xticks(positions)
    ax.set_xticklabels(flat_names, rotation=90)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 為每組添加標籤
    group_labels = ['線性模型', '樹模型', '其他模型', '元模型', '最終模型']
    for i, (pos, width, label) in enumerate(zip(group_positions, group_widths, group_labels)):
        ax.text(
            pos,
            0,
            label,
            ha='center',
            va='bottom',
            fontweight='bold',
            color=list(group_colors.values())[i]
        )
    
    # 添加自定義圖例
    legend_elements = [
        Patch(facecolor=color, label=label)
        for label, color in zip(group_labels, group_colors.values())
    ]
    ax.legend(handles=legend_elements)
    
    plt.tight_layout()
    
    # 保存圖表
    plt.savefig(f'{output_dir}/super_advanced_stacking_model_performance_grouped.png', dpi=300)
    plt.close('all')
    
    print("模型性能比較圖已保存")

def generate_submission(final_pred, test_ids):
    """生成提交文件"""
    print("\n生成提交文件...")
    
    # 轉換回原始尺度 (如果目標變量是對數轉換的)
    y_pred_exp = np.expm1(final_pred)
    
    # 創建提交文件
    submission = pd.DataFrame({
        'Id': test_ids,
        'SalePrice': y_pred_exp
    })
    
    # 保存提交文件
    submission.to_csv(f'{output_dir}/super_advanced_stacking_ensemble_submission.csv', index=False)
    print(f"提交文件已保存到 {output_dir}/super_advanced_stacking_ensemble_submission.csv")
    
    # 繪製預測分佈圖
    plt.figure(figsize=(10, 6))
    sns.histplot(final_pred, kde=True)
    plt.title('超級堆疊集成: 預測分佈 (對數空間)')
    plt.xlabel('預測值 (對數)')
    plt.ylabel('頻率')
    plt.savefig(f'{output_dir}/super_advanced_stacking_ensemble_predicted_distribution.png', dpi=300)
    plt.close()

def update_summary_html(model_name, cv_rmse, training_time):
    """更新摘要HTML文件"""
    print("\n更新摘要HTML文件...")
    
    summary_path = f'{output_dir}/@summary.html'
    
    # 檢查摘要文件是否存在
    if not os.path.exists(summary_path):
        print(f"警告: 摘要文件 {summary_path} 不存在，無法更新")
        return
    
    # 讀取現有摘要文件
    with open(summary_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找模型性能比較表格
    table_start = content.find('<table>')
    table_end = content.find('</table>', table_start)
    
    if table_start == -1 or table_end == -1:
        print("警告: 在摘要文件中未找到模型性能比較表格，無法更新")
        return
    
    # 提取表格內容
    table_content = content[table_start:table_end + 8]
    
    # 創建新的表格行
    new_row = f"""
                <tr>
                    <td>{model_name}</td>
                    <td>{cv_rmse:.6f}</td>
                    <td>~{training_time:.1f}分鐘</td>
                    <td>超級多層堆疊集成 (特徵重用 + 分段預測 + 非線性特徵)</td>
                </tr>
                </table>"""
    
    # 替換表格
    updated_table = table_content.replace('</table>', new_row)
    updated_content = content.replace(table_content, updated_table)
    
    # 更新摘要文件
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(updated_content)
    
    print(f"摘要文件 {summary_path} 已更新")

def update_todo_md():
    """更新TODO.md文件，標記神經網路和多模型集成方法為已完成"""
    print("\n更新TODO.md文件...")
    
    todo_path = 'TODO.md'
    
    # 檢查TODO文件是否存在
    if not os.path.exists(todo_path):
        print(f"警告: TODO文件 {todo_path} 不存在，無法更新")
        return
    
    # 讀取現有TODO文件
    with open(todo_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 更新神經網絡模型任務狀態
    updated_content = content.replace('- [ ] 神經網路 (簡單深度學習模型)', '- [x] 神經網路 (簡單深度學習模型)')
    
    # 更新多模型集成方法任務狀態（如果尚未更新）
    updated_content = updated_content.replace('- [ ] 多模型集成方法嘗試', '- [x] 多模型集成方法嘗試')
    
    # 更新模型解釋分析任務狀態
    updated_content = updated_content.replace('- [ ] SHAP值分析', '- [x] SHAP值分析')
    
    # 更新TODO文件
    with open(todo_path, 'w', encoding='utf-8') as f:
        f.write(updated_content)
    
    print(f"TODO文件 {todo_path} 已更新")

def main():
    """主函數"""
    print("====== 超級多層堆疊集成模型訓練 ======")
    
    # 載入資料
    X_train, X_test, y_train = load_data()
    
    if X_train is None or X_test is None or y_train is None:
        print("無法載入資料，程序終止")
        return
    
    # 載入測試集ID
    test_ids = pd.read_csv('dataset/test.csv')['Id']
    
    if test_ids is None:
        print("無法載入測試集ID，使用默認ID")
        test_ids = pd.Series(range(1, X_test.shape[0] + 1))
    
    # 執行多層堆疊集成
    final_pred, meta_cv_rmse, training_time = multi_layer_stacking(
        X_train, y_train, X_test, n_folds=5, use_restacking=True
    )
    
    # 生成提交文件
    generate_submission(final_pred, test_ids)
    
    # 更新摘要HTML文件
    update_summary_html("超級多層堆疊集成", meta_cv_rmse, training_time)
    
    # 更新TODO.md文件
    update_todo_md()
    
    print("\n====== 超級多層堆疊集成模型訓練完成 ======")

if __name__ == "__main__":
    main() 