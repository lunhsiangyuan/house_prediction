#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
房價預測競賽 - 高級特徵工程
House Price Prediction Competition - Advanced Feature Engineering
==============================================================================

此腳本實現更高級的特徵工程技術，用於進一步提高模型性能：
1. 多項式特徵 - 創建更複雜的非線性特徵
2. 主成分分析 (PCA) - 降維並捕捉特徵間的關係
3. 目標編碼 - 使用目標變量對分類特徵進行編碼
4. 特徵交互 - 創建更多特徵交互項
5. 聚類特徵 - 使用聚類算法創建新特徵
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import skew, norm
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from category_encoders import TargetEncoder
import warnings

warnings.filterwarnings('ignore')

# 設定輸出目錄
output_dir = 'feature_engineering_results/advanced'
os.makedirs(output_dir, exist_ok=True)

# 設定隨機種子
np.random.seed(42)

def load_engineered_data(train_path='feature_engineering_results/train_engineered.csv', 
                         test_path='feature_engineering_results/test_engineered.csv',
                         target_path='feature_engineering_results/target_engineered.csv'):
    """載入基本特徵工程後的資料"""
    print("正在載入基本特徵工程後的資料...")
    
    try:
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        target = pd.read_csv(target_path) if os.path.exists(target_path) else None
        
        print(f"基本特徵工程後訓練集大小: {train.shape}")
        print(f"基本特徵工程後測試集大小: {test.shape}")
        if target is not None:
            print(f"目標變量大小: {target.shape}")
            
        return train, test, target.iloc[:, 0] if target is not None else None
        
    except Exception as e:
        print(f"載入特徵工程資料時發生錯誤: {e}")
        return None, None, None

def create_polynomial_features(train, test, degree=2, interaction_only=True, include_bias=False):
    """創建多項式特徵"""
    print("\n正在創建多項式特徵...")
    
    # 合併訓練集和測試集
    ntrain = train.shape[0]
    # 確保索引是唯一的
    train_reset = train.reset_index(drop=True)
    test_reset = test.reset_index(drop=True)
    all_data = pd.concat([train_reset, test_reset])
    
    # 選擇數值特徵
    numeric_features = all_data.select_dtypes(include=['int64', 'float64']).columns
    
    # 選擇最重要的特徵進行多項式轉換 (避免維度爆炸)
    important_features = [
        'OverallQual', 'GrLivArea', 'TotalSF', 'GarageArea', 
        'TotalBath', 'YearBuilt', '1stFlrSF', 'LotArea'
    ]
    
    # 確保所有特徵都存在
    poly_features = [f for f in important_features if f in numeric_features]
    
    if len(poly_features) > 0:
        print(f"為{len(poly_features)}個重要特徵創建多項式特徵...")
        
        # 初始化多項式特徵轉換器
        poly = PolynomialFeatures(
            degree=degree, 
            interaction_only=interaction_only, 
            include_bias=include_bias
        )
        
        # 轉換選定的特徵
        poly_features_array = poly.fit_transform(all_data[poly_features])
        
        # 創建特徵名稱
        poly_feature_names = poly.get_feature_names_out(poly_features)
        
        # 轉換為DataFrame
        poly_df = pd.DataFrame(
            poly_features_array, 
            columns=poly_feature_names,
            index=all_data.index
        )
        
        # 移除原始特徵 (第一列)
        poly_df = poly_df.iloc[:, 1:]
        
        print(f"創建了{poly_df.shape[1]}個多項式特徵")
        
        # 合併回原始數據
        all_data_poly = pd.concat([all_data, poly_df], axis=1)
        
        # 分割回訓練集和測試集
        train_poly = all_data_poly[:ntrain]
        test_poly = all_data_poly[ntrain:]
        
        return train_poly, test_poly
    else:
        print("未找到指定的重要特徵，跳過多項式特徵創建")
        return train_reset, test_reset

def create_pca_features(train, test, n_components=10):
    """使用PCA創建新特徵"""
    print("\n正在使用PCA創建新特徵...")
    
    # 合併訓練集和測試集
    ntrain = train.shape[0]
    # 確保索引是唯一的
    train_reset = train.reset_index(drop=True)
    test_reset = test.reset_index(drop=True)
    all_data = pd.concat([train_reset, test_reset])
    
    # 選擇數值特徵
    numeric_features = all_data.select_dtypes(include=['int64', 'float64']).columns
    
    if len(numeric_features) > 0:
        # 標準化數值特徵
        scaler = StandardScaler()
        all_data_scaled = scaler.fit_transform(all_data[numeric_features])
        
        # 初始化PCA
        n_components = min(n_components, len(numeric_features))
        pca = PCA(n_components=n_components)
        
        # 轉換數據
        pca_result = pca.fit_transform(all_data_scaled)
        
        # 創建PCA特徵DataFrame
        pca_df = pd.DataFrame(
            pca_result,
            columns=[f'PCA_{i+1}' for i in range(n_components)],
            index=all_data.index
        )
        
        # 計算解釋方差比例
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        # 繪製解釋方差圖
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, n_components + 1), explained_variance, alpha=0.7, label='個別解釋方差')
        plt.step(range(1, n_components + 1), cumulative_variance, where='mid', label='累積解釋方差')
        plt.axhline(y=0.95, color='r', linestyle='--', label='95% 解釋方差閾值')
        plt.xlabel('主成分數量')
        plt.ylabel('解釋方差比例')
        plt.title('PCA解釋方差')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{output_dir}/pca_explained_variance.png', dpi=300)
        plt.close()
        
        print(f"創建了{n_components}個PCA特徵，累積解釋方差: {cumulative_variance[-1]:.4f}")
        
        # 合併回原始數據
        all_data_pca = pd.concat([all_data, pca_df], axis=1)
        
        # 分割回訓練集和測試集
        train_pca = all_data_pca[:ntrain]
        test_pca = all_data_pca[ntrain:]
        
        return train_pca, test_pca
    else:
        print("未找到數值特徵，跳過PCA特徵創建")
        return train_reset, test_reset

def create_target_encoded_features(train, test, y_train, cv=5):
    """使用目標編碼處理分類特徵"""
    print("\n正在使用目標編碼處理分類特徵...")
    
    # 確保索引是唯一的
    train_reset = train.reset_index(drop=True)
    test_reset = test.reset_index(drop=True)
    
    # 選擇分類特徵
    categorical_features = train_reset.select_dtypes(include=['object']).columns.tolist()
    
    if len(categorical_features) > 0:
        print(f"對{len(categorical_features)}個分類特徵進行目標編碼...")
        
        # 設置交叉驗證
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        
        # 初始化目標編碼器
        encoder = TargetEncoder(cols=categorical_features)
        
        # 使用交叉驗證進行編碼，避免數據洩漏
        train_encoded = train_reset.copy()
        
        # 確保y_train的索引與train_reset一致
        if hasattr(y_train, 'reset_index'):
            y_train_reset = y_train.reset_index(drop=True)
        else:
            y_train_reset = y_train
        
        # 對訓練集進行交叉驗證編碼
        for train_idx, valid_idx in kf.split(train_reset):
            # 分割訓練集和驗證集
            train_fold, valid_fold = train_reset.iloc[train_idx], train_reset.iloc[valid_idx]
            y_train_fold = y_train_reset.iloc[train_idx] if hasattr(y_train_reset, 'iloc') else y_train_reset[train_idx]
            
            # 在訓練集上擬合編碼器
            encoder.fit(train_fold, y_train_fold)
            
            # 轉換驗證集
            valid_encoded = encoder.transform(valid_fold)
            
            # 更新訓練集中的驗證部分
            train_encoded.iloc[valid_idx, :] = valid_encoded
        
        # 在整個訓練集上擬合編碼器，用於轉換測試集
        encoder.fit(train_reset, y_train_reset)
        test_encoded = encoder.transform(test_reset)
        
        print("目標編碼完成")
        
        return train_encoded, test_encoded
    else:
        print("未找到分類特徵，跳過目標編碼")
        return train_reset, test_reset

def create_cluster_features(train, test, n_clusters=5):
    """使用聚類算法創建新特徵"""
    print("\n正在使用聚類算法創建新特徵...")
    
    # 合併訓練集和測試集
    ntrain = train.shape[0]
    # 確保索引是唯一的
    train_reset = train.reset_index(drop=True)
    test_reset = test.reset_index(drop=True)
    all_data = pd.concat([train_reset, test_reset])
    
    # 選擇數值特徵
    numeric_features = all_data.select_dtypes(include=['int64', 'float64']).columns
    
    if len(numeric_features) > 0:
        # 標準化數值特徵
        scaler = StandardScaler()
        all_data_scaled = scaler.fit_transform(all_data[numeric_features])
        
        # 初始化KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        
        # 擬合模型
        cluster_labels = kmeans.fit_predict(all_data_scaled)
        
        # 添加聚類標籤作為新特徵
        all_data['Cluster'] = cluster_labels
        
        # 為每個聚類計算到聚類中心的距離
        distances = kmeans.transform(all_data_scaled)
        
        # 添加距離特徵
        for i in range(n_clusters):
            all_data[f'Cluster_{i+1}_Distance'] = distances[:, i]
        
        print(f"創建了1個聚類標籤特徵和{n_clusters}個聚類距離特徵")
        
        # 繪製聚類結果 (使用PCA降維到2D)
        pca = PCA(n_components=2)
        all_data_pca = pca.fit_transform(all_data_scaled)
        
        plt.figure(figsize=(10, 8))
        for i in range(n_clusters):
            plt.scatter(
                all_data_pca[cluster_labels == i, 0],
                all_data_pca[cluster_labels == i, 1],
                label=f'Cluster {i+1}',
                alpha=0.7
            )
        
        plt.scatter(
            pca.transform(kmeans.cluster_centers_)[:, 0],
            pca.transform(kmeans.cluster_centers_)[:, 1],
            s=100,
            c='black',
            marker='X',
            label='Centroids'
        )
        
        plt.title('聚類結果 (PCA降維)')
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{output_dir}/cluster_visualization.png', dpi=300)
        plt.close()
        
        # 分割回訓練集和測試集
        train_cluster = all_data[:ntrain]
        test_cluster = all_data[ntrain:]
        
        return train_cluster, test_cluster
    else:
        print("未找到數值特徵，跳過聚類特徵創建")
        return train_reset, test_reset

def create_advanced_interaction_features(train, test):
    """創建更多高級特徵交互項"""
    print("\n正在創建高級特徵交互項...")
    
    # 合併訓練集和測試集
    ntrain = train.shape[0]
    # 確保索引是唯一的
    train_reset = train.reset_index(drop=True)
    test_reset = test.reset_index(drop=True)
    all_data = pd.concat([train_reset, test_reset])
    
    # 定義重要特徵及其交互組合
    interaction_pairs = [
        # 質量與面積的交互
        ('OverallQual', 'GrLivArea'),
        ('OverallQual', 'TotalSF'),
        ('OverallQual', 'LotArea'),
        
        # 年齡與質量的交互
        ('OverallQual', 'YearBuilt'),
        ('OverallQual', 'HouseAge'),
        
        # 位置與質量的交互
        ('NeighborhoodQuality', 'OverallQual'),
        ('NeighborhoodQuality', 'GrLivArea'),
        
        # 面積之間的交互
        ('1stFlrSF', '2ndFlrSF'),
        ('TotalSF', 'LotArea'),
        ('GrLivArea', 'GarageArea'),
        
        # 浴室與面積的交互
        ('TotalBath', 'GrLivArea'),
        
        # 房間與面積的交互
        ('TotRmsAbvGrd', 'GrLivArea'),
        
        # 車庫與質量的交互
        ('GarageCars', 'OverallQual'),
        ('GarageArea', 'OverallQual')
    ]
    
    # 創建交互特徵
    for col1, col2 in interaction_pairs:
        if col1 in all_data.columns and col2 in all_data.columns:
            # 乘積
            all_data[f'{col1}_{col2}_Mult'] = all_data[col1] * all_data[col2]
            
            # 比例
            all_data[f'{col1}_{col2}_Ratio'] = all_data[col1] / all_data[col2].replace(0, 1)
            
            # 和
            all_data[f'{col1}_{col2}_Sum'] = all_data[col1] + all_data[col2]
            
            # 差
            all_data[f'{col1}_{col2}_Diff'] = all_data[col1] - all_data[col2]
    
    # 處理無限值
    for col in all_data.columns:
        if all_data[col].dtype in [np.float64, np.int64]:
            all_data[col] = all_data[col].replace([np.inf, -np.inf], np.nan)
            all_data[col] = all_data[col].fillna(all_data[col].median())
    
    print(f"創建了多種高級交互特徵，特徵總數: {all_data.shape[1]}")
    
    # 分割回訓練集和測試集
    train_interact = all_data[:ntrain]
    test_interact = all_data[ntrain:]
    
    return train_interact, test_interact

def advanced_feature_engineering_pipeline(train, test, y_train=None):
    """完整的高級特徵工程管道"""
    print("\n====== 執行高級特徵工程管道 ======")
    
    # 確保索引是唯一的
    train_reset = train.reset_index(drop=True)
    test_reset = test.reset_index(drop=True)
    
    # 確保y_train的索引與train_reset一致
    if y_train is not None and hasattr(y_train, 'reset_index'):
        y_train_reset = y_train.reset_index(drop=True)
    else:
        y_train_reset = y_train
    
    # 1. 創建多項式特徵
    train_reset, test_reset = create_polynomial_features(train_reset, test_reset, degree=2, interaction_only=True)
    
    # 2. 創建PCA特徵
    train_reset, test_reset = create_pca_features(train_reset, test_reset, n_components=15)
    
    # 3. 創建目標編碼特徵 (如果提供了目標變量)
    if y_train_reset is not None:
        train_reset, test_reset = create_target_encoded_features(train_reset, test_reset, y_train_reset, cv=5)
    
    # 4. 創建聚類特徵
    train_reset, test_reset = create_cluster_features(train_reset, test_reset, n_clusters=5)
    
    # 5. 創建高級交互特徵 - 暫時跳過，避免重複索引問題
    # train_reset, test_reset = create_advanced_interaction_features(train_reset, test_reset)
    print("\n跳過高級交互特徵創建，避免重複索引問題")
    
    # 保存高級特徵工程結果
    os.makedirs(output_dir, exist_ok=True)
    train_reset.to_csv(f'{output_dir}/train_advanced_engineered.csv', index=False)
    test_reset.to_csv(f'{output_dir}/test_advanced_engineered.csv', index=False)
    if y_train_reset is not None:
        pd.Series(y_train_reset).to_csv(f'{output_dir}/target_advanced_engineered.csv', index=False)
    
    print(f"\n高級特徵工程後的訓練集大小: {train_reset.shape}")
    print(f"高級特徵工程後的測試集大小: {test_reset.shape}")
    print(f"高級特徵工程結果已保存到 {output_dir} 目錄")
    
    return train_reset, test_reset, y_train_reset

if __name__ == "__main__":
    # 載入基本特徵工程資料
    train, test, y_train = load_engineered_data()
    
    if train is not None and test is not None:
        # 執行高級特徵工程管道
        train, test, y_train = advanced_feature_engineering_pipeline(train, test, y_train)
        
        print("\n高級特徵工程完成!")
    else:
        print("無法載入基本特徵工程資料，高級特徵工程失敗!") 