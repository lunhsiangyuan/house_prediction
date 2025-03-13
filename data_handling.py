"""
數據處理模塊
用於加載和處理特徵資料
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data():
    """
    載入進階特徵工程後的數據
    
    Returns:
    --------
    X_train : DataFrame
        訓練特徵
    X_test : DataFrame
        測試特徵
    y_train : Series
        訓練目標
    feature_names : list
        特徵名稱列表
    """
    try:
        # 載入訓練集
        X_train = pd.read_csv('feature_engineering_results/advanced_nonlinear/train_nonlinear_engineered.csv')
        y_train = pd.read_csv('dataset/train.csv')['SalePrice']
        
        # 載入測試集
        X_test = pd.read_csv('feature_engineering_results/advanced_nonlinear/test_nonlinear_engineered.csv')
        
        # 獲取特徵名稱
        feature_names = X_train.columns.tolist()
        
        print(f"訓練集大小: {X_train.shape}")
        print(f"測試集大小: {X_test.shape}")
        
        return X_train, X_test, y_train, feature_names
        
    except Exception as e:
        print(f"載入數據時出錯: {str(e)}")
        return None, None, None, None

def load_test_ids():
    """載入測試集ID"""
    try:
        # 載入測試集ID
        test_ids = pd.read_csv('dataset/test.csv')['Id']
        return test_ids
    except Exception as e:
        print(f"載入測試集ID時出錯: {str(e)}")
        return None

def prepare_data_for_training(X_train, X_test, y_train):
    """準備模型訓練數據"""
    print("\n準備數據...")
    
    # 檢查是否已經對數化了y值
    if y_train.skew() > 0.5:  # 如果右偏較大，執行對數轉換
        print("對目標變量進行對數轉換...")
        y_train = np.log1p(y_train)
        log_transformed = True
    else:
        log_transformed = False
    
    # 標準化數值特徵
    print("標準化特徵...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, log_transformed

def segment_data_by_price(X_train, y_train, n_segments=3):
    """
    根據價格將數據分段
    
    Parameters:
    -----------
    X_train : array-like
        訓練特徵
    y_train : Series
        訓練目標
    n_segments : int
        分段數量
        
    Returns:
    --------
    segments : list
        包含(X, y)元組的列表，每個元組對應一個價格段
    """
    print(f"\n將數據分為{n_segments}個價格段...")
    
    # 使用分位數分段
    price_segments = pd.qcut(y_train, n_segments, labels=False)
    
    # 分割數據
    segments = []
    for i in range(n_segments):
        segment_mask = (price_segments == i)
        segment_X = X_train[segment_mask]
        segment_y = y_train[segment_mask]
        segments.append((segment_X, segment_y))
        print(f"  段 {i+1}: {segment_X.shape[0]}個樣本 (價格範圍: {y_train[segment_mask].min():.2f}-{y_train[segment_mask].max():.2f})")
    
    return segments 