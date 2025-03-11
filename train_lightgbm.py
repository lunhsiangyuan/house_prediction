#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
房價預測競賽 - LightGBM模型
House Price Prediction Competition - LightGBM Model
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import time
from sklearn.model_selection import KFold, cross_val_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# 設定中文字體
import matplotlib as mpl
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'STHeiti', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 設定輸出目錄
model_dir = 'house_prices_data/model_results'
os.makedirs(model_dir, exist_ok=True)

# 設定隨機種子
np.random.seed(42)

def main():
    """主程式"""
    # 載入原始數據，而不是特徵工程後的數據
    print("載入原始數據...")
    train_data = pd.read_csv('house_prices_data/dataset/train.csv')
    test_data = pd.read_csv('house_prices_data/dataset/test.csv')
    
    # 讀取特徵工程後的數據（僅用於特徵，不使用其目標變量）
    print("載入特徵工程後的資料...")
    X_train = pd.read_csv('house_prices_data/feature_engineering_results/train_engineered.csv')
    X_test = pd.read_csv('house_prices_data/feature_engineering_results/test_engineered.csv')
    
    # 從原始數據中提取目標變量（SalePrice）
    y_train_original = train_data['SalePrice']
    
    print(f"訓練集大小: {X_train.shape}")
    print(f"測試集大小: {X_test.shape}")
    print(f"目標變量大小: {y_train_original.shape}")
    
    # 目標變量的對數轉換
    print("\n=== 對目標變量進行對數轉換 ===")
    y_train_log = np.log1p(y_train_original)
    
    print(f"原始目標變量範圍: [{y_train_original.min()}, {y_train_original.max()}]")
    print(f"對數轉換後的目標變量範圍: [{y_train_log.min():.2f}, {y_train_log.max():.2f}]")
    
    # 訓練LightGBM模型（使用對數轉換後的目標變量）
    print("\n=== 訓練LightGBM模型 ===")
    start_time = time.time()
    
    # 定義LightGBM參數
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'n_estimators': 500,
        'learning_rate': 0.01,
        'max_depth': 4,
        'num_leaves': 31,
        'min_data_in_leaf': 20,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'random_state': 42
    }
    
    # 初始化模型
    model = lgb.LGBMRegressor(**params)
    
    # 評估模型（交叉驗證）
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train_log, cv=kf, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-cv_scores)
    mean_rmse = rmse_scores.mean()
    std_rmse = rmse_scores.std()
    
    print(f"交叉驗證 RMSE (對數空間): {mean_rmse:.6f} (±{std_rmse:.6f})")
    
    # 訓練最終模型
    print("訓練最終模型...")
    model.fit(X_train, y_train_log)
    
    # 計算訓練時間
    training_time = (time.time() - start_time) / 60
    print(f"模型訓練耗時: {training_time:.2f} 分鐘")
    
    # 儲存模型
    model_path = f'{model_dir}/lightgbm_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"LightGBM模型已儲存到 {model_path}")
    
    # 分析特徵重要性
    print("\n=== 特徵重要性分析 ===")
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # 儲存特徵重要性
    feature_importance.to_csv(f'{model_dir}/lightgbm_feature_importance.csv', index=False)
    
    # 繪製前20個最重要的特徵
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(20)
    
    sns.barplot(x='Importance', y='Feature', data=top_features)
    plt.title('LightGBM: Top 20 特徵重要性')
    plt.xlabel('重要性分數')
    plt.tight_layout()
    plt.savefig(f'{model_dir}/lightgbm_top_features.png', dpi=300)
    
    # 輸出最重要的10個特徵
    print("\n最重要的10個特徵:")
    for i, (feature, importance) in enumerate(zip(top_features['Feature'].values[:10], 
                                                top_features['Importance'].values[:10])):
        print(f"{i+1}. {feature}: {importance:.6f}")
    
    # 生成預測
    print("\n=== 生成預測 ===")
    # 在對數空間預測
    y_pred_log = model.predict(X_test)
    
    # 輸出對數預測值的統計訊息
    print("\n對數預測值統計摘要:")
    log_stats = pd.Series(y_pred_log).describe()
    print(log_stats)
    
    # 將對數預測值轉換回原始空間
    y_pred = np.expm1(y_pred_log)
    
    # 預測結果統計摘要
    pred_stats = pd.Series(y_pred).describe()
    print("\n預測銷售價格統計摘要:")
    print(pred_stats)
    
    # 繪製預測值分佈
    plt.figure(figsize=(10, 6))
    sns.histplot(y_pred, kde=True)
    plt.title('LightGBM預測銷售價格分佈')
    plt.xlabel('預測銷售價格')
    plt.ylabel('頻率')
    plt.tight_layout()
    plt.savefig(f'{model_dir}/lightgbm_predicted_distribution.png', dpi=300)
    
    # 創建提交文件
    print("\n=== 創建提交文件 ===")
    test_ids = test_data['Id']
    submission = pd.DataFrame({
        'Id': test_ids,
        'SalePrice': y_pred
    })
    
    # 儲存提交文件
    submission_path = f'{model_dir}/lightgbm_submission.csv'
    submission.to_csv(submission_path, index=False)
    
    print(f"提交文件已儲存到 {submission_path}")
    print("\n房價預測LightGBM模型訓練和預測完成!")

if __name__ == "__main__":
    main()
