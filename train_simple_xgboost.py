#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
房價預測競賽 - 簡化版 XGBoost 模型訓練腳本
House Price Prediction Competition - Simplified XGBoost Model Training Script
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import time
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor
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
    # 載入特徵工程後的資料
    print("載入特徵工程後的資料...")
    train = pd.read_csv('house_prices_data/feature_engineering_results/train_engineered.csv')
    test = pd.read_csv('house_prices_data/feature_engineering_results/test_engineered.csv')
    target = pd.read_csv('house_prices_data/feature_engineering_results/target_engineered.csv')
    y_train = target.iloc[:, 0]
    
    print(f"訓練集大小: {train.shape}")
    print(f"測試集大小: {test.shape}")
    print(f"目標變量大小: {y_train.shape}")
    
    # 載入測試集ID
    test_ids = pd.read_csv('house_prices_data/dataset/test.csv')['Id']
    
    # 訓練XGBoost模型
    print("\n=== 訓練XGBoost模型 ===")
    start_time = time.time()
    
    # 定義基本的XGBoost參數
    params = {
        'n_estimators': 250,
        'learning_rate': 0.05,
        'max_depth': 5,
        'min_child_weight': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'reg_alpha': 0.01,
        'reg_lambda': 1.0,
        'objective': 'reg:squarederror',
        'random_state': 42
    }
    
    # 初始化模型
    model = XGBRegressor(**params)
    
    # 評估模型（交叉驗證）
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, train, y_train, cv=kf, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-cv_scores)
    mean_rmse = rmse_scores.mean()
    std_rmse = rmse_scores.std()
    
    print(f"交叉驗證 RMSE: {mean_rmse:.6f} (±{std_rmse:.6f})")
    
    # 訓練最終模型
    print("訓練最終模型...")
    model.fit(train, y_train)
    
    # 計算訓練時間
    training_time = (time.time() - start_time) / 60
    print(f"模型訓練耗時: {training_time:.2f} 分鐘")
    
    # 儲存模型
    model_path = f'{model_dir}/simple_xgboost_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"XGBoost模型已儲存到 {model_path}")
    
    # 分析特徵重要性
    print("\n=== 特徵重要性分析 ===")
    feature_importance = pd.DataFrame({
        'Feature': train.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # 儲存特徵重要性
    feature_importance.to_csv(f'{model_dir}/simple_xgboost_feature_importance.csv', index=False)
    
    # 繪製前20個最重要的特徵
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(20)
    
    sns.barplot(x='Importance', y='Feature', data=top_features)
    plt.title('XGBoost: Top 20 特徵重要性')
    plt.xlabel('重要性分數')
    plt.tight_layout()
    plt.savefig(f'{model_dir}/simple_xgboost_top_features.png', dpi=300)
    plt.close()
    
    # 輸出最重要的10個特徵
    print("\n最重要的10個特徵:")
    for i, (feature, importance) in enumerate(zip(top_features['Feature'].values[:10], 
                                                top_features['Importance'].values[:10])):
        print(f"{i+1}. {feature}: {importance:.6f}")
    
    # 生成預測
    print("\n=== 生成預測 ===")
    # 預測目標變量（假設是對數轉換後的）
    y_pred_log = model.predict(test)
    
    # 分析對數預測值
    print("\n對數預測值統計摘要:")
    log_stats = pd.Series(y_pred_log).describe()
    print(log_stats)
    
    # 檢查對數預測值是否過大
    if log_stats['max'] > 14:
        print(f"警告: 最大對數預測值 {log_stats['max']:.2f} 超過了14，將進行調整")
        
    # 更合理地限制預測值範圍，避免指數爆炸
    y_pred_log = np.clip(y_pred_log, 0, 13.5)  # log(價格) 通常不太可能超過13.5 (≈$725K)
    
    # 轉換回原始尺度
    y_pred = np.expm1(y_pred_log)
    
    # 分析直接預測結果
    print("\n轉換後的預測值範圍:")
    print(f"最小值: {np.min(y_pred):.2f}, 最大值: {np.max(y_pred):.2f}")
    
    # 設置更合理的房價上下限
    # 根據數據集的實際分布，設置更合理的上下限
    y_pred = np.clip(y_pred, 50000, 750000)  # 設置更合理的房價範圍
    
    # 預測結果統計摘要
    pred_stats = pd.Series(y_pred).describe()
    print("\n預測銷售價格統計摘要:")
    print(pred_stats)
    
    # 繪製預測值分佈
    plt.figure(figsize=(10, 6))
    sns.histplot(y_pred, kde=True)
    plt.title('XGBoost預測銷售價格分佈')
    plt.xlabel('預測銷售價格')
    plt.ylabel('頻率')
    plt.tight_layout()
    plt.savefig(f'{model_dir}/simple_xgboost_predicted_distribution.png', dpi=300)
    plt.close()
    
    # 創建提交文件
    print("\n=== 創建提交文件 ===")
    submission = pd.DataFrame({
        'Id': test_ids,
        'SalePrice': y_pred
    })
    
    # 處理可能的異常值
    if submission['SalePrice'].isna().any() or np.isinf(submission['SalePrice']).any():
        print("警告: 發現無效預測值! 正在替換為中位數...")
        median_price = submission['SalePrice'].median()
        submission['SalePrice'] = submission['SalePrice'].fillna(median_price)
        submission['SalePrice'] = submission['SalePrice'].replace([np.inf, -np.inf], median_price)
    
    if (submission['SalePrice'] < 0).any():
        print("警告: 發現負的預測價格! 正在替換為絕對值...")
        submission['SalePrice'] = submission['SalePrice'].abs()
    
    # 儲存提交文件
    submission_path = f'{model_dir}/simple_xgboost_submission.csv'
    submission.to_csv(submission_path, index=False)
    
    print(f"提交文件已儲存到 {submission_path}")
    print("\n房價預測XGBoost模型訓練和預測完成!")

if __name__ == "__main__":
    main()
