#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
房價預測競賽 - 支持向量回歸 (SVR) 模型
House Price Prediction Competition - Support Vector Regression Model
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import time
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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
    
    # 特徵選擇（由於SVR計算成本較高，只選擇最重要的特徵）
    print("\n=== 特徵選擇 ===")
    # 載入已訓練模型的特徵重要性
    try:
        feature_importance = pd.read_csv(f'{model_dir}/lightgbm_feature_importance.csv')
        print("使用LightGBM模型的特徵重要性進行特徵選擇")
    except FileNotFoundError:
        try:
            feature_importance = pd.read_csv(f'{model_dir}/fixed_xgboost_feature_importance.csv')
            print("使用XGBoost模型的特徵重要性進行特徵選擇")
        except FileNotFoundError:
            print("找不到特徵重要性文件，將使用所有特徵")
            feature_importance = pd.DataFrame({
                'Feature': train.columns,
                'Importance': np.ones(train.shape[1])
            })
    
    # 選擇前30個最重要的特徵
    top_features = feature_importance.sort_values('Importance', ascending=False).head(30)['Feature'].values
    print(f"選擇了前30個最重要的特徵: {', '.join(top_features[:5])}...")
    
    # 篩選特徵
    X_train = train[top_features]
    X_test = test[top_features]
    
    # 目標變量的對數轉換
    print("\n=== 對目標變量進行對數轉換 ===")
    y_train_original = y_train.copy()
    y_train_log = np.log1p(y_train_original)
    
    print(f"原始目標變量範圍: [{y_train_original.min()}, {y_train_original.max()}]")
    print(f"對數轉換後的目標變量範圍: [{y_train_log.min():.2f}, {y_train_log.max():.2f}]")
    
    # 訓練支持向量回歸模型
    print("\n=== 訓練支持向量回歸模型 ===")
    start_time = time.time()
    
    # 創建SVR管道
    # 註: StandardScaler用於特徵縮放，對於SVR很重要
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR(
            kernel='rbf',  # 徑向基函數核
            C=1.0,         # 正則化參數
            gamma='scale', # 核係數
            epsilon=0.1,   # epsilon管道帶寬度
            cache_size=1000 # 快取大小，提高性能
        ))
    ])
    
    # 評估模型（由於SVR計算成本高，只使用3折交叉驗證）
    print("執行交叉驗證...")
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train_log, cv=kf, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-cv_scores)
    mean_rmse = rmse_scores.mean()
    std_rmse = rmse_scores.std()
    
    print(f"交叉驗證 RMSE (對數空間): {mean_rmse:.6f} (±{std_rmse:.6f})")
    
    # 訓練最終模型
    print("訓練最終模型...")
    pipeline.fit(X_train, y_train_log)
    
    # 計算訓練時間
    training_time = (time.time() - start_time) / 60
    print(f"模型訓練耗時: {training_time:.2f} 分鐘")
    
    # 儲存模型
    model_path = f'{model_dir}/svr_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)
    
    print(f"SVR模型已儲存到 {model_path}")
    
    # 生成預測
    print("\n=== 生成預測 ===")
    # 在對數空間預測
    y_pred_log = pipeline.predict(X_test)
    
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
    plt.title('SVR預測銷售價格分佈')
    plt.xlabel('預測銷售價格')
    plt.ylabel('頻率')
    plt.tight_layout()
    plt.savefig(f'{model_dir}/svr_predicted_distribution.png', dpi=300)
    plt.close()
    
    # 創建提交文件
    print("\n=== 創建提交文件 ===")
    submission = pd.DataFrame({
        'Id': test_ids,
        'SalePrice': y_pred
    })
    
    # 儲存提交文件
    submission_path = f'{model_dir}/svr_submission.csv'
    submission.to_csv(submission_path, index=False)
    
    print(f"提交文件已儲存到 {submission_path}")
    print("\n房價預測SVR模型訓練和預測完成!")

if __name__ == "__main__":
    main()
