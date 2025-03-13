#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
房價預測競賽 - 超高級特徵工程：非線性轉換與互動特徵擴展
House Price Prediction Competition - Advanced Nonlinear and Interaction Features
==============================================================================

此腳本實現更複雜的非線性特徵轉換和高階特徵互動：
1. 高階非線性轉換 - 對數、平方根、平方、立方等多種轉換
2. 高階特徵互動 - 三個特徵的交互項
3. 房屋專業領域知識特徵 - 基於專業知識創建的複合特徵
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import skew, norm
import warnings

warnings.filterwarnings('ignore')

# 設定輸出目錄
output_dir = 'feature_engineering_results/advanced_nonlinear'
os.makedirs(output_dir, exist_ok=True)

# 設定隨機種子
np.random.seed(42)

def load_advanced_engineered_data(train_path='feature_engineering_results/advanced/train_advanced_engineered.csv', 
                         test_path='feature_engineering_results/advanced/test_advanced_engineered.csv',
                         target_path='feature_engineering_results/advanced/target_advanced_engineered.csv'):
    """載入高級特徵工程後的資料"""
    print("正在載入高級特徵工程後的資料...")
    
    try:
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        target = pd.read_csv(target_path) if os.path.exists(target_path) else None
        
        print(f"高級特徵工程後訓練集大小: {train.shape}")
        print(f"高級特徵工程後測試集大小: {test.shape}")
        if target is not None:
            print(f"目標變量大小: {target.shape}")
            
        return train, test, target.iloc[:, 0] if target is not None else None
        
    except Exception as e:
        print(f"載入高級特徵工程資料時發生錯誤: {e}")
        return None, None, None

def create_advanced_nonlinear_features(X_train, X_test):
    """創建高級非線性特徵和交互特徵"""
    print("\n正在創建高級非線性特徵和交互特徵...")
    
    # 合併訓練集和測試集處理
    ntrain = X_train.shape[0]
    all_data = pd.concat([X_train, X_test]).reset_index(drop=True)
    
    # 記錄原始特徵數
    initial_features = all_data.shape[1]
    
    # 核心特徵清單
    key_features = ['OverallQual', 'GrLivArea', 'TotalSF', 'GarageArea', 'TotalBath', 
                   'YearBuilt', '1stFlrSF', 'LotArea', 'OverallCond', 'YearRemodAdd']
    
    # 確認這些特徵存在
    present_features = [f for f in key_features if f in all_data.columns]
    
    print(f"找到{len(present_features)}/{len(key_features)}個核心特徵")
    
    # 1. 複雜非線性轉換
    print("創建非線性轉換特徵...")
    for feature in present_features:
        # 對數轉換 (適用於正偏斜特徵)
        all_data[f'{feature}_log'] = np.log1p(all_data[feature].clip(lower=0) + 1e-5)
        
        # 平方根轉換 (中等非線性)
        all_data[f'{feature}_sqrt'] = np.sqrt(all_data[feature].clip(lower=0) + 1e-5)
        
        # 平方轉換 (增強大值影響)
        all_data[f'{feature}_squared'] = np.square(all_data[feature])
        
        # 三次方轉換 (強化非線性)
        all_data[f'{feature}_cubed'] = np.power(all_data[feature], 3)
    
    # 2. 高階互動特徵 (3個特徵交互)
    print("創建高階特徵互動項...")
    for i in range(min(len(present_features), 5)):
        for j in range(i+1, min(len(present_features), 6)):
            for k in range(j+1, min(len(present_features), 7)):
                # 乘積交互
                feat_name = f'{present_features[i]}_{present_features[j]}_{present_features[k]}_inter'
                all_data[feat_name] = all_data[present_features[i]] * all_data[present_features[j]] * all_data[present_features[k]]
    
    # 3. 域知識特徵
    print("創建房屋專業知識特徵...")
    
    # 每平方尺單價
    if 'SalePrice' in all_data.columns:
        if 'TotalSF' in all_data.columns:
            mask = (all_data['TotalSF'] > 0) & ~all_data['SalePrice'].isna()
            all_data.loc[mask, 'PricePerSF'] = all_data.loc[mask, 'SalePrice'] / all_data.loc[mask, 'TotalSF']
    
    # 房齡特徵與交互
    if 'YearBuilt' in all_data.columns:
        all_data['HouseAge'] = 2023 - all_data['YearBuilt']
        all_data['IsNew'] = (all_data['HouseAge'] <= 5).astype(int)
        
        # 房齡與質量交互
        if 'OverallQual' in all_data.columns:
            all_data['AgeQual_Ratio'] = all_data['HouseAge'] / all_data['OverallQual']
            all_data['Age_Quality_Interaction'] = all_data['HouseAge'] * all_data['OverallQual']
    
    # 浴室廚房比
    if 'TotalBath' in all_data.columns and 'KitchenAbvGr' in all_data.columns:
        all_data['Bath_Kitchen_Ratio'] = all_data['TotalBath'] / (all_data['KitchenAbvGr'] + 0.5)
    
    # 面積特徵比例
    if 'LotArea' in all_data.columns and 'GrLivArea' in all_data.columns:
        all_data['LivingArea_LotRatio'] = all_data['GrLivArea'] / all_data['LotArea'].clip(lower=1e-3)
    
    # 總建築面積與車庫比
    if 'TotalSF' in all_data.columns and 'GarageArea' in all_data.columns:
        all_data['Garage_House_Ratio'] = all_data['GarageArea'] / all_data['TotalSF'].clip(lower=1e-3)
    
    # 建築年份與裝修年份差
    if 'YearBuilt' in all_data.columns and 'YearRemodAdd' in all_data.columns:
        all_data['Remod_Age'] = all_data['YearRemodAdd'] - all_data['YearBuilt']
        all_data['IsRemod'] = (all_data['Remod_Age'] > 0).astype(int)
    
    # 處理無限值和NaN
    for col in all_data.columns:
        if all_data[col].dtype in [np.float64, np.int64]:
            all_data[col] = all_data[col].replace([np.inf, -np.inf], np.nan)
            all_data[col] = all_data[col].fillna(all_data[col].median())
    
    # 計算新增特徵數量
    new_features = all_data.shape[1] - initial_features
    print(f"成功創建了{new_features}個新特徵，特徵總數: {all_data.shape[1]}")
    
    # 分割回訓練集和測試集
    X_train_new = all_data[:ntrain]
    X_test_new = all_data[ntrain:]
    
    return X_train_new, X_test_new

def create_advanced_nonlinear_features_pipeline(train, test, y_train=None):
    """完整的高級非線性特徵管道"""
    print("\n====== 執行高級非線性特徵管道 ======")
    
    # 確保索引是唯一的
    train_reset = train.reset_index(drop=True)
    test_reset = test.reset_index(drop=True)
    
    # 確保y_train的索引與train_reset一致
    if y_train is not None and hasattr(y_train, 'reset_index'):
        y_train_reset = y_train.reset_index(drop=True)
    else:
        y_train_reset = y_train
    
    # 創建高級非線性特徵和交互特徵
    train_reset, test_reset = create_advanced_nonlinear_features(train_reset, test_reset)
    
    # 保存高級特徵工程結果
    os.makedirs(output_dir, exist_ok=True)
    train_reset.to_csv(f'{output_dir}/train_nonlinear_engineered.csv', index=False)
    test_reset.to_csv(f'{output_dir}/test_nonlinear_engineered.csv', index=False)
    if y_train_reset is not None:
        pd.Series(y_train_reset).to_csv(f'{output_dir}/target_nonlinear_engineered.csv', index=False)
    
    print(f"\n高級非線性特徵後的訓練集大小: {train_reset.shape}")
    print(f"高級非線性特徵後的測試集大小: {test_reset.shape}")
    print(f"高級非線性特徵結果已保存到 {output_dir} 目錄")
    
    return train_reset, test_reset, y_train_reset

if __name__ == "__main__":
    # 載入高級特徵工程資料
    train, test, y_train = load_advanced_engineered_data()
    
    if train is not None and test is not None:
        # 執行高級非線性特徵管道
        train, test, y_train = create_advanced_nonlinear_features_pipeline(train, test, y_train)
        
        print("\n高級非線性特徵工程完成!")
    else:
        print("無法載入高級特徵工程資料，高級非線性特徵工程失敗!") 