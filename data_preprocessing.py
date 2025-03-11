#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
房價預測競賽 - 數據預處理
House Price Prediction Competition - Data Preprocessing
==============================================================================

此腳本專注於數據清洗、缺失值處理和基本變量轉換，為特徵工程做準備。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import skew, norm
from scipy import stats

# 設定輸出目錄
output_dir = 'house_prices_data/preprocessing_results'
os.makedirs(output_dir, exist_ok=True)

# 設定隨機種子
np.random.seed(42)

def load_data(train_path='house_prices_data/dataset/train.csv', 
              test_path='house_prices_data/dataset/test.csv'):
    """載入原始數據"""
    print("正在載入原始數據...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    print(f"訓練集大小: {train.shape}")
    print(f"測試集大小: {test.shape}")
    
    # 檢查目標變量
    if 'SalePrice' in train.columns:
        print(f"目標變量範圍: {train['SalePrice'].min()} - {train['SalePrice'].max()}")
        print(f"目標變量均值: {train['SalePrice'].mean():.2f}")
        print(f"目標變量中位數: {train['SalePrice'].median():.2f}")
    
    return train, test

def analyze_missing_values(train, test):
    """分析並可視化缺失值"""
    print("\n正在分析缺失值...")
    
    # 合併數據集以分析所有缺失值
    all_data = pd.concat([train.drop('SalePrice', axis=1), test])
    
    # 計算每個列的缺失值數量
    missing_data = all_data.isnull().sum().sort_values(ascending=False)
    missing_ratio = (missing_data / len(all_data) * 100).sort_values(ascending=False)
    
    # 創建缺失值數據框
    missing_df = pd.DataFrame({
        '缺失值數量': missing_data, 
        '缺失比例(%)': missing_ratio
    })
    
    # 只保留有缺失值的列
    missing_df = missing_df[missing_df['缺失值數量'] > 0]
    
    print(f"發現{len(missing_df)}個有缺失值的特徵:")
    print(missing_df)
    
    # 只可視化缺失比例>1%的特徵
    if len(missing_df) > 0:
        to_plot = missing_df[missing_df['缺失比例(%)'] > 1]
        
        if len(to_plot) > 0:
            plt.figure(figsize=(10, 6))
            plt.bar(to_plot.index, to_plot['缺失比例(%)'])
            plt.title('缺失值比例超過1%的特徵')
            plt.xlabel('特徵名稱')
            plt.ylabel('缺失比例 (%)')
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/missing_values.png', dpi=300)
            plt.close()
    
    return missing_df

def handle_missing_values(train, test):
    """處理缺失值"""
    print("\n正在處理缺失值...")
    
    # 合併數據集以統一處理缺失值
    ntrain = train.shape[0]
    ntest = test.shape[0]
    y_train = train.SalePrice.values
    
    all_data = pd.concat([train.drop('SalePrice', axis=1), test])
    
    # 處理缺失值 - 對特定列使用適當的填充策略
    # 根據數據描述來填充
    
    # 處理None表示"沒有"的類別特徵
    none_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 
                 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                 'MasVnrType']
    
    for col in none_cols:
        all_data[col] = all_data[col].fillna('None')
    
    # 處理數值型缺失值
    # LotFrontage - 使用同一鄰域的中位數填充
    all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(
        lambda x: x.fillna(x.median()))
    
    # GarageYrBlt - 使用建築年份填充
    all_data['GarageYrBlt'] = all_data['GarageYrBlt'].fillna(all_data['YearBuilt'])
    
    # MasVnrArea - 填充為0（無外牆面積）
    all_data['MasVnrArea'] = all_data['MasVnrArea'].fillna(0)
    
    # 對於基本特徵，填充眾數
    all_data['Functional'] = all_data['Functional'].fillna('Typ')
    all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
    all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
    all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
    all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
    all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
    all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
    
    # 面積相關特徵填充為0
    zero_cols = ['GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']
    for col in zero_cols:
        all_data[col] = all_data[col].fillna(0)
    
    # 檢查是否還有缺失值
    missing_after = all_data.isnull().sum().sum()
    if missing_after > 0:
        print(f"警告: 仍有{missing_after}個缺失值.")
        # 對剩餘的缺失值填充0或眾數
        numeric_cols = all_data.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            if all_data[col].isnull().sum() > 0:
                all_data[col] = all_data[col].fillna(0)
        
        categorical_cols = all_data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if all_data[col].isnull().sum() > 0:
                all_data[col] = all_data[col].fillna(all_data[col].mode()[0])
    
    # 檢查處理後的缺失值
    assert all_data.isnull().sum().sum() == 0, "仍有缺失值未處理"
    print("缺失值處理完成!")
    
    # 分割回訓練集和測試集
    train_processed = all_data[:ntrain]
    test_processed = all_data[ntrain:]
    
    return train_processed, test_processed, y_train, train.Id, test.Id

def transform_skewed_features(train, test):
    """轉換偏態數值特徵"""
    print("\n正在轉換偏態數值特徵...")
    
    # 合併數據集以統一處理
    ntrain = train.shape[0]
    all_data = pd.concat([train, test])
    
    # 獲取數值特徵
    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
    
    # 計算偏度
    skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    skewness = pd.DataFrame({'Skew': skewed_feats})
    
    # 只轉換偏度高的特徵 (絕對值 > 0.75)
    skewed_features = skewness[abs(skewness['Skew']) > 0.75].index
    lam = 0.15  # 設定Box-Cox變換的lambda值
    
    print(f"對{len(skewed_features)}個高偏度特徵進行Box-Cox變換:")
    print(skewness[abs(skewness['Skew']) > 0.75].head())
    
    # 對偏態特徵進行Box-Cox變換
    for feat in skewed_features:
        # 確保所有值都是正的
        all_data[feat] = all_data[feat] + 1  # 加1以確保沒有零值
        all_data[feat] = np.power(all_data[feat], lam)
    
    # 分割回訓練集和測試集
    train_transformed = all_data[:ntrain]
    test_transformed = all_data[ntrain:]
    
    return train_transformed, test_transformed

def encode_categorical_features(train, test):
    """編碼分類特徵"""
    print("\n正在編碼分類特徵...")
    
    # 合併數據集以統一處理
    ntrain = train.shape[0]
    all_data = pd.concat([train, test])
    
    # 獲取所有分類特徵
    categorical_feats = all_data.dtypes[all_data.dtypes == "object"].index
    
    print(f"發現{len(categorical_feats)}個分類特徵:")
    print(list(categorical_feats))
    
    # 整數編碼順序型分類變量（根據數據描述）
    ordinal_mapping = {
        'ExterQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'ExterCond': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'BsmtQual': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'BsmtCond': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'BsmtExposure': {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4},
        'HeatingQC': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'KitchenQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'FireplaceQu': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'GarageQual': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'GarageCond': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'PoolQC': {'None': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
        'Fence': {'None': 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4},
        'LotShape': {'IR3': 1, 'IR2': 2, 'IR1': 3, 'Reg': 4},
        'LandContour': {'Low': 1, 'HLS': 2, 'Bnk': 3, 'Lvl': 4},
        'BsmtFinType1': {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6},
        'BsmtFinType2': {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6},
        'Functional': {'Sal': 1, 'Sev': 2, 'Maj2': 3, 'Maj1': 4, 'Mod': 5, 'Min2': 6, 'Min1': 7, 'Typ': 8},
        'GarageFinish': {'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3},
        'Utilities': {'ELO': 1, 'NoSeWa': 2, 'NoSewr': 3, 'AllPub': 4},
        'LandSlope': {'Sev': 1, 'Mod': 2, 'Gtl': 3},
        'PavedDrive': {'N': 0, 'P': 1, 'Y': 2},
        'Street': {'Grvl': 0, 'Pave': 1},
        'CentralAir': {'N': 0, 'Y': 1},
        'Alley': {'None': 0, 'Grvl': 1, 'Pave': 2},
    }
    
    # 應用序數映射
    for col, mapping in ordinal_mapping.items():
        if col in all_data.columns:
            all_data[col] = all_data[col].map(mapping)
    
    # 對剩餘分類特徵進行獨熱編碼
    one_hot_cols = [col for col in categorical_feats if col not in ordinal_mapping]
    
    if one_hot_cols:
        print(f"對{len(one_hot_cols)}個分類特徵進行獨熱編碼")
        all_data = pd.get_dummies(all_data, columns=one_hot_cols, drop_first=True)
    
    # 分割回訓練集和測試集
    train_encoded = all_data[:ntrain]
    test_encoded = all_data[ntrain:]
    
    return train_encoded, test_encoded

def preprocess_pipeline(train_orig, test_orig):
    """完整的預處理管道"""
    print("\n====== 執行完整的數據預處理管道 ======")
    
    # 1. 分析缺失值
    missing_df = analyze_missing_values(train_orig, test_orig)
    
    # 2. 處理缺失值
    train, test, y_train, train_id, test_id = handle_missing_values(train_orig, test_orig)
    
    # 3. 轉換偏態特徵
    train, test = transform_skewed_features(train, test)
    
    # 4. 編碼分類特徵
    train, test = encode_categorical_features(train, test)
    
    # 保存預處理結果
    os.makedirs(output_dir, exist_ok=True)
    train.to_csv(f'{output_dir}/train_preprocessed.csv', index=False)
    test.to_csv(f'{output_dir}/test_preprocessed.csv', index=False)
    pd.Series(y_train).to_csv(f'{output_dir}/target_preprocessed.csv', index=False)
    
    print(f"\n預處理後的訓練集大小: {train.shape}")
    print(f"預處理後的測試集大小: {test.shape}")
    print(f"預處理結果已保存到 {output_dir} 目錄")
    
    return train, test, y_train, train_id, test_id

if __name__ == "__main__":
    # 載入原始數據
    train_orig, test_orig = load_data()
    
    # 執行預處理管道
    train, test, y_train, train_id, test_id = preprocess_pipeline(train_orig, test_orig)
    
    print("\n數據預處理完成!")
