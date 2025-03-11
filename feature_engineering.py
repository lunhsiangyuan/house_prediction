#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
房價預測競賽 - 特徵工程
House Price Prediction Competition - Feature Engineering
==============================================================================

此腳本專注於特徵工程，創建新特徵、選擇重要特徵，為模型訓練做準備。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import skew, norm
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor

# 設定輸出目錄
output_dir = 'house_prices_data/feature_engineering_results'
os.makedirs(output_dir, exist_ok=True)

# 設定隨機種子
np.random.seed(42)

def load_preprocessed_data(train_path='house_prices_data/preprocessing_results/train_preprocessed.csv', 
                          test_path='house_prices_data/preprocessing_results/test_preprocessed.csv',
                          target_path='house_prices_data/preprocessing_results/target_preprocessed.csv'):
    """載入預處理後的資料"""
    print("正在載入預處理後的資料...")
    
    try:
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        target = pd.read_csv(target_path) if os.path.exists(target_path) else None
        
        print(f"預處理後訓練集大小: {train.shape}")
        print(f"預處理後測試集大小: {test.shape}")
        if target is not None:
            print(f"目標變量大小: {target.shape}")
            
        return train, test, target.iloc[:, 0] if target is not None else None
        
    except Exception as e:
        print(f"載入預處理資料時發生錯誤: {e}")
        print("嘗試重新執行預處理...")
        
        from data_preprocessing import load_data, preprocess_pipeline
        
        try:
            # 從原始資料開始
            train_orig, test_orig = load_data()
            train, test, target, train_id, test_id = preprocess_pipeline(train_orig, test_orig)
            
            return train, test, target
        except Exception as e2:
            print(f"重新執行預處理時發生錯誤: {e2}")
            return None, None, None

def create_basic_features(train, test):
    """創建基本衍生特徵"""
    print("\n正在創建基本衍生特徵...")
    
    # 合併訓練集和測試集
    ntrain = train.shape[0]
    all_data = pd.concat([train, test])
    
    # 1. 創建年齡相關特徵
    print("創建年齡相關特徵...")
    all_data['HouseAge'] = all_data['YrSold'] - all_data['YearBuilt']
    all_data['RemodAge'] = all_data['YrSold'] - all_data['YearRemodAdd']
    all_data['GarageAge'] = all_data['YrSold'] - all_data['GarageYrBlt']
    
    # 處理負值或NaN
    all_data['GarageAge'] = all_data['GarageAge'].apply(lambda x: max(0, x) if pd.notnull(x) else 0)
    
    # 2. 創建面積相關特徵
    print("創建面積相關特徵...")
    # 房屋總面積
    all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
    
    # 各種面積佔總面積的比例
    all_data['BasementRatio'] = all_data['TotalBsmtSF'] / all_data['TotalSF']
    all_data['1stFlrRatio'] = all_data['1stFlrSF'] / all_data['TotalSF']
    all_data['2ndFlrRatio'] = all_data['2ndFlrSF'] / all_data['TotalSF']
    
    # 批次替換無限值為0
    for col in ['BasementRatio', '1stFlrRatio', '2ndFlrRatio']:
        all_data[col] = all_data[col].replace([np.inf, -np.inf], 0)
        all_data[col] = all_data[col].fillna(0)
    
    # 3. 創建房間相關特徵
    print("創建房間相關特徵...")
    # 總房間數 (包含地下室)
    all_data['TotalRooms'] = all_data['TotRmsAbvGrd'] + all_data['BsmtFullBath'] + all_data['BsmtHalfBath'] * 0.5
    
    # 總浴室數
    all_data['TotalBath'] = all_data['FullBath'] + all_data['HalfBath'] * 0.5 + all_data['BsmtFullBath'] + all_data['BsmtHalfBath'] * 0.5
    
    # 每室面積
    all_data['AreaPerRoom'] = all_data['TotalSF'] / all_data['TotalRooms']
    all_data['AreaPerRoom'] = all_data['AreaPerRoom'].replace([np.inf, -np.inf], 0)
    all_data['AreaPerRoom'] = all_data['AreaPerRoom'].fillna(0)
    
    # 4. 創建質量相關特徵
    print("創建質量相關特徵...")
    # 屬性總質量分數 (均值)
    quality_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 
                    'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond']
    
    # 確保所有特徵都存在
    quality_cols = [col for col in quality_cols if col in all_data.columns]
    
    # 計算平均質量
    if quality_cols:
        all_data['OverallQualityScore'] = all_data[quality_cols].mean(axis=1)
    
    # 5. 創建位置相關特徵
    print("創建位置相關特徵...")
    # 由於MSSubClass實際上是住房類型的分類編碼，我們可以將其轉換回分類特徵
    all_data['MSSubClass'] = all_data['MSSubClass'].astype(str)
    
    # 重要：將分類變量轉回字符串類型以便後續編碼
    all_data['MSSubClass'] = all_data['MSSubClass'].astype(str)
    
    # 6. 交互特徵
    print("創建交互特徵...")
    # 地理位置 * 質量
    if 'OverallQual' in all_data.columns and 'Neighborhood' in all_data.columns:
        neighborhood_quality = all_data.groupby('Neighborhood')['OverallQual'].mean()
        all_data['NeighborhoodQuality'] = all_data['Neighborhood'].map(neighborhood_quality)
    
    # 年齡 * 質量 (質量對價格的影響可能因房屋年齡而異)
    if 'OverallQual' in all_data.columns and 'HouseAge' in all_data.columns:
        all_data['AgeQuality'] = all_data['OverallQual'] * (1 / (all_data['HouseAge'] + 1))
    
    # 分割回訓練集和測試集
    train_new = all_data[:ntrain]
    test_new = all_data[ntrain:]
    
    print(f"基本特徵創建後訓練集大小: {train_new.shape}")
    print(f"基本特徵創建後測試集大小: {test_new.shape}")
    
    return train_new, test_new

def create_advanced_features(train, test):
    """創建進階衍生特徵"""
    print("\n正在創建進階衍生特徵...")
    
    # 合併訓練集和測試集
    ntrain = train.shape[0]
    all_data = pd.concat([train, test])
    
    # 1. 平方特徵 (捕捉非線性關係)
    print("創建平方特徵...")
    squared_cols = ['OverallQual', 'GrLivArea', 'TotalSF']
    for col in squared_cols:
        if col in all_data.columns:
            all_data[f'{col}Sq'] = all_data[col] ** 2
    
    # 2. 多項式交互特徵 (選擇重要特徵創建交互項)
    print("創建多項式交互特徵...")
    interact_cols = [
        ('OverallQual', 'GrLivArea'), 
        ('OverallQual', 'TotalSF'),
        ('OverallQual', 'GarageArea'),
        ('GrLivArea', 'TotalBath')
    ]
    
    for col1, col2 in interact_cols:
        if col1 in all_data.columns and col2 in all_data.columns:
            all_data[f'{col1}_{col2}_Interact'] = all_data[col1] * all_data[col2]
    
    # 3. 分组平均特征 (社區特征)
    print("創建分组平均特征...")
    if 'Neighborhood' in all_data.columns:
        # 面積在社區中的相對大小
        grp_vars = ['GrLivArea', 'TotalSF', 'LotArea', 'GarageArea']
        for col in grp_vars:
            if col in all_data.columns:
                # 計算每個社區的平均值
                neighborhood_means = all_data.groupby('Neighborhood')[col].mean()
                # 相對於社區平均值的比例
                all_data[f'{col}_Relative'] = all_data[col] / all_data['Neighborhood'].map(neighborhood_means)
                # 處理無限值
                all_data[f'{col}_Relative'] = all_data[f'{col}_Relative'].replace([np.inf, -np.inf], 1)
                all_data[f'{col}_Relative'] = all_data[f'{col}_Relative'].fillna(1)
    
    # 4. 缺失值指示器 (使缺失值信息成為特徵)
    print("創建缺失值指示器...")
    # 這部分可能在預處理階段已處理，但如果有特殊情況可以在此加入
    
    # 5. 特徵聚合 (複合特徵)
    print("創建特徵聚合...")
    # 室外特徵總評分
    exterior_cols = ['ExterQual', 'ExterCond']
    if all(col in all_data.columns for col in exterior_cols):
        all_data['ExteriorScore'] = all_data[exterior_cols].mean(axis=1)
    
    # 地下室特徵總評分
    basement_cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure']
    if all(col in all_data.columns for col in basement_cols):
        all_data['BasementScore'] = all_data[basement_cols].mean(axis=1)
    
    # 車庫特徵總評分
    garage_cols = ['GarageQual', 'GarageCond', 'GarageCars']
    if all(col in all_data.columns for col in garage_cols):
        all_data['GarageScore'] = all_data[garage_cols[:-1]].mean(axis=1) * all_data[garage_cols[-1]]
    
    # 分割回訓練集和測試集
    train_new = all_data[:ntrain]
    test_new = all_data[ntrain:]
    
    print(f"進階特徵創建後訓練集大小: {train_new.shape}")
    print(f"進階特徵創建後測試集大小: {test_new.shape}")
    
    return train_new, test_new

def standardize_features(train, test):
    """標準化數值特徵"""
    print("\n正在標準化數值特徵...")
    
    # 識別數值特徵
    numeric_features = train.select_dtypes(include=['int64', 'float64']).columns
    print(f"識別到{len(numeric_features)}個數值特徵")
    
    # 合併訓練集和測試集進行標準化
    all_data = pd.concat([train, test])
    
    # 處理無限值或超大值
    print("處理無限值和超大值...")
    for col in numeric_features:
        # 替換無限值為NaN，然後填充NaN
        all_data[col] = all_data[col].replace([np.inf, -np.inf], np.nan)
        
        # 檢查是否有NaN值
        if all_data[col].isna().any():
            # 若有NaN值，使用中位數填充
            median_val = all_data[col].median()
            all_data[col] = all_data[col].fillna(median_val)
        
        # 處理超大值（超過均值±5個標準差）
        mean = all_data[col].mean()
        std = all_data[col].std()
        upper_limit = mean + 5 * std
        lower_limit = mean - 5 * std
        
        # 使用截斷函數限制極端值
        all_data[col] = all_data[col].clip(lower=lower_limit, upper=upper_limit)
    
    # 初始化標準化器
    scaler = StandardScaler()
    
    # 標準化數值特徵
    all_data_scaled = all_data.copy()
    all_data_scaled[numeric_features] = scaler.fit_transform(all_data[numeric_features])
    
    print("特徵標準化完成")
    
    # 分割回訓練集和測試集
    train_scaled = all_data_scaled[:train.shape[0]]
    test_scaled = all_data_scaled[train.shape[0]:]
    
    return train_scaled, test_scaled

def select_features(train, test, y_train, method='random_forest', n_features=50):
    """特徵選擇"""
    print(f"\n正在使用{method}方法選擇特徵...")
    
    if method == 'random_forest':
        # 使用隨機森林特徵重要性
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(train, y_train)
        
        # 獲取特徵重要性
        feature_importance = pd.DataFrame({
            'feature': train.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # 儲存特徵重要性
        feature_importance.to_csv(f'{output_dir}/feature_importance.csv', index=False)
        
        # 繪製特徵重要性圖
        top_features = feature_importance.head(min(20, len(feature_importance)))
        plt.figure(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=top_features)
        plt.title('Random Forest: Top 20 Feature Importance')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/feature_importance.png', dpi=300)
        plt.close()
        
        print(f"Top 10 最重要特徵:")
        print(feature_importance.head(10))
        
        # 選擇前n_features個特徵
        selected_features = feature_importance['feature'][:n_features].values
        
    elif method == 'f_regression':
        # 使用F-value統計檢驗
        selector = SelectKBest(f_regression, k=n_features)
        selector.fit(train, y_train)
        
        # 獲取特徵得分
        feature_scores = pd.DataFrame({
            'feature': train.columns,
            'score': selector.scores_
        }).sort_values('score', ascending=False)
        
        # 儲存特徵得分
        feature_scores.to_csv(f'{output_dir}/feature_f_scores.csv', index=False)
        
        # 繪製特徵得分圖
        top_features = feature_scores.head(min(20, len(feature_scores)))
        plt.figure(figsize=(10, 8))
        sns.barplot(x='score', y='feature', data=top_features)
        plt.title('F-Regression: Top 20 Feature Scores')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/feature_f_scores.png', dpi=300)
        plt.close()
        
        print(f"Top 10 F-值特徵:")
        print(feature_scores.head(10))
        
        # 選擇前n_features個特徵
        selected_features = feature_scores['feature'][:n_features].values
    
    else:
        # 預設保留所有特徵
        print("未指定有效的特徵選擇方法，將保留所有特徵")
        selected_features = train.columns.values
    
    # 篩選特徵
    train_selected = train[selected_features]
    test_selected = test[selected_features]
    
    print(f"特徵選擇後訓練集大小: {train_selected.shape}")
    print(f"特徵選擇後測試集大小: {test_selected.shape}")
    
    return train_selected, test_selected

def feature_engineering_pipeline(train, test, y_train=None):
    """完整的特徵工程管道"""
    print("\n====== 執行完整的特徵工程管道 ======")
    
    # 1. 創建基本衍生特徵
    train, test = create_basic_features(train, test)
    
    # 2. 創建進階衍生特徵
    train, test = create_advanced_features(train, test)
    
    # 3. 標準化數值特徵
    train, test = standardize_features(train, test)
    
    # 4. 特徵選擇 (如果提供了目標變量)
    if y_train is not None:
        train, test = select_features(train, test, y_train, method='random_forest', n_features=100)
    
    # 保存特徵工程結果
    os.makedirs(output_dir, exist_ok=True)
    train.to_csv(f'{output_dir}/train_engineered.csv', index=False)
    test.to_csv(f'{output_dir}/test_engineered.csv', index=False)
    if y_train is not None:
        pd.Series(y_train).to_csv(f'{output_dir}/target_engineered.csv', index=False)
    
    print(f"\n特徵工程後的訓練集大小: {train.shape}")
    print(f"特徵工程後的測試集大小: {test.shape}")
    print(f"特徵工程結果已保存到 {output_dir} 目錄")
    
    return train, test, y_train

if __name__ == "__main__":
    # 載入預處理資料
    train, test, y_train = load_preprocessed_data()
    
    if train is not None and test is not None:
        # 執行特徵工程管道
        train, test, y_train = feature_engineering_pipeline(train, test, y_train)
        
        print("\n特徵工程完成!")
    else:
        print("無法載入預處理資料，特徵工程失敗!")
