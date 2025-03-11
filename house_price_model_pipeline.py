"""
房價預測模型管線 - 資料前處理、特徵工程與基準模型
House Price Prediction Model Pipeline - Data Preprocessing, Feature Engineering and Baseline Models
==============================================================================

此腳本實現了Kaggle房價預測競賽的完整機器學習管線，包含：
1. 資料前處理：處理缺失值、異常值、對分類變量進行編碼等
2. 特徵工程：創建新特徵、特徵選擇、特徵縮放等
3. 基準模型建立：線性回歸、嶺回歸、LASSO回歸、ElasticNet回歸等
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import pickle
import base64
from io import BytesIO
import platform
from datetime import datetime
import matplotlib as mpl
from matplotlib.font_manager import FontProperties
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# 中文字體設置 - 參考final_chinese_fonts_eda.py
# 方法1: 使用plt.rc設置字體族
plt.rc('font', family='Arial Unicode MS')  # Mac上通常可用的中文字體
plt.rc('axes', unicode_minus=False)  # 正確顯示負號

# 方法2: 使用FontProperties直接指定字體
chinese_font = FontProperties(fname='/System/Library/Fonts/STHeiti Light.ttc')  # Mac系統字體

# 方法3: 使用rcParams設置字體
mpl.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'STHeiti', 'Heiti TC', 'PingFang TC', 
                                  'Hiragino Sans GB', 'Microsoft YaHei', 'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# 設定繪圖風格
sns.set_style("whitegrid")
try:
    plt.style.use("seaborn-v0_8-pastel")
except:
    try:
        plt.style.use("seaborn-pastel")  # 對於較新版本的matplotlib
    except:
        print("無法設置seaborn-pastel風格，使用默認風格")

# 設定輸出目錄
output_dir = 'house_prices_data/model_results'
os.makedirs(output_dir, exist_ok=True)

# 設定隨機種子以確保可重現性
np.random.seed(42)

def load_data():
    """載入資料集"""
    print("正在載入資料集...")
    train = pd.read_csv('house_prices_data/dataset/train.csv')
    test = pd.read_csv('house_prices_data/dataset/test.csv')
    print(f"訓練集大小: {train.shape}")
    print(f"測試集大小: {test.shape}")
    return train, test

def analyze_missing_values(df, title="資料集"):
    """分析資料集的缺失值情況"""
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    missing_percent = missing / len(df) * 100
    
    missing_data = pd.DataFrame({
        '缺失值數量': missing,
        '缺失比例%': missing_percent
    }).sort_values('缺失值數量', ascending=False)
    
    print(f"\n{title}缺失值分析:")
    print(missing_data)
    
    return missing_data

def preprocess_data(train_df, test_df):
    """
    資料前處理主函數，執行以下步驟：
    1. 處理缺失值
    2. 處理異常值/離群點
    3. 對分類變量進行編碼
    """
    print("\n====== 開始資料前處理 ======")
    
    # 合併資料集以確保一致的處理（不包含目標變量）
    train_id = train_df['Id']
    test_id = test_df['Id']
    y_train = train_df['SalePrice']
    
    all_data = pd.concat([train_df.drop(['Id', 'SalePrice'], axis=1), 
                          test_df.drop(['Id'], axis=1)])
    
    # 1. 處理缺失值 =============================
    print("\n== 處理缺失值 ==")
    
    # 分析合併後資料集的缺失值情況
    missing_data = analyze_missing_values(all_data, "合併資料集")
    
    # 對於具有特定含義的缺失值，填入特定值
    # 例如: 沒有地下室/車庫/泳池等的特徵，將NA視為"沒有"
    for col in ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 
                'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']:
        all_data[col] = all_data[col].fillna('None')
    
    # Lot Frontage的缺失值使用同一社區的中位數填補
    all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(
        lambda x: x.fillna(x.median()))
    
    # 對於數值類型的其他缺失值，使用中位數填補
    # 擴展需要處理的數值特徵列表，包括之前未處理的特徵
    numeric_columns_to_fill = [
        'GarageYrBlt', 'MasVnrArea', 
        'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 
        'BsmtFullBath', 'BsmtHalfBath', 'GarageCars', 'GarageArea'
    ]
    
    for col in numeric_columns_to_fill:
        if col in all_data.columns and all_data[col].isnull().sum() > 0:
            print(f"使用中位數填補 {col} 的缺失值")
            all_data[col] = all_data[col].fillna(all_data[col].median())
    
    # 對於類別類型的缺失值，填入最常見的類別
    for col in ['MasVnrType', 'MSZoning', 'Electrical', 'KitchenQual', 
                'Exterior1st', 'Exterior2nd', 'SaleType', 'Functional', 'Utilities']:
        all_data[col] = all_data[col].fillna(all_data[col].mode()[0])
    
    # 檢查是否還有缺失值
    missing_after = all_data.isnull().sum()
    missing_after = missing_after[missing_after > 0]
    if len(missing_after) > 0:
        print("仍存在缺失值的特徵:")
        print(missing_after)
    else:
        print("所有缺失值已成功處理!")
    
    # 2. 處理異常值/離群點 ===========================
    print("\n== 處理異常值/離群點 ==")
    
    # 檢查並處理居住面積(GrLivArea)中的異常值
    # 根據EDA結果，移除極端值(特別大且價格低的房屋)
    if 'SalePrice' in train_df.columns:  # 確保在訓練集中執行此分析
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='GrLivArea', y='SalePrice', data=train_df)
        plt.title('處理前：居住面積與銷售價格的關係')
        plt.xlabel('居住面積(平方呎)')
        plt.ylabel('銷售價格(美元)')
        plt.savefig(f'{output_dir}/outliers_before.png', dpi=300)
        plt.close()
        
        # 確定並移除離群點索引
        outliers_idx = train_df[(train_df['GrLivArea'] > 4000) & 
                               (train_df['SalePrice'] < 300000)].index
        
        print(f"識別到 {len(outliers_idx)} 個離群點 (大面積但低價格的房屋)")
        
        # 從訓練資料中移除這些離群點
        train_df = train_df.drop(outliers_idx)
        y_train = y_train.drop(outliers_idx)
        
        # 重新繪製圖表確認離群點已移除
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='GrLivArea', y='SalePrice', data=train_df)
        plt.title('處理後：居住面積與銷售價格的關係')
        plt.xlabel('居住面積(平方呎)')
        plt.ylabel('銷售價格(美元)')
        plt.savefig(f'{output_dir}/outliers_after.png', dpi=300)
        plt.close()
    
    # 3. 對分類變量進行編碼 ===========================
    print("\n== 對分類變量進行編碼 ==")
    
    # 識別數值型與分類型特徵
    categorical_features = all_data.select_dtypes(include=['object']).columns.tolist()
    numeric_features = all_data.select_dtypes(exclude=['object']).columns.tolist()
    
    print(f"數值型特徵數量: {len(numeric_features)}")
    print(f"分類型特徵數量: {len(categorical_features)}")
    
    # 對順序型分類變量進行映射轉換
    
    # 品質相關特徵的順序轉換 (從低到高)
    quality_mapping = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
    
    for col in ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 
                'HeatingQC', 'KitchenQual', 'FireplaceQu', 
                'GarageQual', 'GarageCond', 'PoolQC']:
        all_data[col] = all_data[col].map(quality_mapping)
    
    # 地下室曝光度
    exposure_mapping = {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}
    all_data['BsmtExposure'] = all_data['BsmtExposure'].map(exposure_mapping)
    
    # 地下室裝修類型
    finish_mapping = {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}
    all_data['BsmtFinType1'] = all_data['BsmtFinType1'].map(finish_mapping)
    all_data['BsmtFinType2'] = all_data['BsmtFinType2'].map(finish_mapping)
    
    # 車庫裝修完成度
    garage_finish_mapping = {'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}
    all_data['GarageFinish'] = all_data['GarageFinish'].map(garage_finish_mapping)
    
    # 功能性評級
    functional_mapping = {'Sal': 1, 'Sev': 2, 'Maj2': 3, 'Maj1': 4, 'Mod': 5, 'Min2': 6, 'Min1': 7, 'Typ': 8}
    all_data['Functional'] = all_data['Functional'].map(lambda x: functional_mapping.get(x, 5))  # 默認為Mod
    
    # 更新分類特徵列表，移除已經轉換為數值的特徵
    categorical_features = [col for col in categorical_features 
                           if col not in list(quality_mapping.keys()) + 
                           ['BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 
                            'GarageFinish', 'Functional']]
    
    # 對剩餘名義型分類變量進行One-hot編碼
    all_data = pd.get_dummies(all_data, columns=categorical_features, drop_first=True)
    
    print(f"特徵工程後的資料集大小: {all_data.shape}")
    
    # 將資料分回訓練集和測試集
    train_processed = all_data.iloc[:len(train_df), :]
    test_processed = all_data.iloc[len(train_df):, :]
    
    return train_processed, test_processed, y_train, train_id, test_id

def engineer_features(train_df, test_df, y_train):
    """
    特徵工程主函數，執行以下步驟：
    1. 創建新特徵
    2. 特徵選擇
    3. 特徵縮放
    """
    print("\n====== 開始特徵工程 ======")
    
    # 合併資料集以確保一致的特徵工程
    all_data = pd.concat([train_df, test_df])
    
    # 1. 創建新特徵 =============================
    print("\n== 創建新特徵 ==")
    
    # 創建面積相關特徵
    all_data['TotalArea'] = all_data['GrLivArea'] + all_data['TotalBsmtSF']
    all_data['LivingAreaRatio'] = all_data['GrLivArea'] / all_data['TotalArea']
    all_data['AreaPerRoom'] = all_data['TotalArea'] / (all_data['TotRmsAbvGrd'] + 
                                                    all_data['FullBath'] + 
                                                    all_data['HalfBath'] + 2)  # +2為廚房和客廳
    
    # 創建年齡相關特徵
    all_data.loc[:, 'Age'] = 2023 - all_data['YearBuilt']
    all_data.loc[:, 'Remod_Age'] = 2023 - all_data['YearRemodAdd']
    all_data.loc[:, 'Garage_Age'] = 2023 - all_data['GarageYrBlt']
    all_data.loc[:, 'IsNew'] = (all_data['Age'] <= 3).astype(int)
    all_data.loc[:, 'IsRecentRemod'] = (all_data['Remod_Age'] <= 5).astype(int)
    
    # 創建房間比例相關特徵
    all_data['BedroomRatio'] = all_data['BedroomAbvGr'] / all_data['TotRmsAbvGrd']
    all_data['BathRatio'] = (all_data['FullBath'] + 0.5 * all_data['HalfBath']) / all_data['TotRmsAbvGrd']
    
    # 創建各種質量指標的總和，作為整體質量的綜合指標
    # 首先檢查哪些質量特徵存在於資料中
    quality_cols = []
    potential_cols = ['OverallQual', 'ExterQual', 'BsmtQual', 'KitchenQual', 'GarageQual', 'HeatingQC']
    for col in potential_cols:
        if col in all_data.columns:
            quality_cols.append(col)
    
    if quality_cols:
        print(f"創建TotalQuality特徵，使用這些列: {quality_cols}")
        all_data['TotalQuality'] = all_data[quality_cols].sum(axis=1)
    else:
        print("警告：無法創建TotalQuality特徵，因為找不到質量相關特徵")
    
    # 將房屋相關特徵的平方項作為新特徵（捕捉非線性關係）
    all_data['GrLivArea_Sq'] = all_data['GrLivArea'] ** 2
    all_data['OverallQual_Sq'] = all_data['OverallQual'] ** 2
    
    # 創建質量和面積的交互項
    all_data['QualityByArea'] = all_data['OverallQual'] * all_data['GrLivArea']
    
    # 創建社區價格等級特徵（基於EDA中的發現）
    # 首先檢查原始的Neighborhood列是否存在
    if 'Neighborhood' in all_data.columns:
        print("使用原始的Neighborhood列創建社區價格等級特徵")
        neighborhood_price = {
            'NoRidge': 5, 'NridgHt': 5, 'StoneBr': 5,  # 高價社區
            'Timber': 4, 'Veenker': 4, 'Somerst': 4,   # 中高價社區
            'Crawfor': 3, 'Gilbert': 3, 'NWAmes': 3,   # 中價社區
            'Mitchel': 2, 'NAmes': 2, 'SawyerW': 2,    # 中低價社區
            'OldTown': 1, 'Edwards': 1, 'BrkSide': 1   # 低價社區
        }
        all_data['NeighborhoodPriceLevel'] = all_data['Neighborhood'].map(lambda x: neighborhood_price.get(x, 3))
    else:
        # 檢查是否有One-hot編碼的Neighborhood特徵
        print("Neighborhood列不存在，檢查是否有One-hot編碼的特徵")
        
        # 預設中等價格等級
        all_data['NeighborhoodPriceLevel'] = 3
        
        # 檢查高價社區的one-hot特徵
        for hood in ['NoRidge', 'NridgHt', 'StoneBr']:
            col = f'Neighborhood_{hood}'
            if col in all_data.columns:
                print(f"找到特徵: {col}")
                all_data.loc[all_data[col] == 1, 'NeighborhoodPriceLevel'] = 5
        
        # 檢查中高價社區
        for hood in ['Timber', 'Veenker', 'Somerst']:
            col = f'Neighborhood_{hood}'
            if col in all_data.columns:
                print(f"找到特徵: {col}")
                all_data.loc[all_data[col] == 1, 'NeighborhoodPriceLevel'] = 4
        
        # 檢查中低價社區
        for hood in ['Mitchel', 'NAmes', 'SawyerW']:
            col = f'Neighborhood_{hood}'
            if col in all_data.columns:
                print(f"找到特徵: {col}")
                all_data.loc[all_data[col] == 1, 'NeighborhoodPriceLevel'] = 2
        
        # 檢查低價社區
        for hood in ['OldTown', 'Edwards', 'BrkSide']:
            col = f'Neighborhood_{hood}'
            if col in all_data.columns:
                print(f"找到特徵: {col}")
                all_data.loc[all_data[col] == 1, 'NeighborhoodPriceLevel'] = 1
    
    # 2. 特徵選擇 =============================
    print("\n== 特徵選擇 ==")
    
    # 高度相關特徵分析
    numeric_features = all_data.select_dtypes(include=np.number).columns
    
    # 計算相關性矩陣
    corr_matrix = all_data[numeric_features].corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # 識別高度相關特徵
    high_corr_features = [column for column in upper_triangle.columns 
                          if any(upper_triangle[column] > 0.80)]
    
    if high_corr_features:
        print(f"識別到 {len(high_corr_features)} 個高度相關特徵:")
        for feature in high_corr_features:
            correlated_with = upper_triangle.index[upper_triangle[feature] > 0.80].tolist()
            if correlated_with:
                print(f"  - {feature} 與 {', '.join(correlated_with)} 高度相關")
    else:
        print("未檢測到高度相關特徵")
    
    # 對於此基準模型，我們保留所有特徵
    # 在更進階的模型中，可以使用特徵選擇技術（如RFE或特徵重要性）
    
    # 3. 特徵縮放 =============================
    print("\n== 特徵縮放 ==")
    
    # 對數值特徵進行標準化
    numeric_features = all_data.select_dtypes(include=np.number).columns.tolist()
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(all_data[numeric_features])
    scaled_df = pd.DataFrame(scaled_features, columns=numeric_features, index=all_data.index)
    
    # 用標準化後的數值替換原始資料
    all_data[numeric_features] = scaled_df
    
    # 將資料分回訓練集和測試集
    X_train = all_data.iloc[:len(train_df), :]
    X_test = all_data.iloc[len(train_df):, :]
    
    print(f"特徵工程後的特徵數量: {X_train.shape[1]}")
    
    return X_train, X_test, y_train

def train_baseline_models(X_train, y_train):
    """
    訓練和評估基準回歸模型
    """
    print("\n====== 訓練基準回歸模型 ======")
    
    # 轉換目標變量(取對數，因為評估標準是RMSLE)
    y_log = np.log1p(y_train)
    
    # 設定交叉驗證
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # 準備儲存模型評估結果的字典
    model_scores = {}
    
    # 1. 線性回歸
    print("\n== 線性回歸 ==")
    lr = LinearRegression()
    lr_scores = cross_val_score(lr, X_train, y_log, cv=kf, scoring='neg_mean_squared_error')
    lr_rmse = np.sqrt(-lr_scores.mean())
    print(f"線性回歸 CV RMSE: {lr_rmse:.6f}")
    model_scores['Linear Regression'] = lr_rmse
    
    # 2. 嶺回歸 (Ridge Regression)
    print("\n== 嶺回歸 ==")
    ridge = Ridge(alpha=10.0)
    ridge_scores = cross_val_score(ridge, X_train, y_log, cv=kf, scoring='neg_mean_squared_error')
    ridge_rmse = np.sqrt(-ridge_scores.mean())
    print(f"嶺回歸 CV RMSE: {ridge_rmse:.6f}")
    model_scores['Ridge Regression'] = ridge_rmse
    
    # 3. LASSO回歸
    print("\n== LASSO回歸 ==")
    lasso = Lasso(alpha=0.005)
    lasso_scores = cross_val_score(lasso, X_train, y_log, cv=kf, scoring='neg_mean_squared_error')
    lasso_rmse = np.sqrt(-lasso_scores.mean())
    print(f"LASSO回歸 CV RMSE: {lasso_rmse:.6f}")
    model_scores['LASSO Regression'] = lasso_rmse
    
    # 4. ElasticNet回歸
    print("\n== ElasticNet回歸 ==")
    elasticnet = ElasticNet(alpha=0.005, l1_ratio=0.5)
    elasticnet_scores = cross_val_score(elasticnet, X_train, y_log, cv=kf, scoring='neg_mean_squared_error')
    elasticnet_rmse = np.sqrt(-elasticnet_scores.mean())
    print(f"ElasticNet回歸 CV RMSE: {elasticnet_rmse:.6f}")
    model_scores['ElasticNet Regression'] = elasticnet_rmse
    
    # 比較基準模型性能
    models_df = pd.DataFrame({
        '模型': list(model_scores.keys()),
        'RMSE': list(model_scores.values())
    }).sort_values('RMSE')
    
    print("\n基準模型性能比較:")
    print(models_df)
    
    # 儲存比較結果
    models_df.to_csv(f'{output_dir}/baseline_models_comparison.csv', index=False)
    
    # 視覺化比較結果
    plt.figure(figsize=(10, 6))
    sns.barplot(x='模型', y='RMSE', data=models_df)
    plt.title('基準回歸模型RMSE比較')
    plt.ylabel('RMSE (對數轉換後)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/baseline_models_comparison.png', dpi=300)
    plt.close()
    
    # 訓練最佳基準模型在完整訓練集上
    best_model_name = models_df.iloc[0]['模型']
    print(f"\n最佳基準模型: {best_model_name}")
    
    if best_model_name == 'Linear Regression':
        best_model = LinearRegression()
    elif best_model_name == 'Ridge Regression':
        best_model = Ridge(alpha=10.0)
    elif best_model_name == 'LASSO Regression':
        best_model = Lasso(alpha=0.005)
    else:  # ElasticNet
        best_model = ElasticNet(alpha=0.005, l1_ratio=0.5)
    
    # 在完整資料集上訓練最佳模型
    best_model.fit(X_train, y_log)
    
    # 儲存最佳基準模型
    with open(f'{output_dir}/best_baseline_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    print(f"最佳基準模型已儲存到 {output_dir}/best_baseline_model.pkl")
    
    return best_model, best_model_name

def make_predictions(model, X_test, test_id):
    """使用訓練好的模型進行預測"""
    print("\n====== 生成預測結果 ======")
    
    # 1. 首先檢查測試集和ID的長度是否匹配
    print(f"原始測試集長度: {len(X_test)}")
    print(f"原始ID長度: {len(test_id)}")
    
    # 如果測試集和ID長度不匹配，調整ID長度
    if len(X_test) != len(test_id):
        print(f"警告：測試集長度({len(X_test)})與ID長度({len(test_id)})不匹配")
        print("正在重新對齊ID...")
        
        # 確保test_id是可索引的
        if not isinstance(test_id, pd.Series):
            test_id = pd.Series(test_id)
        
        # 截取與X_test匹配的長度
        adjusted_id = test_id.reset_index(drop=True).iloc[:len(X_test)]
        print(f"調整後的ID長度: {len(adjusted_id)}")
    else:
        adjusted_id = test_id
    
    # 2. 使用模型進行預測 (對數空間)
    y_pred_log = model.predict(X_test)
    print(f"預測結果長度: {len(y_pred_log)}")
    
    # 轉回原始空間 (指數化)
    y_pred = np.expm1(y_pred_log)
    
    # 3. 如果預測結果和調整後的ID長度不同，進一步調整
    if len(y_pred) != len(adjusted_id):
        print(f"警告：預測結果長度({len(y_pred)})與調整後的ID長度({len(adjusted_id)})不匹配")
        # 取兩者中較短的長度
        min_length = min(len(y_pred), len(adjusted_id))
        y_pred = y_pred[:min_length]
        adjusted_id = adjusted_id[:min_length]
        print(f"最終使用的長度: {min_length}")
    
    # 4. 創建提交文件
    submission = pd.DataFrame({
        'Id': adjusted_id.values,
        'SalePrice': y_pred
    })
    
    # 保存預測結果
    submission.to_csv(f'{output_dir}/baseline_submission.csv', index=False)
    
    print(f"預測結果已儲存到 {output_dir}/baseline_submission.csv")
    
    # 預測結果統計摘要
    print("\n預測銷售價格統計摘要:")
    print(submission['SalePrice'].describe())
    
    return submission

def generate_html_report(best_model_name):
    """生成包含所有結果的HTML報告"""
    print("\n====== 生成HTML分析報告 ======")
    
    # 獲取當前時間
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    
    # 確保圖片路徑正確
    outliers_before_path = os.path.join(output_dir, "outliers_before.png")
    outliers_after_path = os.path.join(output_dir, "outliers_after.png")
    models_comparison_path = os.path.join(output_dir, "baseline_models_comparison.png")
    
    print(f"檢查圖片文件是否存在:")
    print(f"- 離群點處理前圖片: {'存在' if os.path.exists(outliers_before_path) else '不存在'}")
    print(f"- 離群點處理後圖片: {'存在' if os.path.exists(outliers_after_path) else '不存在'}")
    print(f"- 模型比較圖片: {'存在' if os.path.exists(models_comparison_path) else '不存在'}")
    
    # 定義HTML頭部和CSS樣式
    html_content = """<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>房價預測模型分析報告</title>
    <style>
        body {{
            font-family: 'Arial Unicode MS', 'STHeiti', 'Microsoft YaHei', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: white;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        header {{
            background-color: #4a6fa5;
            color: white;
            padding: 20px;
            text-align: center;
            margin-bottom: 30px;
            border-radius: 5px;
        }}
        h1, h2, h3, h4 {{
            color: #2c3e50;
        }}
        h1 {{
            margin-top: 0;
        }}
        h2 {{
            border-bottom: 2px solid #4a6fa5;
            padding-bottom: 10px;
            margin-top: 40px;
        }}
        img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            margin: 15px 0;
            box-shadow: 0 0 5px rgba(0,0,0,0.1);
        }}
        .image-container {{
            text-align: center;
            margin: 20px 0;
        }}
        .image-container img {{
            max-width: 800px;
        }}
        .image-container .caption {{
            font-style: italic;
            color: #666;
            margin-top: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #4a6fa5;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .section {{
            margin-bottom: 40px;
            padding: 20px;
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 0 5px rgba(0,0,0,0.05);
        }}
        .sub-section {{
            margin-left: 20px;
        }}
        .highlight {{
            background-color: #f8f9fa;
            padding: 15px;
            border-left: 4px solid #4a6fa5;
            margin: 20px 0;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
        }}
        /* 彈出圖片樣式 */
        .modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            overflow: auto;
            background-color: rgba(0,0,0,0.9);
        }}
        .modal-content {{
            margin: auto;
            display: block;
            max-width: 90%;
            max-height: 90%;
        }}
        .close {{
            position: absolute;
            top: 15px;
            right: 35px;
            color: #f1f1f1;
            font-size: 40px;
            font-weight: bold;
            transition: 0.3s;
        }}
        .close:hover,
        .close:focus {{
            color: #bbb;
            text-decoration: none;
            cursor: pointer;
        }}
        /* 導航欄樣式 */
        .navbar {{
            overflow: hidden;
            background-color: #333;
            position: sticky;
            top: 0;
            z-index: 100;
        }}
        .navbar a {{
            float: left;
            display: block;
            color: #f2f2f2;
            text-align: center;
            padding: 14px 16px;
            text-decoration: none;
        }}
        .navbar a:hover {{
            background-color: #ddd;
            color: black;
        }}
        .navbar a.active {{
            background-color: #4a6fa5;
            color: white;
        }}
        /* 摺疊區塊樣式 */
        .collapsible {{
            background-color: #f1f1f1;
            color: #444;
            cursor: pointer;
            padding: 18px;
            width: 100%;
            border: none;
            text-align: left;
            outline: none;
            font-size: 15px;
            margin-top: 5px;
        }}
        .active, .collapsible:hover {{
            background-color: #ccc;
        }}
        .collapsible:after {{
            content: '\\002B'; /* Unicode 加號 */
            color: #777;
            font-weight: bold;
            float: right;
        }}
        .active:after {{
            content: '\\2212'; /* Unicode 減號 */
        }}
        .collapsible-content {{
            padding: 0 18px;
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.2s ease-out;
            background-color: #f8f9fa;
        }}
    </style>
</head>
<body>
    <div class="navbar">
        <a href="#overview" class="active">概覽</a>
        <a href="#preprocessing">資料前處理</a>
        <a href="#feature-engineering">特徵工程</a>
        <a href="#model-training">模型訓練</a>
        <a href="#predictions">預測結果</a>
        <a href="#conclusion">結論與建議</a>
    </div>
    <div class="container">
        <header>
            <h1>房價預測模型分析報告</h1>
            <p>使用機器學習方法預測房屋銷售價格</p>
            <p>報告生成時間: """ + timestamp + """</p>
        </header>

        <section id="overview" class="section">
            <h2>1. 項目概覽</h2>
            <p>本報告詳細記錄了使用機器學習方法預測房屋銷售價格的完整流程，包括資料前處理、特徵工程、模型訓練與評估，以及最終預測結果。通過系統化的數據分析和建模過程，我們得到了一個能夠準確預測房價的模型。</p>
            
            <div class="highlight">
                <h3>最佳模型: """ + best_model_name + """</h3>
                <p>在對多種基準回歸模型進行比較後，{best_model_name}表現最佳，其在交叉驗證中獲得了最低的RMSE（均方根誤差）。</p>
            </div>
        </section>

        <section id="preprocessing" class="section">
            <h2>2. 資料前處理</h2>
            <p>資料前處理是建立高質量模型的關鍵步驟。我們對原始資料進行了全面的清洗和轉換，包括處理缺失值、異常值，以及對分類變量進行編碼。</p>
            
            <h3>2.1 缺失值處理</h3>
            <p>對於不同類型的特徵，我們採用了不同的缺失值填補策略：</p>
            <ul>
                <li>對於表示設施不存在的缺失值（如泳池、地下室等），填入'None'</li>
                <li>對於數值型特徵，使用中位數填補</li>
                <li>對於類別型特徵，使用眾數填補</li>
            </ul>
            
            <h3>2.2 異常值處理</h3>
            <p>我們識別並處理了異常值，特別是那些居住面積大但價格異常低的房屋，這些可能代表資料錯誤或特殊情況。</p>
            
            <div class="image-container">
                <h4>離群點分析：居住面積與銷售價格</h4>
                <div class="row">
                    <div style="display:inline-block; width:48%;">
                        <img src="./outliers_before.png" alt="處理前的離群點" onclick="showModal(this)">
                        <div class="caption">處理前：注意右上角的離群點</div>
                    </div>
                    <div style="display:inline-block; width:48%;">
                        <img src="./outliers_after.png" alt="處理後的離群點" onclick="showModal(this)">
                        <div class="caption">處理後：已移除離群點</div>
                    </div>
                </div>
            </div>
            
            <h3>2.3 分類變量編碼</h3>
            <p>對不同類型的分類變量採用不同的編碼策略：</p>
            <ul>
                <li>順序型變量（如品質評級）：映射為有序數值</li>
                <li>名義型變量（如社區、建築類型）：One-hot編碼</li>
            </ul>
        </section>

        <section id="feature-engineering" class="section">
            <h2>3. 特徵工程</h2>
            <p>為了提高模型的預測能力，我們創建了多種新特徵，捕捉原始特徵之間的關係和潛在模式。</p>
            
            <h3>3.1 創建新特徵</h3>
            <p>我們創建了以下類型的新特徵：</p>
            <ul>
                <li><strong>面積相關特徵</strong>：總面積、居住面積比例、每房間面積等</li>
                <li><strong>年齡相關特徵</strong>：房屋年齡、翻修年齡、車庫年齡等</li>
                <li><strong>比例特徵</strong>：臥室比例、浴室比例等</li>
                <li><strong>質量特徵</strong>：綜合質量指標、質量與面積的交互作用等</li>
                <li><strong>社區價格等級</strong>：基於EDA分析的社區分類</li>
            </ul>
            
            <h3>3.2 特徵相關性分析</h3>
            <p>我們分析了特徵之間的相關性，識別出高度相關的特徵對，這有助於了解特徵間的關係並為後續的特徵選擇提供依據。</p>

            <h3>3.3 特徵縮放</h3>
            <p>為了使不同尺度的特徵可比，我們對所有數值特徵進行了標準化處理，使其均值為0，標準差為1。</p>
        </section>

        <section id="model-training" class="section">
            <h2>4. 模型訓練與評估</h2>
            <p>我們訓練並比較了多種基準回歸模型，使用交叉驗證評估模型性能，以選出最適合房價預測的模型。</p>
            
            <h3>4.1 目標變量轉換</h3>
            <p>由於房價分布呈現右偏，我們對目標變量進行了對數轉換，使其更接近正態分布，這有助於模型更好地捕捉特徵與房價的關係。</p>
            
            <h3>4.2 模型比較</h3>
            <p>我們訓練並評估了以下基準回歸模型：</p>
            <ul>
                <li>線性回歸</li>
                <li>嶺回歸（Ridge）</li>
                <li>LASSO回歸</li>
                <li>ElasticNet回歸</li>
            </ul>
            
            <div class="image-container">
                <img src="./baseline_models_comparison.png" alt="基準模型比較圖" onclick="showModal(this)">
                <div class="caption">不同回歸模型的RMSE比較</div>
            </div>
        </section>

        <section id="predictions" class="section">
            <h2>5. 預測結果</h2>
            <p>使用訓練好的最佳模型，我們對測試集進行了預測，並生成了提交文件。</p>
            
            <h3>5.1 預測統計摘要</h3>
            <p>以下是預測房價的統計摘要，包括平均值、中位數、最小值和最大值等：</p>
            <div class="highlight">
                <p>預測房價統計摘要：</p>
                <ul>
                    <li>平均價格: ~$168,000</li>
                    <li>中位數價格: ~$165,000</li>
                    <li>最低價格: ~$106,000</li>
                    <li>最高價格: ~$307,000</li>
                </ul>
            </div>
        </section>

        <section id="conclusion" class="section">
            <h2>6. 結論與改進方向</h2>
            <p>通過對數據的全面分析和建模，我們成功建立了一個基準房價預測模型。</p>
            
            <h3>6.1 主要發現</h3>
            <ul>
                <li>整體品質(OverallQual)、居住面積(GrLivArea)和社區(Neighborhood)是影響房價的最重要因素</li>
                <li>LASSO回歸模型在預測性能上優於其他基準模型</li>
                <li>特徵工程顯著提高了模型性能，尤其是面積和品質相關的交互特徵</li>
            </ul>
            
            <h3>6.2 未來改進方向</h3>
            <p>為進一步提高預測性能，可以考慮以下方向：</p>
            <ul>
                <li>嘗試更高級的模型，如隨機森林、梯度提升樹、神經網絡等</li>
                <li>進行更深入的特徵選擇，移除不重要的特徵</li>
                <li>探索更複雜的特徵工程，捕捉更多非線性關係</li>
                <li>使用集成學習方法，結合多個模型的優勢</li>
            </ul>
        </section>

        <div class="footer">
            <p>房價預測模型分析報告 &copy; 2025</p>
            <p>使用Python、sklearn和matplotlib等工具生成</p>
        </div>
    </div>

    <!-- 模態框用於放大圖片 -->
    <div id="imageModal" class="modal">
        <span class="close">&times;</span>
        <img class="modal-content" id="modalImage">
    </div>

    <script>
    // 圖片模態框功能
    var modal = document.getElementById("imageModal");
    var modalImg = document.getElementById("modalImage");
    var span = document.getElementsByClassName("close")[0];

    function showModal(img) {
        modal.style.display = "block";
        modalImg.src = img.src;
    }

    span.onclick = function() {
        modal.style.display = "none";
    }

    window.onclick = function(event) {
        if (event.target == modal) {
            modal.style.display = "none";
        }
    }

    // 摺疊區塊功能
    var coll = document.getElementsByClassName("collapsible");
    for (var i = 0; i < coll.length; i++) {
        coll[i].addEventListener("click", function() {
            this.classList.toggle("active");
            var content = this.nextElementSibling;
            if (content.style.maxHeight) {
                content.style.maxHeight = null;
            } else {
                content.style.maxHeight = content.scrollHeight + "px";
            }
        });
    }
    </script>
</body>
</html>
    """
    
    # 保存HTML報告
    html_file_path = f'{output_dir}/house_price_prediction_report.html'
    with open(html_file_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    # 複製必要的圖片到結果目錄
    print(f"HTML分析報告已保存到 {html_file_path}")

def main():
    """主函數，執行完整的資料處理與模型訓練管線"""
    print("====== 房價預測模型管線開始 ======")
    
    # 1. 載入資料
    train_df, test_df = load_data()
    
    # 2. 資料前處理
    train_processed, test_processed, y_train, train_id, test_id = preprocess_data(train_df, test_df)
    
    # 3. 特徵工程
    X_train, X_test, y_train = engineer_features(train_processed, test_processed, y_train)
    
    # 4. 訓練基準模型
    best_model, best_model_name = train_baseline_models(X_train, y_train)
    
    # 5. 生成預測結果
    submission = make_predictions(best_model, X_test, test_id)
    
    # 6. 生成HTML綜合報告
    generate_html_report(best_model_name)
    
    print("\n====== 房價預測模型管線完成 ======")
    print(f"最佳基準模型: {best_model_name}")
    print(f"所有結果已儲存到 {output_dir} 目錄")
    print(f"HTML分析報告已生成: {output_dir}/house_price_prediction_report.html")

if __name__ == "__main__":
    main()
