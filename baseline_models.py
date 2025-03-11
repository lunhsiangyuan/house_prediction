"""
房價預測競賽 - 基準模型模組
House Price Prediction Competition - Baseline Models Module
==============================================================================

此腳本專注於Kaggle房價預測競賽的基準模型部分，包含：
1. 線性回歸基準模型
2. 嶺回歸(Ridge Regression)
3. LASSO回歸
4. ElasticNet回歸
5. 模型比較與評估
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import pickle
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats

# 設定輸出目錄
output_dir = 'house_prices_data/baseline_model_results'
os.makedirs(output_dir, exist_ok=True)

def load_engineered_data(train_path='house_prices_data/feature_engineering_results/train_engineered.csv', 
                         test_path='house_prices_data/feature_engineering_results/test_engineered.csv',
                         target_path='house_prices_data/feature_engineering_results/target_engineered.csv'):
    """載入特徵工程後的資料"""
    print("正在載入特徵工程後的資料...")
    
    try:
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        target = pd.read_csv(target_path) if os.path.exists(target_path) else None
        
        print(f"特徵工程後訓練集大小: {train.shape}")
        print(f"特徵工程後測試集大小: {test.shape}")
        if target is not None:
            print(f"目標變量大小: {target.shape}")
            
        return train, test, target.iloc[:, 0] if target is not None else None
        
    except Exception as e:
        print(f"載入特徵工程資料時發生錯誤: {e}")
        print("嘗試直接載入原始資料並執行基本的預處理和特徵工程...")
        
        # 如果特徵工程檔案不存在，載入預處理資料
        try:
            from feature_engineering import load_preprocessed_data, feature_engineering_pipeline
            
            train_preprocessed, test_preprocessed, target = load_preprocessed_data()
            train, test, target = feature_engineering_pipeline(
                train_preprocessed, test_preprocessed, target)
            
            return train, test, target
            
        except Exception as e2:
            print(f"載入預處理資料時發生錯誤: {e2}")
            print("嘗試從原始資料開始...")
            
            # 如果預處理檔案也不存在，載入原始資料
            from data_preprocessing import load_data, preprocess_pipeline
            from feature_engineering import feature_engineering_pipeline
            
            train_orig, test_orig = load_data()
            train_preprocessed, test_preprocessed, target, _, _ = preprocess_pipeline(train_orig, test_orig)
            train, test, target = feature_engineering_pipeline(
                train_preprocessed, test_preprocessed, target)
            
            return train, test, target

def prepare_target(y_train, log_transform=True):
    """準備目標變量，包括對數轉換（如果需要）"""
    print("\n準備目標變量...")
    
    if log_transform:
        print("對目標變量進行對數轉換 (log1p)")
        y_log = np.log1p(y_train)
        
        # 繪製原始與轉換後的分佈
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        sns.histplot(y_train, kde=True)
        plt.title('原始銷售價格分佈')
        plt.xlabel('銷售價格')
        plt.ylabel('頻率')
        
        plt.subplot(1, 2, 2)
        sns.histplot(y_log, kde=True)
        plt.title('對數轉換後的銷售價格分佈')
        plt.xlabel('log(銷售價格+1)')
        plt.ylabel('頻率')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/target_transformation.png', dpi=300)
        plt.close()
        
        print("對數轉換完成，目標變量更接近正態分佈")
        return y_log
    
    return y_train

def train_linear_regression(X_train, y_train, cv=5):
    """訓練並評估線性回歸模型"""
    print("\n====== 線性回歸 ======")
    
    # 建立模型
    lr = LinearRegression()
    
    # 設定交叉驗證
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    # 執行交叉驗證
    cv_scores = cross_val_score(
        lr, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')
    
    # 計算RMSE (取負值後平方根)
    rmse_scores = np.sqrt(-cv_scores)
    avg_rmse = rmse_scores.mean()
    std_rmse = rmse_scores.std()
    
    print(f"線性回歸 {cv}折交叉驗證 RMSE: {avg_rmse:.6f} (±{std_rmse:.6f})")
    
    # 在完整訓練集上訓練模型
    lr.fit(X_train, y_train)
    
    # 計算R²分數
    train_r2 = lr.score(X_train, y_train)
    print(f"訓練集 R²: {train_r2:.6f}")
    
    # 儲存模型
    with open(f'{output_dir}/linear_regression_model.pkl', 'wb') as f:
        pickle.dump(lr, f)
    
    print(f"線性回歸模型已儲存到 {output_dir}/linear_regression_model.pkl")
    
    return lr, avg_rmse

def train_ridge_regression(X_train, y_train, cv=5, alphas=None):
    """訓練並評估嶺回歸模型，包括超參數調優"""
    print("\n====== 嶺回歸 ======")
    
    if alphas is None:
        alphas = [0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0]
    
    # 設定交叉驗證
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    # 使用網格搜索尋找最佳alpha值
    print("正在使用網格搜索尋找最佳alpha值...")
    
    ridge_grid = GridSearchCV(
        estimator=Ridge(random_state=42),
        param_grid={'alpha': alphas},
        cv=kf,
        scoring='neg_mean_squared_error',
        return_train_score=True
    )
    
    ridge_grid.fit(X_train, y_train)
    
    # 取得最佳參數和分數
    best_alpha = ridge_grid.best_params_['alpha']
    best_score = np.sqrt(-ridge_grid.best_score_)
    
    print(f"最佳alpha值: {best_alpha}")
    print(f"對應的RMSE: {best_score:.6f}")
    
    # 使用最佳參數重新訓練模型
    ridge = Ridge(alpha=best_alpha, random_state=42)
    ridge.fit(X_train, y_train)
    
    # 計算R²分數
    train_r2 = ridge.score(X_train, y_train)
    print(f"訓練集 R²: {train_r2:.6f}")
    
    # 繪製alpha值與RMSE的關係
    plt.figure(figsize=(10, 6))
    alphas_log = np.log10(alphas)
    cv_scores = np.sqrt(-ridge_grid.cv_results_['mean_test_score'])
    
    plt.plot(alphas_log, cv_scores, marker='o', linestyle='-')
    plt.xlabel('log(alpha)')
    plt.ylabel('RMSE (交叉驗證)')
    plt.title('嶺回歸: Alpha參數對RMSE的影響')
    plt.axvline(np.log10(best_alpha), color='r', linestyle='--', 
                label=f'最佳alpha = {best_alpha}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_dir}/ridge_alpha_optimization.png', dpi=300)
    plt.close()
    
    # 儲存模型
    with open(f'{output_dir}/ridge_regression_model.pkl', 'wb') as f:
        pickle.dump(ridge, f)
    
    print(f"嶺回歸模型已儲存到 {output_dir}/ridge_regression_model.pkl")
    
    return ridge, best_score, best_alpha

def train_lasso_regression(X_train, y_train, cv=5, alphas=None):
    """訓練並評估LASSO回歸模型，包括超參數調優"""
    print("\n====== LASSO回歸 ======")
    
    if alphas is None:
        alphas = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
    
    # 設定交叉驗證
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    # 使用網格搜索尋找最佳alpha值
    print("正在使用網格搜索尋找最佳alpha值...")
    
    lasso_grid = GridSearchCV(
        estimator=Lasso(random_state=42, max_iter=10000),
        param_grid={'alpha': alphas},
        cv=kf,
        scoring='neg_mean_squared_error',
        return_train_score=True
    )
    
    lasso_grid.fit(X_train, y_train)
    
    # 取得最佳參數和分數
    best_alpha = lasso_grid.best_params_['alpha']
    best_score = np.sqrt(-lasso_grid.best_score_)
    
    print(f"最佳alpha值: {best_alpha}")
    print(f"對應的RMSE: {best_score:.6f}")
    
    # 使用最佳參數重新訓練模型
    lasso = Lasso(alpha=best_alpha, random_state=42, max_iter=10000)
    lasso.fit(X_train, y_train)
    
    # 計算R²分數
    train_r2 = lasso.score(X_train, y_train)
    print(f"訓練集 R²: {train_r2:.6f}")
    
    # 特徵重要性分析 (非零係數)
    coef = pd.Series(lasso.coef_, index=X_train.columns)
    non_zero_coef = coef[coef != 0].sort_values(ascending=False)
    
    print(f"LASSO選擇了 {len(non_zero_coef)}/{len(coef)} 個特徵")
    
    # 繪製非零係數前20個和後20個
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 1, 1)
    top_coef = non_zero_coef.head(20)
    sns.barplot(x=top_coef.values, y=top_coef.index)
    plt.title('LASSO: 前20個正係數特徵')
    plt.xlabel('係數值')
    
    plt.subplot(2, 1, 2)
    bottom_coef = non_zero_coef.tail(20)
    sns.barplot(x=bottom_coef.values, y=bottom_coef.index)
    plt.title('LASSO: 前20個負係數特徵')
    plt.xlabel('係數值')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/lasso_feature_importance.png', dpi=300)
    plt.close()
    
    # 儲存非零係數
    non_zero_coef.to_csv(f'{output_dir}/lasso_non_zero_coefficients.csv')
    
    # 儲存模型
    with open(f'{output_dir}/lasso_regression_model.pkl', 'wb') as f:
        pickle.dump(lasso, f)
    
    print(f"LASSO回歸模型已儲存到 {output_dir}/lasso_regression_model.pkl")
    
    return lasso, best_score, best_alpha

def train_elasticnet_regression(X_train, y_train, cv=5, alphas=None, l1_ratios=None):
    """訓練並評估ElasticNet回歸模型，包括超參數調優"""
    print("\n====== ElasticNet回歸 ======")
    
    if alphas is None:
        alphas = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
    
    if l1_ratios is None:
        l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    # 設定交叉驗證
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    # 使用網格搜索尋找最佳參數
    print("正在使用網格搜索尋找最佳alpha和l1_ratio值...")
    
    elasticnet_grid = GridSearchCV(
        estimator=ElasticNet(random_state=42, max_iter=10000),
        param_grid={'alpha': alphas, 'l1_ratio': l1_ratios},
        cv=kf,
        scoring='neg_mean_squared_error',
        return_train_score=True
    )
    
    elasticnet_grid.fit(X_train, y_train)
    
    # 取得最佳參數和分數
    best_alpha = elasticnet_grid.best_params_['alpha']
    best_l1_ratio = elasticnet_grid.best_params_['l1_ratio']
    best_score = np.sqrt(-elasticnet_grid.best_score_)
    
    print(f"最佳alpha值: {best_alpha}")
    print(f"最佳l1_ratio值: {best_l1_ratio}")
    print(f"對應的RMSE: {best_score:.6f}")
    
    # 使用最佳參數重新訓練模型
    elasticnet = ElasticNet(
        alpha=best_alpha, l1_ratio=best_l1_ratio, 
        random_state=42, max_iter=10000
    )
    elasticnet.fit(X_train, y_train)
    
    # 計算R²分數
    train_r2 = elasticnet.score(X_train, y_train)
    print(f"訓練集 R²: {train_r2:.6f}")
    
    # 特徵重要性分析 (非零係數)
    coef = pd.Series(elasticnet.coef_, index=X_train.columns)
    non_zero_coef = coef[coef != 0].sort_values(ascending=False)
    
    print(f"ElasticNet選擇了 {len(non_zero_coef)}/{len(coef)} 個特徵")
    
    # 儲存非零係數
    non_zero_coef.to_csv(f'{output_dir}/elasticnet_non_zero_coefficients.csv')
    
    # 儲存模型
    with open(f'{output_dir}/elasticnet_regression_model.pkl', 'wb') as f:
        pickle.dump(elasticnet, f)
    
    print(f"ElasticNet回歸模型已儲存到 {output_dir}/elasticnet_regression_model.pkl")
    
    return elasticnet, best_score, best_alpha, best_l1_ratio

def compare_models(models_info):
    """比較不同模型的性能，並選擇最佳模型"""
    print("\n====== 模型比較 ======")
    
    # 創建比較表格
    comparison_df = pd.DataFrame({
        '模型': [info['name'] for info in models_info],
        'RMSE': [info['rmse'] for info in models_info]
    })
    
    # 按RMSE排序
    comparison_df = comparison_df.sort_values('RMSE')
    
    print("基準模型性能比較:")
    print(comparison_df)
    
    # 儲存比較結果
    comparison_df.to_csv(f'{output_dir}/model_comparison.csv', index=False)
    
    # 繪製比較圖
    plt.figure(figsize=(10, 6))
    sns.barplot(x='模型', y='RMSE', data=comparison_df)
    plt.title('基準回歸模型RMSE比較')
    plt.ylabel('RMSE (對數轉換後)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_comparison.png', dpi=300)
    plt.close()
    
    # 選擇RMSE最低的模型
    best_model_info = comparison_df.iloc[0]
    best_model_name = best_model_info['模型']
    best_model_rmse = best_model_info['RMSE']
    
    print(f"最佳基準模型: {best_model_name}，RMSE: {best_model_rmse:.6f}")
    
    # 找到最佳模型的物件
    best_model = next(info['model'] for info in models_info if info['name'] == best_model_name)
    
    # 將最佳模型複製為best_baseline_model.pkl
    with open(f'{output_dir}/best_baseline_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    # 創建最佳模型信息文件
    best_model_info_dict = next(info for info in models_info if info['name'] == best_model_name)
    
    with open(f'{output_dir}/best_model_info.txt', 'w') as f:
        f.write(f"最佳基準模型: {best_model_name}\n")
        f.write(f"RMSE: {best_model_rmse:.6f}\n")
        f.write(f"超參數: {best_model_info_dict.get('params', 'N/A')}\n")
    
    print(f"最佳基準模型信息已儲存到 {output_dir}/best_model_info.txt")
    
    return best_model, best_model_name

def make_predictions(model, X_test, log_transformed=True):
    """使用選定的模型生成預測結果"""
    print("\n====== 生成預測結果 ======")
    
    # 使用模型進行預測
    if log_transformed:
        print("正在預測對數轉換後的目標變量...")
        y_pred_log = model.predict(X_test)
        
        # 轉回原始空間
        print("將預測結果從對數空間轉回原始空間...")
        y_pred = np.expm1(y_pred_log)
    else:
        print("正在直接預測目標變量...")
        y_pred = model.predict(X_test)
    
    # 預測結果統計摘要
    print("\n預測銷售價格統計摘要:")
    print(pd.Series(y_pred).describe())
    
    # 繪製預測值分佈
    plt.figure(figsize=(10, 6))
    sns.histplot(y_pred, kde=True)
    plt.title('預測銷售價格分佈')
    plt.xlabel('預測銷售價格')
    plt.ylabel('頻率')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/predicted_price_distribution.png', dpi=300)
    plt.close()
    
    return y_pred

def create_submission_file(test_ids, predictions, filename='baseline_submission.csv'):
    """創建提交文件"""
    print("\n====== 創建提交文件 ======")
    
    # 確認ID和預測值長度相同
    if len(test_ids) != len(predictions):
        print(f"錯誤: ID數量({len(test_ids)})與預測值數量({len(predictions)})不匹配")
        return
    
    # 創建提交數據框
    submission = pd.DataFrame({
        'Id': test_ids,
        'SalePrice': predictions
    })
    
    # 儲存提交文件
    submission_path = f'{output_dir}/{filename}'
    submission.to_csv(submission_path, index=False)
    
    print(f"提交文件已儲存到 {submission_path}")
    
    return submission

def baseline_models_pipeline(train_df, test_df, y_train, test_ids=None):
    """
    完整的基準模型管線，包含：
    1. 準備目標變量
    2. 訓練與評估多個基準模型
    3. 比較模型性能
    4. 選擇最佳模型
    5. 生成預測結果
    """
    print("====== 開始執行基準模型管線 ======")
    
    # 1. 準備目標變量 (對數轉換)
    y_log = prepare_target(y_train, log_transform=True)
    
    # 2. 訓練與評估多個基準模型
    print("\n===== 訓練基準模型 =====")
    
    # 訓練線性回歸模型
    lr_model, lr_rmse = train_linear_regression(train_df, y_log)
    
    # 訓練嶺回歸模型
    ridge_model, ridge_rmse, ridge_alpha = train_ridge_regression(train_df, y_log)
    
    # 訓練LASSO回歸模型
    lasso_model, lasso_rmse, lasso_alpha = train_lasso_regression(train_df, y_log)
    
    # 訓練ElasticNet回歸模型
    elasticnet_model, elasticnet_rmse, elasticnet_alpha, elasticnet_l1 = train_elasticnet_regression(train_df, y_log)
    
    # 3. 比較模型性能
    models_info = [
        {'name': '線性回歸', 'model': lr_model, 'rmse': lr_rmse, 'params': None},
        {'name': '嶺回歸', 'model': ridge_model, 'rmse': ridge_rmse, 'params': {'alpha': ridge_alpha}},
        {'name': 'LASSO回歸', 'model': lasso_model, 'rmse': lasso_rmse, 'params': {'alpha': lasso_alpha}},
        {'name': 'ElasticNet回歸', 'model': elasticnet_model, 'rmse': elasticnet_rmse, 
         'params': {'alpha': elasticnet_alpha, 'l1_ratio': elasticnet_l1}}
    ]
    
    # 4. 選擇最佳模型
    best_model, best_model_name = compare_models(models_info)
    
    # 5. 生成預測結果
    predictions = make_predictions(best_model, test_df, log_transformed=True)
    
    # 創建提交文件
    if test_ids is not None:
        submission = create_submission_file(test_ids, predictions)
    
    print("\n====== 基準模型管線完成 ======")
    print(f"最佳基準模型: {best_model_name}")
    print(f"所有結果已儲存到 {output_dir} 目錄下")
    
    return best_model, predictions

def load_test_ids(test_path='house_prices_data/dataset/test.csv'):
    """載入測試集ID"""
    try:
        test = pd.read_csv(test_path)
        if 'Id' in test.columns:
            return test['Id']
        else:
            print(f"警告: 測試集中未找到'Id'列")
            return pd.Series(range(1, len(test) + 1))
    except Exception as e:
        print(f"載入測試集ID時發生錯誤: {e}")
        return None

if __name__ == "__main__":
    print("開始執行基準模型訓練...")
    
    # 載入特徵工程後的資料
    train_df, test_df, y_train = load_engineered_data()
    
    # 載入測試集ID (用於提交文件)
    test_ids = load_test_ids()
    
    # 執行基準模型管線
    best_model, predictions = baseline_models_pipeline(
        train_df, test_df, y_train, test_ids)
    
    print("基準模型獨立模組執行完畢!")
