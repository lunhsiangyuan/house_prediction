#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
房價預測競賽 - 貝葉斯優化XGBoost模型
House Price Prediction Competition - Bayesian Optimized XGBoost Model
==============================================================================

此腳本使用貝葉斯優化方法來調整XGBoost模型的超參數，以進一步提高預測性能。
貝葉斯優化相比傳統的網格搜索和隨機搜索，能更有效地探索超參數空間。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import time
from datetime import datetime
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import warnings
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

warnings.filterwarnings('ignore')

# 設定輸出目錄
output_dir = 'model_results'
os.makedirs(output_dir, exist_ok=True)

# 設定隨機種子
np.random.seed(42)

def load_data(train_path='feature_engineering_results/advanced/train_advanced_engineered.csv', 
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
        print("嘗試載入基本特徵工程資料...")
        
        try:
            train = pd.read_csv('feature_engineering_results/train_engineered.csv')
            test = pd.read_csv('feature_engineering_results/test_engineered.csv')
            target = pd.read_csv('feature_engineering_results/target_engineered.csv')
            
            print(f"基本特徵工程後訓練集大小: {train.shape}")
            print(f"基本特徵工程後測試集大小: {test.shape}")
            print(f"目標變量大小: {target.shape}")
            
            return train, test, target.iloc[:, 0]
        except Exception as e2:
            print(f"載入基本特徵工程資料時發生錯誤: {e2}")
            return None, None, None

def load_test_ids(test_path='dataset/test.csv'):
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

def bayesian_optimize_xgboost(X_train, y_train, cv=5, n_iter=50):
    """使用貝葉斯優化調整XGBoost超參數"""
    print("\n====== 使用貝葉斯優化調整XGBoost超參數 ======")
    
    # 記錄開始時間
    start_time = time.time()
    
    # 設置交叉驗證
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    # 定義超參數搜索空間
    search_spaces = {
        'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
        'n_estimators': Integer(100, 1000),
        'max_depth': Integer(3, 10),
        'min_child_weight': Real(0.5, 10, prior='log-uniform'),
        'subsample': Real(0.5, 1.0),
        'colsample_bytree': Real(0.5, 1.0),
        'gamma': Real(0.001, 5, prior='log-uniform'),
        'reg_alpha': Real(0.001, 10, prior='log-uniform'),
        'reg_lambda': Real(0.001, 10, prior='log-uniform')
    }
    
    # 初始化XGBoost模型
    xgb = XGBRegressor(
        objective='reg:squarederror',
        random_state=42
    )
    
    # 初始化貝葉斯搜索
    bayes_search = BayesSearchCV(
        estimator=xgb,
        search_spaces=search_spaces,
        n_iter=n_iter,
        scoring='neg_mean_squared_error',
        cv=kf,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    
    # 執行搜索
    print(f"開始貝葉斯優化，迭代次數: {n_iter}...")
    bayes_search.fit(X_train, y_train)
    
    # 獲取最佳參數和分數
    best_params = bayes_search.best_params_
    best_score = np.sqrt(-bayes_search.best_score_)
    
    print(f"最佳參數: {best_params}")
    print(f"最佳RMSE: {best_score:.6f}")
    
    # 計算調優時間
    tuning_time = (time.time() - start_time) / 60
    print(f"貝葉斯優化耗時: {tuning_time:.2f} 分鐘")
    
    # 使用最佳參數構建最終模型
    best_model = XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        **best_params
    )
    
    # 在整個訓練集上擬合模型
    best_model.fit(X_train, y_train)
    
    # 儲存調優結果
    tuning_results = {
        'best_params': best_params,
        'best_score': best_score,
        'tuning_time_minutes': tuning_time,
        'all_results': bayes_search.cv_results_
    }
    
    with open(f'{output_dir}/bayesian_xgboost_tuning_results.pkl', 'wb') as f:
        pickle.dump(tuning_results, f)
    
    # 繪製超參數重要性圖
    plot_hyperparameter_importance(bayes_search)
    
    return best_model, best_params, best_score

def plot_hyperparameter_importance(bayes_search):
    """繪製超參數重要性圖"""
    print("\n繪製超參數重要性圖...")
    
    # 獲取所有結果
    results = pd.DataFrame(bayes_search.cv_results_)
    
    # 提取參數名稱
    param_names = [name for name in results.columns if "param_" in name]
    param_names = [name.replace("param_", "") for name in param_names]
    
    # 計算每個參數的重要性
    importances = []
    
    for param in param_names:
        # 獲取參數值
        param_values = results[f"param_{param}"].values
        
        # 如果參數值都相同，則跳過
        if len(set(param_values)) <= 1:
            continue
        
        # 計算參數值與分數的相關性
        param_importance = np.abs(np.corrcoef(
            [param_values, -results["mean_test_score"].values]
        )[0, 1])
        
        importances.append((param, param_importance))
    
    # 按重要性排序
    importances.sort(key=lambda x: x[1], reverse=True)
    
    # 繪製重要性圖
    plt.figure(figsize=(10, 6))
    params = [imp[0] for imp in importances]
    importance_values = [imp[1] for imp in importances]
    
    sns.barplot(x=importance_values, y=params)
    plt.title('XGBoost超參數重要性')
    plt.xlabel('重要性 (與分數的絕對相關性)')
    plt.ylabel('超參數')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/bayesian_xgboost_hyperparameter_importance.png', dpi=300)
    plt.close()
    
    print("超參數重要性圖已保存")

def train_and_evaluate_model(X_train, y_train, X_test, test_ids, best_params=None, cv=5):
    """訓練並評估XGBoost模型"""
    print("\n====== 訓練並評估XGBoost模型 ======")
    
    # 記錄開始時間
    start_time = time.time()
    
    # 如果沒有提供最佳參數，則使用默認參數
    if best_params is None:
        print("使用默認參數...")
        model = XGBRegressor(
            objective='reg:squarederror',
            random_state=42
        )
    else:
        print("使用貝葉斯優化的最佳參數...")
        model = XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            **best_params
        )
    
    # 設置交叉驗證
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    # 交叉驗證評估
    print(f"執行{cv}折交叉驗證...")
    cv_scores = cross_val_score(
        model, X_train, y_train, 
        scoring='neg_mean_squared_error',
        cv=kf,
        verbose=1,
        n_jobs=-1
    )
    
    # 計算RMSE
    cv_rmse = np.sqrt(-cv_scores)
    mean_cv_rmse = cv_rmse.mean()
    std_cv_rmse = cv_rmse.std()
    
    print(f"交叉驗證RMSE: {mean_cv_rmse:.6f} (±{std_cv_rmse:.6f})")
    
    # 在整個訓練集上擬合模型
    print("在整個訓練集上擬合模型...")
    model.fit(X_train, y_train)
    
    # 計算訓練時間
    training_time = (time.time() - start_time) / 60
    print(f"模型訓練耗時: {training_time:.2f} 分鐘")
    
    # 保存模型
    print("保存模型...")
    with open(f'{output_dir}/bayesian_xgboost_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # 特徵重要性分析
    print("\n分析特徵重要性...")
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # 保存特徵重要性
    feature_importance.to_csv(f'{output_dir}/bayesian_xgboost_feature_importance.csv', index=False)
    
    # 繪製特徵重要性圖
    top_features = feature_importance.head(20)
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=top_features)
    plt.title('貝葉斯優化XGBoost: Top 20 特徵重要性')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/bayesian_xgboost_top_features.png', dpi=300)
    plt.close()
    
    # 生成預測
    print("\n生成預測...")
    y_pred = model.predict(X_test)
    
    # 轉換回原始尺度 (如果目標變量是對數轉換的)
    y_pred_exp = np.expm1(y_pred)
    
    # 創建提交文件
    submission = pd.DataFrame({
        'Id': test_ids,
        'SalePrice': y_pred_exp
    })
    
    # 保存提交文件
    submission.to_csv(f'{output_dir}/bayesian_xgboost_submission.csv', index=False)
    print(f"提交文件已保存到 {output_dir}/bayesian_xgboost_submission.csv")
    
    # 繪製預測分佈圖
    plt.figure(figsize=(10, 6))
    sns.histplot(y_pred, kde=True)
    plt.title('貝葉斯優化XGBoost: 預測分佈 (對數空間)')
    plt.xlabel('預測值 (對數)')
    plt.ylabel('頻率')
    plt.savefig(f'{output_dir}/bayesian_xgboost_predicted_distribution.png', dpi=300)
    plt.close()
    
    # 返回模型、交叉驗證RMSE和訓練時間
    return model, mean_cv_rmse, training_time

def update_summary_html(model_name, cv_rmse, training_time, best_params=None):
    """更新摘要HTML文件"""
    print("\n更新摘要HTML文件...")
    
    summary_path = f'{output_dir}/@summary.html'
    
    # 檢查摘要文件是否存在
    if not os.path.exists(summary_path):
        print(f"警告: 摘要文件 {summary_path} 不存在，無法更新")
        return
    
    # 讀取現有摘要文件
    with open(summary_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找模型性能比較表格
    table_start = content.find('<table>')
    table_end = content.find('</table>', table_start)
    
    if table_start == -1 or table_end == -1:
        print("警告: 在摘要文件中未找到模型性能比較表格，無法更新")
        return
    
    # 提取表格內容
    table_content = content[table_start:table_end + 8]
    
    # 創建新的表格行
    params_str = "貝葉斯優化" if best_params else "默認參數"
    new_row = f"""
                <tr>
                    <td>{model_name}</td>
                    <td>{cv_rmse:.6f}</td>
                    <td>~{training_time:.1f}分鐘</td>
                    <td>{params_str}</td>
                </tr>
                </table>"""
    
    # 替換表格
    updated_table = table_content.replace('</table>', new_row)
    updated_content = content.replace(table_content, updated_table)
    
    # 更新摘要文件
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(updated_content)
    
    print(f"摘要文件 {summary_path} 已更新")

def update_todo_md():
    """更新TODO.md文件，標記貝葉斯優化任務為已完成"""
    print("\n更新TODO.md文件...")
    
    todo_path = 'TODO.md'
    
    # 檢查TODO文件是否存在
    if not os.path.exists(todo_path):
        print(f"警告: TODO文件 {todo_path} 不存在，無法更新")
        return
    
    # 讀取現有TODO文件
    with open(todo_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 更新貝葉斯優化任務狀態
    updated_content = content.replace('- [ ] 貝葉斯優化', '- [x] 貝葉斯優化')
    
    # 更新TODO文件
    with open(todo_path, 'w', encoding='utf-8') as f:
        f.write(updated_content)
    
    print(f"TODO文件 {todo_path} 已更新")

def main():
    """主函數"""
    print("====== 貝葉斯優化XGBoost模型訓練 ======")
    
    # 載入資料
    X_train, X_test, y_train = load_data()
    
    if X_train is None or X_test is None or y_train is None:
        print("無法載入資料，程序終止")
        return
    
    # 載入測試集ID
    test_ids = load_test_ids()
    
    if test_ids is None:
        print("無法載入測試集ID，使用默認ID")
        test_ids = pd.Series(range(1, X_test.shape[0] + 1))
    
    # 使用貝葉斯優化調整超參數
    best_model, best_params, best_score = bayesian_optimize_xgboost(X_train, y_train, cv=5, n_iter=50)
    
    # 訓練並評估模型
    model, cv_rmse, training_time = train_and_evaluate_model(X_train, y_train, X_test, test_ids, best_params)
    
    # 更新摘要HTML文件
    update_summary_html("貝葉斯優化XGBoost", cv_rmse, training_time, best_params)
    
    # 更新TODO.md文件
    update_todo_md()
    
    print("\n====== 貝葉斯優化XGBoost模型訓練完成 ======")

if __name__ == "__main__":
    main() 