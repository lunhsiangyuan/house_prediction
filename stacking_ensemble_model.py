#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
房價預測競賽 - 堆疊集成模型
House Price Prediction Competition - Stacking Ensemble Model
==============================================================================

此腳本實現堆疊集成方法，結合多個基礎模型的預測，通過元模型進行最終預測。
堆疊集成相比簡單的加權平均，能夠學習更複雜的模型組合方式，進一步提高預測性能。
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
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import warnings

warnings.filterwarnings('ignore')

# 設定輸出目錄
output_dir = 'model_results'
os.makedirs(output_dir, exist_ok=True)

# 設定隨機種子
np.random.seed(42)

def load_data(train_path='feature_engineering_results/train_engineered.csv', 
              test_path='feature_engineering_results/test_engineered.csv',
              target_path='feature_engineering_results/target_engineered.csv'):
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

def get_base_models():
    """定義基礎模型"""
    models = {
        'ridge': Ridge(alpha=0.5, random_state=42),
        'lasso': Lasso(alpha=0.0005, random_state=42),
        'elasticnet': ElasticNet(alpha=0.0005, l1_ratio=0.5, random_state=42),
        'rf': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42),
        'gbm': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42),
        'xgb': XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42),
        'lgbm': LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
    }
    
    return models

def get_meta_model():
    """定義元模型"""
    return Ridge(alpha=0.5, random_state=42)

def stacking_cv(models, X_train, y_train, X_test, n_folds=5):
    """使用交叉驗證進行堆疊集成"""
    print("\n====== 執行堆疊集成 ======")
    
    # 記錄開始時間
    start_time = time.time()
    
    # 設置交叉驗證
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # 初始化訓練集和測試集的元特徵
    train_meta_features = np.zeros((X_train.shape[0], len(models)))
    test_meta_features = np.zeros((X_test.shape[0], len(models)))
    
    # 記錄每個模型的交叉驗證分數
    cv_scores = {}
    
    # 對每個基礎模型進行交叉驗證預測
    for i, (model_name, model) in enumerate(models.items()):
        print(f"\n訓練基礎模型: {model_name}")
        
        # 初始化當前模型的測試集預測
        test_pred = np.zeros(X_test.shape[0])
        
        # 記錄當前模型的交叉驗證分數
        fold_scores = []
        
        # 對每個折進行訓練和預測
        for train_idx, val_idx in kf.split(X_train):
            # 分割訓練集和驗證集
            X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # 訓練模型
            model.fit(X_train_fold, y_train_fold)
            
            # 預測驗證集
            val_pred = model.predict(X_val_fold)
            
            # 計算RMSE
            rmse = np.sqrt(mean_squared_error(y_val_fold, val_pred))
            fold_scores.append(rmse)
            
            # 將預測結果存入元特徵
            train_meta_features[val_idx, i] = val_pred
            
            # 預測測試集並累加
            test_pred += model.predict(X_test)
        
        # 計算測試集的平均預測
        test_meta_features[:, i] = test_pred / n_folds
        
        # 計算平均交叉驗證分數
        mean_score = np.mean(fold_scores)
        cv_scores[model_name] = mean_score
        
        print(f"{model_name} 交叉驗證RMSE: {mean_score:.6f}")
    
    # 訓練元模型
    print("\n訓練元模型...")
    meta_model = get_meta_model()
    meta_model.fit(train_meta_features, y_train)
    
    # 使用元模型進行最終預測
    final_pred = meta_model.predict(test_meta_features)
    
    # 計算元模型的交叉驗證分數
    meta_cv_scores = cross_val_score(
        meta_model, train_meta_features, y_train,
        scoring='neg_mean_squared_error',
        cv=kf,
        n_jobs=-1
    )
    meta_cv_rmse = np.sqrt(-meta_cv_scores).mean()
    
    print(f"元模型交叉驗證RMSE: {meta_cv_rmse:.6f}")
    
    # 計算訓練時間
    training_time = (time.time() - start_time) / 60
    print(f"堆疊集成訓練耗時: {training_time:.2f} 分鐘")
    
    # 保存模型
    print("\n保存模型...")
    model_data = {
        'base_models': models,
        'meta_model': meta_model,
        'cv_scores': cv_scores,
        'meta_cv_rmse': meta_cv_rmse,
        'training_time': training_time
    }
    
    with open(f'{output_dir}/stacking_ensemble_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    # 繪製模型性能比較圖
    plot_model_performance(cv_scores, meta_cv_rmse)
    
    return final_pred, meta_cv_rmse, training_time, model_data

def plot_model_performance(cv_scores, meta_cv_rmse):
    """繪製模型性能比較圖"""
    print("\n繪製模型性能比較圖...")
    
    # 準備數據
    models = list(cv_scores.keys()) + ['Stacking Ensemble']
    scores = list(cv_scores.values()) + [meta_cv_rmse]
    
    # 繪製條形圖
    plt.figure(figsize=(12, 8))
    bars = plt.bar(models, scores, color='skyblue')
    
    # 突出顯示堆疊集成模型
    bars[-1].set_color('orange')
    
    # 添加數據標籤
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.6f}',
                 ha='center', va='bottom', rotation=0)
    
    plt.title('模型性能比較 (RMSE)')
    plt.xlabel('模型')
    plt.ylabel('RMSE (對數空間)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/stacking_model_performance.png', dpi=300)
    plt.close()
    
    print("模型性能比較圖已保存")

def generate_submission(final_pred, test_ids):
    """生成提交文件"""
    print("\n生成提交文件...")
    
    # 轉換回原始尺度 (如果目標變量是對數轉換的)
    y_pred_exp = np.expm1(final_pred)
    
    # 創建提交文件
    submission = pd.DataFrame({
        'Id': test_ids,
        'SalePrice': y_pred_exp
    })
    
    # 保存提交文件
    submission.to_csv(f'{output_dir}/stacking_ensemble_submission.csv', index=False)
    print(f"提交文件已保存到 {output_dir}/stacking_ensemble_submission.csv")
    
    # 繪製預測分佈圖
    plt.figure(figsize=(10, 6))
    sns.histplot(final_pred, kde=True)
    plt.title('堆疊集成: 預測分佈 (對數空間)')
    plt.xlabel('預測值 (對數)')
    plt.ylabel('頻率')
    plt.savefig(f'{output_dir}/stacking_ensemble_predicted_distribution.png', dpi=300)
    plt.close()

def update_summary_html(model_name, cv_rmse, training_time):
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
    new_row = f"""
                <tr>
                    <td>{model_name}</td>
                    <td>{cv_rmse:.6f}</td>
                    <td>~{training_time:.1f}分鐘</td>
                    <td>堆疊集成 (7個基礎模型)</td>
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
    """更新TODO.md文件，標記多模型集成方法為已完成"""
    print("\n更新TODO.md文件...")
    
    todo_path = 'TODO.md'
    
    # 檢查TODO文件是否存在
    if not os.path.exists(todo_path):
        print(f"警告: TODO文件 {todo_path} 不存在，無法更新")
        return
    
    # 讀取現有TODO文件
    with open(todo_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 更新多模型集成方法任務狀態
    updated_content = content.replace('- [ ] 多模型集成方法嘗試', '- [x] 多模型集成方法嘗試')
    
    # 更新TODO文件
    with open(todo_path, 'w', encoding='utf-8') as f:
        f.write(updated_content)
    
    print(f"TODO文件 {todo_path} 已更新")

def main():
    """主函數"""
    print("====== 堆疊集成模型訓練 ======")
    
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
    
    # 獲取基礎模型
    base_models = get_base_models()
    
    # 執行堆疊集成
    final_pred, meta_cv_rmse, training_time, model_data = stacking_cv(
        base_models, X_train, y_train, X_test, n_folds=5
    )
    
    # 生成提交文件
    generate_submission(final_pred, test_ids)
    
    # 更新摘要HTML文件
    update_summary_html("堆疊集成", meta_cv_rmse, training_time)
    
    # 更新TODO.md文件
    update_todo_md()
    
    print("\n====== 堆疊集成模型訓練完成 ======")

if __name__ == "__main__":
    main() 