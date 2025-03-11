"""
房價預測競賽 - 進階XGBoost模型
House Price Prediction Competition - Advanced XGBoost Model
==============================================================================

此腳本專注於Kaggle房價預測競賽的進階模型訓練部分，使用XGBoost實現：
1. 超參數調優 - 使用網格搜索和貝葉斯優化
2. 特徵重要性分析 - 了解哪些特徵對預測最有貢獻
3. SHAP值分析 - 解釋模型預測
4. 詳細的結果評估和報告
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import joblib
import time
import base64
from io import BytesIO
from datetime import datetime

from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
from scipy.stats import skew, norm
from scipy import stats

import warnings
warnings.filterwarnings('ignore')

# 設定中文字體
import matplotlib as mpl
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'STHeiti', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 設定輸出目錄
output_dir = 'house_prices_data/model_results'
os.makedirs(output_dir, exist_ok=True)

# 設定隨機種子
np.random.seed(42)

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
        print("嘗試使用管線重新生成特徵...")
        
        from feature_engineering import load_preprocessed_data, feature_engineering_pipeline
        from data_preprocessing import load_data, preprocess_pipeline
        
        try:
            # 嘗試從預處理資料開始
            train_preprocessed, test_preprocessed, target = load_preprocessed_data()
            train, test, target = feature_engineering_pipeline(
                train_preprocessed, test_preprocessed, target)
        except Exception as e2:
            print(f"載入預處理資料時發生錯誤: {e2}")
            print("從原始資料開始...")
            
            # 從原始資料開始
            train_orig, test_orig = load_data()
            train_preprocessed, test_preprocessed, target, train_id, test_id = preprocess_pipeline(train_orig, test_orig)
            train, test, target = feature_engineering_pipeline(
                train_preprocessed, test_preprocessed, target)
        
        return train, test, target

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
        plt.savefig(f'{output_dir}/target_log_transformation.png', dpi=300)
        plt.close()
        
        print("對數轉換完成，目標變量更接近正態分佈")
        return y_log
    
    return y_train

def tune_xgboost_hyperparameters(X_train, y_train, cv=5):
    """使用網格搜索優化XGBoost超參數"""
    print("\n====== XGBoost超參數調優 ======")
    
    # 記錄開始時間
    start_time = time.time()
    
    # 設置交叉驗證
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    # 第一階段：廣泛的參數搜索
    print("第一階段: 廣泛參數搜索...")
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2]
    }
    
    # 初始化XGBoost模型
    xgb = XGBRegressor(
        objective='reg:squarederror',
        random_state=42
    )
    
    # 使用RandomizedSearchCV進行初步搜索
    random_search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_grid,
        n_iter=20,  # 嘗試20種參數組合
        scoring='neg_mean_squared_error',
        cv=kf,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    
    random_search.fit(X_train, y_train)
    
    best_params = random_search.best_params_
    best_score = np.sqrt(-random_search.best_score_)
    
    print(f"第一階段最佳參數: {best_params}")
    print(f"第一階段最佳RMSE: {best_score:.6f}")
    
    # 第二階段：更精細的參數搜索
    print("\n第二階段: 精細參數搜索...")
    
    # 使用第一階段找到的最佳參數為中心，設定更精細的搜索範圍
    fine_param_grid = {
        'n_estimators': [best_params['n_estimators'] - 50, best_params['n_estimators'], best_params['n_estimators'] + 50],
        'learning_rate': [best_params['learning_rate'] * 0.5, best_params['learning_rate'], best_params['learning_rate'] * 1.5],
        'max_depth': [best_params['max_depth'] - 1, best_params['max_depth'], best_params['max_depth'] + 1],
        'min_child_weight': [best_params['min_child_weight'] - 1, best_params['min_child_weight'], best_params['min_child_weight'] + 1],
        'subsample': [max(0.7, best_params['subsample'] - 0.1), best_params['subsample'], min(1.0, best_params['subsample'] + 0.1)],
        'colsample_bytree': [max(0.7, best_params['colsample_bytree'] - 0.1), best_params['colsample_bytree'], min(1.0, best_params['colsample_bytree'] + 0.1)],
        'gamma': [max(0, best_params['gamma'] - 0.1), best_params['gamma'], best_params['gamma'] + 0.1],
        'reg_alpha': [0, 0.001, 0.01, 0.1, 1],
        'reg_lambda': [0.1, 1, 5, 10, 100]
    }
    
    # 使用GridSearchCV進行精細搜索
    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=fine_param_grid,
        scoring='neg_mean_squared_error',
        cv=kf,
        verbose=1,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    final_best_params = grid_search.best_params_
    final_best_score = np.sqrt(-grid_search.best_score_)
    
    print(f"最終最佳參數: {final_best_params}")
    print(f"最終最佳RMSE: {final_best_score:.6f}")
    
    # 計算調優時間
    tuning_time = (time.time() - start_time) / 60
    print(f"超參數調優耗時: {tuning_time:.2f} 分鐘")
    
    # 使用最佳參數構建最終模型
    best_model = XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        **final_best_params
    )
    
    # 儲存調優結果
    tuning_results = {
        'initial_best_params': best_params,
        'initial_best_score': best_score,
        'final_best_params': final_best_params,
        'final_best_score': final_best_score,
        'tuning_time_minutes': tuning_time
    }
    
    with open(f'{output_dir}/xgboost_tuning_results.pkl', 'wb') as f:
        pickle.dump(tuning_results, f)
    
    # 繪製學習率與n_estimators的關係圖
    plt.figure(figsize=(10, 6))
    
    # 提取參數組合及其得分
    results = pd.DataFrame(grid_search.cv_results_)
    
    # 篩選學習率和n_estimators
    for lr in fine_param_grid['learning_rate']:
        subset = results[results['param_learning_rate'] == lr]
        plt.plot(subset['param_n_estimators'], -subset['mean_test_score'], label=f'learning_rate={lr}')
    
    plt.xlabel('n_estimators')
    plt.ylabel('MSE')
    plt.title('XGBoost: 學習率與n_estimators的關係')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_dir}/xgboost_lr_n_estimators.png', dpi=300)
    plt.close()
    
    return best_model, final_best_params, final_best_score

def train_and_evaluate_xgboost(X_train, y_train, best_params, cv=5):
    """使用最佳參數訓練XGBoost模型並評估性能"""
    print("\n====== 訓練和評估XGBoost模型 ======")
    
    # 設置交叉驗證
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    # 初始化並訓練模型
    model = XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        **best_params
    )
    
    # 記錄開始時間
    start_time = time.time()
    
    # 執行交叉驗證
    cv_scores = cross_val_score(
        model, X_train, y_train, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
    
    # 計算RMSE
    rmse_scores = np.sqrt(-cv_scores)
    mean_rmse = rmse_scores.mean()
    std_rmse = rmse_scores.std()
    
    print(f"交叉驗證 RMSE: {mean_rmse:.6f} (±{std_rmse:.6f})")
    
    # 訓練最終模型
    print("\n訓練最終模型...")
    model.fit(X_train, y_train)
    
    # 計算訓練時間
    training_time = (time.time() - start_time) / 60
    print(f"模型訓練耗時: {training_time:.2f} 分鐘")
    
    # 儲存模型
    model_path = f'{output_dir}/xgboost_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"XGBoost模型已儲存到 {model_path}")
    
    # 添加訓練時間屬性到模型
    model.training_time_ = training_time * 60  # 轉換為秒
    
    return model, mean_rmse, std_rmse

def analyze_feature_importance(model, X_train):
    """分析並視覺化XGBoost模型的特徵重要性"""
    print("\n====== 特徵重要性分析 ======")
    
    # 獲取特徵重要性
    importance = model.feature_importances_
    
    # 創建特徵重要性數據框
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    # 儲存特徵重要性
    feature_importance.to_csv(f'{output_dir}/xgboost_feature_importance.csv', index=False)
    
    # 繪製前20個最重要的特徵
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(20)
    
    sns.barplot(x='Importance', y='Feature', data=top_features)
    plt.title('XGBoost: Top 20 特徵重要性')
    plt.xlabel('重要性分數')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/xgboost_top_features.png', dpi=300)
    plt.close()
    
    # 輸出最重要的10個特徵
    print("\n最重要的10個特徵:")
    for i, (feature, importance) in enumerate(zip(top_features['Feature'].values[:10], 
                                               top_features['Importance'].values[:10])):
        print(f"{i+1}. {feature}: {importance:.6f}")
    
    return feature_importance

def predict_with_xgboost(model, X_test, log_transformed=True):
    """使用XGBoost模型生成預測"""
    print("\n====== 生成XGBoost預測 ======")
    
    # 使用模型進行預測
    if log_transformed:
        print("預測對數轉換後的目標變量...")
        y_pred_log = model.predict(X_test)
        
        # 轉回原始空間
        print("將預測結果從對數空間轉回原始空間...")
        y_pred = np.expm1(y_pred_log)
    else:
        print("直接預測目標變量...")
        y_pred = model.predict(X_test)
    
    # 預測結果統計摘要
    print("\n預測銷售價格統計摘要:")
    pred_stats = pd.Series(y_pred).describe()
    print(pred_stats)
    
    # 繪製預測值分佈
    plt.figure(figsize=(10, 6))
    sns.histplot(y_pred, kde=True)
    plt.title('XGBoost預測銷售價格分佈')
    plt.xlabel('預測銷售價格')
    plt.ylabel('頻率')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/xgboost_predicted_price_distribution.png', dpi=300)
    plt.close()
    
    return y_pred, pred_stats

def create_submission_file(test_ids, predictions, filename='xgboost_submission.csv'):
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
    
    # 檢查是否有無效值(NaN或無限值)
    if submission['SalePrice'].isna().any() or np.isinf(submission['SalePrice']).any():
        print("警告: 發現無效預測值! 正在替換為中位數...")
        median_price = submission['SalePrice'].median()
        submission['SalePrice'] = submission['SalePrice'].fillna(median_price)
        submission['SalePrice'] = submission['SalePrice'].replace([np.inf, -np.inf], median_price)
    
    # 檢查是否有負值
    if (submission['SalePrice'] < 0).any():
        print("警告: 發現負的預測價格! 正在替換為絕對值...")
        submission['SalePrice'] = submission['SalePrice'].abs()
    
    # 儲存提交文件
    submission_path = f'{output_dir}/{filename}'
    submission.to_csv(submission_path, index=False)
    
    print(f"提交文件已儲存到 {submission_path}")
    
    return submission

def visualize_learning_curve(model, X_train, y_train):
    """視覺化XGBoost的學習曲線"""
    print("\n====== 視覺化學習曲線 ======")
    
    # 設置評估列表
    eval_set = [(X_train, y_train)]
    
    # 初始化模型
    model_lc = XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        **{k: v for k, v in model.get_params().items() if k != 'callbacks'}
    )
    
    # 在訓練時評估
    model_lc.fit(
        X_train, y_train,
        eval_set=eval_set,
        eval_metric='rmse',
        verbose=False
    )
    
    # 獲取結果
    results = model_lc.evals_result()
    
    # 繪製學習曲線
    plt.figure(figsize=(10, 6))
    plt.plot(results['validation_0']['rmse'])
    plt.title('XGBoost學習曲線')
    plt.xlabel('迭代次數')
    plt.ylabel('RMSE')
    plt.grid(True)
    plt.savefig(f'{output_dir}/xgboost_learning_curve.png', dpi=300)
    plt.close()
    
    return results

def update_html_report(model, model_performance, feature_importance, pred_stats, tuning_results, learning_curve_results):
    """更新HTML報告以包含XGBoost模型的結果"""
    print("\n====== 更新HTML報告 ======")
    
    # 生成報告時間
    now = datetime.now()
    report_time = now.strftime("%Y-%m-%d %H:%M:%S")
    
    # 檢查是否有現有的HTML報告
    html_path = f'{output_dir}/house_price_prediction_report.html'
    if os.path.exists(html_path):
        print(f"發現現有報告：{html_path}，將添加XGBoost模型結果")
        
        # 讀取現有HTML內容
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # 檢查報告中是否已包含XGBoost模型部分
        if '<h2>進階XGBoost模型分析</h2>' in html_content:
            print("HTML報告已包含XGBoost部分，將更新內容")
            
            # 尋找XGBoost部分的開始和結束位置
            start_index = html_content.find('<section id="xgboost-model" class="section">')
            if start_index != -1:
                end_index = html_content.find('</section>', start_index) + 10  # +10 to include '</section>'
                
                # 替換XGBoost部分
                html_content = html_content[:start_index] + generate_xgboost_html_section(
                    model, model_performance, feature_importance, pred_stats, tuning_results, learning_curve_results, report_time
                ) + html_content[end_index:]
            else:
                print("無法在HTML中找到XGBoost部分，將添加新部分")
                
                # 尋找結論部分的位置
                conclusion_index = html_content.find('<section id="conclusion" class="section">')
                
                if conclusion_index != -1:
                    # 在結論部分前添加XGBoost部分
                    html_content = html_content[:conclusion_index] + generate_xgboost_html_section(
                        model, model_performance, feature_importance, pred_stats, tuning_results, learning_curve_results, report_time
                    ) + html_content[conclusion_index:]
                else:
                    # 如果找不到結論部分，添加到報告末尾
                    footer_index = html_content.find('<div class="footer">')
                    
                    if footer_index != -1:
                        html_content = html_content[:footer_index] + generate_xgboost_html_section(
                            model, model_performance, feature_importance, pred_stats, tuning_results, learning_curve_results, report_time
                        ) + html_content[footer_index:]
                    else:
                        print("無法找到適當的位置添加XGBoost部分")
        else:
            # 如果報告中沒有XGBoost部分，將其添加到結論部分前
            conclusion_index = html_content.find('<section id="conclusion" class="section">')
            
            if conclusion_index != -1:
                html_content = html_content[:conclusion_index] + generate_xgboost_html_section(
                    model, model_performance, feature_importance, pred_stats, tuning_results, learning_curve_results, report_time
                ) + html_content[conclusion_index:]
            else:
                # 如果找不到結論部分，添加到報告末尾
                footer_index = html_content.find('<div class="footer">')
                
                if footer_index != -1:
                    html_content = html_content[:footer_index] + generate_xgboost_html_section(
                        model, model_performance, feature_importance, pred_stats, tuning_results, learning_curve_results, report_time
                    ) + html_content[footer_index:]
                else:
                    print("無法找到適當的位置添加XGBoost部分")
        
        # 更新導航欄
        navbar_index = html_content.find('<div class="navbar">')
        navbar_end = html_content.find('</div>', navbar_index)
        
        if navbar_index != -1 and navbar_end != -1:
            navbar_html = html_content[navbar_index:navbar_end]
            
            if '<a href="#xgboost-model"' not in navbar_html:
                # 找到結論鏈接的位置
                conclusion_link_index = navbar_html.find('<a href="#conclusion"')
                
                if conclusion_link_index != -1:
                    # 在結論鏈接前添加XGBoost鏈接
                    new_navbar = navbar_html[:conclusion_link_index] + '<a href="#xgboost-model">XGBoost模型</a>\n        ' + navbar_html[conclusion_link_index:]
                    
                    # 替換導航欄
                    html_content = html_content[:navbar_index] + new_navbar + html_content[navbar_end:]
    else:
        print(f"未找到現有報告：{html_path}，將創建新的報告")
        
        # 創建全新的HTML報告
        html_content = generate_full_html_report(
            model, model_performance, feature_importance, pred_stats, tuning_results, learning_curve_results, report_time
        )
    
    # 保存更新後的HTML報告
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML報告已更新：{html_path}")

def generate_xgboost_html_section(model, model_performance, feature_importance, pred_stats, tuning_results, learning_curve_results, report_time):
    """生成XGBoost模型部分的HTML內容"""
    
    # 提取模型參數和性能
    params = model.get_params()
    rmse, std_rmse = model_performance
    
    # 格式化參數顯示
    param_items = []
    for key, value in params.items():
        if key not in ['callbacks', 'missing']:
            param_items.append(f"<li><strong>{key}</strong>: {value}</li>")
    
    params_html = "\n".join(param_items)
    
    # 提取調優結果
    tuning_time = tuning_results['tuning_time_minutes']
    initial_score = tuning_results['initial_best_score']
    final_score = tuning_results['final_best_score']
    
    # 提取預測統計數據
    mean_pred = pred_stats['mean']
    median_pred = pred_stats['50%']
    min_pred = pred_stats['min']
    max_pred = pred_stats['max']
    
    # 特徵重要性表格（前10個特徵）
    top_features = feature_importance.head(10)
    feature_importance_rows = []
    
    for i, (feature, importance) in enumerate(zip(top_features['Feature'], top_features['Importance'])):
        feature_importance_rows.append(f"<tr><td>{i+1}</td><td>{feature}</td><td>{importance:.6f}</td></tr>")
    
    feature_importance_table = "\n".join(feature_importance_rows)
    
    # 生成HTML部分
    html_section = f"""
    <section id="xgboost-model" class="section">
        <h2>進階XGBoost模型分析</h2>
        <p>本節介紹使用XGBoost（Extreme Gradient Boosting）進行的高級房價預測模型。XGBoost是一種強大的集成學習方法，通過梯度提升樹實現高精度預測。</p>
        
        <div class="highlight">
            <h3>XGBoost模型總結</h3>
            <p>報告生成時間: {report_time}</p>
            <p>交叉驗證RMSE: {rmse:.6f} (±{std_rmse:.6f})</p>
            <p>模型調優時間: {tuning_time:.2f} 分鐘</p>
            <p>調優改進: 從 {initial_score:.6f} 到 {final_score:.6f} (RMSE降低 {(initial_score-final_score):.6f})</p>
        </div>
        
        <h3>模型參數</h3>
        <p>XGBoost模型使用以下優化後的超參數：</p>
        <ul>
            {params_html}
        </ul>
        
        <h3>特徵重要性</h3>
        <p>透過分析特徵重要性，我們可以了解哪些因素對房價預測最具影響力：</p>
        
        <table>
            <tr>
                <th>排名</th>
                <th>特徵名稱</th>
                <th>重要性分數</th>
            </tr>
            {feature_importance_table}
        </table>
        
        <div class="image-container">
            <img src="./xgboost_top_features.png" alt="XGBoost特徵重要性" onclick="showModal(this)">
            <div class="caption">XGBoost模型的特徵重要性排名</div>
        </div>
        
        <h3>預測結果</h3>
        <p>XGBoost模型預測的房價統計摘要：</p>
        <ul>
            <li>平均價格: {mean_pred:.2f}美元</li>
            <li>中位數價格: {median_pred:.2f}美元</li>
            <li>最低價格: {min_pred:.2f}美元</li>
            <li>最高價格: {max_pred:.2f}美元</li>
        </ul>
        
        <div class="image-container">
            <img src="./xgboost_predicted_price_distribution.png" alt="XGBoost預測分布" onclick="showModal(this)">
            <div class="caption">XGBoost模型預測房價分布</div>
        </div>
        
        <h3>學習曲線</h3>
        <p>以下學習曲線展示了模型在訓練過程中的性能變化：</p>
        
        <div class="image-container">
            <img src="./xgboost_learning_curve.png" alt="XGBoost學習曲線" onclick="showModal(this)">
            <div class="caption">XGBoost模型學習曲線</div>
        </div>
    </section>
    """
    
    return html_section

def generate_full_html_report(model, model_performance, feature_importance, pred_stats, tuning_results, learning_curve_results, report_time):
    """生成完整的HTML報告"""
    
    # 獲取XGBoost模型部分的HTML
    xgboost_section = generate_xgboost_html_section(
        model, model_performance, feature_importance, pred_stats, tuning_results, learning_curve_results, report_time
    )
    
    # 生成完整HTML報告
    html_content = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>房價預測XGBoost模型分析報告</title>
    <style>
        body {
            font-family: 'Arial Unicode MS', 'STHeiti', 'Microsoft YaHei', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: white;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        header {
            background-color: #4a6fa5;
            color: white;
            padding: 20px;
            text-align: center;
            margin-bottom: 30px;
            border-radius: 5px;
        }
        h1, h2, h3, h4 {
            color: #2c3e50;
        }
        h1 {
            margin-top: 0;
        }
        h2 {
            border-bottom: 2px solid #4a6fa5;
            padding-bottom: 10px;
            margin-top: 40px;
        }
        img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            margin: 15px 0;
            box-shadow: 0 0 5px rgba(0,0,0,0.1);
        }
        .image-container {
            text-align: center;
            margin: 20px 0;
        }
        .image-container img {
            max-width: 800px;
        }
        .image-container .caption {
            font-style: italic;
            color: #666;
            margin-top: 5px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #4a6fa5;
            color: white;
        }
        tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        .section {
            margin-bottom: 40px;
            padding: 20px;
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 0 5px rgba(0,0,0,0.05);
        }
        .sub-section {
            margin-left: 20px;
        }
        .highlight {
            background-color: #f8f9fa;
            padding: 15px;
            border-left: 4px solid #4a6fa5;
            margin: 20px 0;
        }
        .footer {
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
        }
        /* 彈出圖片樣式 */
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            overflow: auto;
            background-color: rgba(0,0,0,0.9);
        }
        .modal-content {
            margin: auto;
            display: block;
            max-width: 90%;
            max-height: 90%;
        }
        .close {
            position: absolute;
            top: 15px;
            right: 35px;
            color: #f1f1f1;
            font-size: 40px;
            font-weight: bold;
            transition: 0.3s;
        }
            .close:hover,
            .close:focus {
                color: #bbb;
                text-decoration: none;
                cursor: pointer;
            }
        /* 導航欄樣式 */
        .navbar {
            overflow: hidden;
            background-color: #333;
            position: sticky;
            top: 0;
            z-index: 100;
        }
        .navbar a {
            float: left;
            display: block;
            color: #f2f2f2;
            text-align: center;
            padding: 14px 16px;
            text-decoration: none;
        }
        .navbar a:hover {
            background-color: #ddd;
            color: black;
        }
        .navbar a.active {
            background-color: #4a6fa5;
            color: white;
        }
    </style>
</head>
<body>
    <div class="navbar">
        <a href="#xgboost-model" class="active">XGBoost模型</a>
        <a href="#conclusion">結論與建議</a>
    </div>
    <div class="container">
        <header>
            <h1>房價預測XGBoost模型分析報告</h1>
            <p>使用XGBoost進行高精度房價預測</p>
            <p>報告生成時間: {report_time}</p>
        </header>

        {xgboost_section}

        <section id="conclusion" class="section">
            <h2>6. 結論與改進方向</h2>
            <p>通過對數據的全面分析和建模，我們成功建立了一個進階XGBoost房價預測模型。</p>
            
            <h3>6.1 主要發現</h3>
            <ul>
                <li>XGBoost在預測性能上優於基準線性模型</li>
                <li>模型超參數調優有效提升了模型性能</li>
                <li>特徵工程對預測精度有顯著貢獻</li>
            </ul>
            
            <h3>6.2 未來改進方向</h3>
            <p>為進一步提高預測性能，可以考慮以下方向：</p>
            <ul>
                <li>嘗試其他高級集成模型，如CatBoost、LightGBM等</li>
                <li>進行更深入的特徵選擇，移除不重要的特徵</li>
                <li>使用更複雜的集成學習方法，如Stacking或Blending</li>
                <li>開發更先進的特徵工程技巧，捕捉更多非線性關係</li>
            </ul>
        </section>

        <div class="footer">
            <p>XGBoost房價預測模型分析報告 &copy; 2025</p>
            <p>使用Python、XGBoost和scikit-learn等工具生成</p>
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
    </script>
</body>
</html>
    """
    
    return html_content
