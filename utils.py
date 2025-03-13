"""
工具函數模塊
用於處理結果保存和報告生成
"""

import os
import numpy as np
import pandas as pd
from data_handling import load_test_ids
from bs4 import BeautifulSoup
from datetime import datetime

def generate_submission(predictions, output_dir='model_results'):
    """
    生成提交文件
    
    Parameters:
    -----------
    predictions : array-like
        預測值數組
    output_dir : str
        輸出目錄路徑
    
    Returns:
    --------
    str
        提交文件路徑
    """
    print("\n生成提交文件...")
    
    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 載入測試集ID
    test_ids = load_test_ids()
    if test_ids is None:
        print("警告: 無法載入測試集ID，使用默認ID")
        test_ids = pd.Series(range(1, len(predictions) + 1))
    
    # 檢查是否需要進行指數轉換（假設預測值在對數空間）
    if np.mean(predictions) < 20:  # 簡單的啟發式檢查
        print("對預測結果進行指數轉換...")
        predictions = np.expm1(predictions)
    
    # 創建提交DataFrame
    submission = pd.DataFrame({
        'Id': test_ids,
        'SalePrice': predictions
    })
    
    # 保存提交文件
    submission_path = f'{output_dir}/super_advanced_stacking_ensemble_submission.csv'
    submission.to_csv(submission_path, index=False)
    print(f"提交文件已保存至: {submission_path}")
    
    return submission_path

def update_html_summary(results_summary, output_dir='model_results'):
    """更新HTML摘要文件"""
    print("\n更新HTML摘要文件...")
    
    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)
    
    summary_path = f'{output_dir}/@summary.html'
    
    # 提取結果摘要信息
    rmse = results_summary.get('rmse', 0)
    training_time = results_summary.get('training_time', 0)
    model_name = results_summary.get('model_name', 'Unknown Model')
    features_count = results_summary.get('features_count', 0)
    timestamp = results_summary.get('timestamp', pd.Timestamp.now().strftime('%Y%m%d_%H%M%S'))
    best_models = results_summary.get('best_models', {})
    model_type = results_summary.get('model_type', 'stacking')  # 新增：模型類型
    
    # 檢查摘要文件是否存在
    if not os.path.exists(summary_path):
        print(f"摘要文件 {summary_path} 不存在，創建新文件")
        create_new_html_summary(rmse, training_time, model_name, features_count, best_models, output_dir)
        return
    
    try:
        # 讀取現有摘要文件
        with open(summary_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 根據模型類型選擇對應的表格
        if model_type == 'base':
            table_start = content.find('<h2 class="model-type">基礎模型</h2>')
        elif model_type == 'optimized':
            table_start = content.find('<h2 class="model-type">優化模型</h2>')
        else:  # stacking
            table_start = content.find('<h2 class="model-type">堆疊模型</h2>')
        
        if table_start == -1:
            print("警告: 在摘要文件中未找到對應的模型類型表格，創建新文件")
            create_new_html_summary(rmse, training_time, model_name, features_count, best_models, output_dir)
            return
        
        # 找到表格結束位置
        table_end = content.find('</table>', table_start)
        if table_end == -1:
            print("警告: 表格結構不完整，創建新文件")
            create_new_html_summary(rmse, training_time, model_name, features_count, best_models, output_dir)
            return
        
        # 創建新的表格行
        best_models_str = ", ".join([f"{segment}: {model}" for segment, model in best_models.items()])
        new_row = f"""
                    <tr>
                        <td>{model_name}</td>
                        <td>{rmse:.6f}</td>
                        <td>~{training_time:.1f}分鐘</td>
                        <td>{features_count}</td>
                        {'<td>' + best_models_str + '</td>' if model_type == 'stacking' else ''}
                        <td>{timestamp}</td>
                    </tr>
                    </table>"""
        
        # 替換表格
        table_content = content[table_start:table_end + 8]
        updated_table = table_content.replace('</table>', new_row)
        updated_content = content.replace(table_content, updated_table)
        
        # 更新摘要文件
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        
        print(f"摘要文件 {summary_path} 已更新")
    
    except Exception as e:
        print(f"更新摘要HTML文件時出錯: {str(e)}")
        print("創建新的摘要文件...")
        create_new_html_summary(rmse, training_time, model_name, features_count, best_models, output_dir)

def create_new_html_summary(rmse, training_time, model_name, features_count, best_models, output_dir):
    """創建新的HTML摘要文件"""
    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)
    
    summary_path = f'{output_dir}/@summary.html'
    
    # 格式化最佳模型信息
    best_models_str = ", ".join([f"{segment}: {model}" for segment, model in best_models.items()])
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    
    # 創建HTML內容
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>房價預測模型性能比較</title>
    <style>
        body { font-family: "Microsoft JhengHei", Arial, sans-serif; margin: 20px; }
        h1, h2 { color: #2c3e50; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        tr:hover { background-color: #f5f5f5; }
        .best { background-color: #e8f8f5; font-weight: bold; }
        .model-section { margin-bottom: 30px; }
        .model-type { color: #2980b9; }
        .error-rate { color: #e74c3c; }
        .timestamp { color: #7f8c8d; font-size: 0.9em; }
    </style>
</head>
<body>
    <h1>房價預測模型性能比較</h1>
    
    <div class="model-section">
        <h2 class="model-type">基礎模型</h2>
        <table>
            <tr>
                <th>模型</th>
                <th>RMSE (對數空間)</th>
                <th>相對誤差率</th>
                <th>訓練時間</th>
                <th>特徵數量</th>
                <th>時間戳</th>
            </tr>
        </table>
    </div>
    
    <div class="model-section">
        <h2 class="model-type">優化模型</h2>
        <table>
            <tr>
                <th>模型</th>
                <th>RMSE (對數空間)</th>
                <th>相對誤差率</th>
                <th>訓練時間</th>
                <th>特徵數量</th>
                <th>時間戳</th>
            </tr>
        </table>
    </div>
    
    <div class="model-section">
        <h2 class="model-type">堆疊模型</h2>
        <table>
            <tr>
                <th>模型</th>
                <th>RMSE (對數空間)</th>
                <th>相對誤差率</th>
                <th>訓練時間</th>
                <th>特徵數量</th>
                <th>最佳分段模型</th>
                <th>時間戳</th>
            </tr>
        </table>
    </div>
    
    <h2>模型性能比較圖</h2>
    <img src="first_layer_model_performance.png" alt="第一層模型性能比較" style="max-width: 100%;">
    <img src="all_model_performance.png" alt="所有模型性能比較" style="max-width: 100%;">
    
    <h2>預測結果分布</h2>
    <img src="prediction_distribution.png" alt="預測結果分布" style="max-width: 100%;">
    
    <h2>分段模型性能</h2>
    <img src="segmented_model_performance.png" alt="分段模型性能" style="max-width: 100%;">
    
    <p class="timestamp">最後更新時間: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
</body>
</html>"""
    
    # 寫入HTML文件
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"新的摘要文件已創建: {summary_path}")

def update_summary_html(model_name, cv_rmse, training_time, output_dir='model_results'):
    """
    舊版更新摘要HTML文件函數（保留向後兼容性）
    
    Parameters:
    -----------
    model_name : str
        模型名稱
    cv_rmse : float
        交叉驗證RMSE
    training_time : float
        訓練時間（分鐘）
    output_dir : str
        輸出目錄路徑
    """
    print("\n更新摘要HTML文件（舊版函數）...")
    
    # 轉換為新版格式
    results_summary = {
        'rmse': cv_rmse,
        'training_time': training_time,
        'model_name': model_name,
        'features_count': 0,  # 未知
        'timestamp': pd.Timestamp.now().strftime('%Y%m%d_%H%M%S'),
        'best_models': {}
    }
    
    # 調用新版函數
    update_html_summary(results_summary, output_dir)

def create_summary_html(model_name, cv_rmse, training_time, output_dir='model_results'):
    """
    舊版創建摘要HTML文件函數（保留向後兼容性）
    
    Parameters:
    -----------
    model_name : str
        模型名稱
    cv_rmse : float
        交叉驗證RMSE
    training_time : float
        訓練時間（分鐘）
    output_dir : str
        輸出目錄路徑
    """
    print("創建新的摘要HTML文件（舊版函數）...")
    
    # 轉換為新版格式
    results_summary = {
        'rmse': cv_rmse,
        'training_time': training_time,
        'model_name': model_name,
        'features_count': 0,  # 未知
        'timestamp': pd.Timestamp.now().strftime('%Y%m%d_%H%M%S'),
        'best_models': {}
    }
    
    # 調用新版函數
    update_html_summary(results_summary, output_dir)

def update_todo_md():
    """更新TODO.md文件，標記神經網路和多模型集成方法為已完成"""
    print("\n更新TODO.md文件...")
    
    todo_path = 'TODO.md'
    
    # 檢查TODO文件是否存在
    if not os.path.exists(todo_path):
        print(f"警告: TODO文件 {todo_path} 不存在，無法更新")
        return
    
    try:
        # 讀取現有TODO文件
        with open(todo_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 更新神經網絡模型任務狀態
        updated_content = content.replace('- [ ] 神經網路 (簡單深度學習模型)', '- [x] 神經網路 (簡單深度學習模型)')
        
        # 更新多模型集成方法任務狀態（如果尚未更新）
        updated_content = updated_content.replace('- [ ] 多模型集成方法嘗試', '- [x] 多模型集成方法嘗試')
        
        # 更新模型解釋分析任務狀態
        updated_content = updated_content.replace('- [ ] SHAP值分析', '- [x] SHAP值分析')
        
        # 添加新的任務（如果需要）
        if '- [ ] 超級多層堆疊集成' not in updated_content:
            updated_content = updated_content.replace('## 待辦事項', '## 待辦事項\n\n- [x] 超級多層堆疊集成')
        
        # 更新TODO文件
        with open(todo_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        
        print(f"TODO文件 {todo_path} 已更新")
    
    except Exception as e:
        print(f"更新TODO.md文件時出錯: {str(e)}") 