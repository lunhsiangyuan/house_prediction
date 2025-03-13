"""
環境設置腳本
準備必要的目錄和文件結構
"""

import os
import sys

def setup_directories():
    """創建必要的目錄結構"""
    print("創建目錄結構...")
    
    # 主要目錄
    dirs = [
        'dataset',
        'feature_engineering_results',
        'feature_engineering_results/advanced',
        'feature_engineering_results/advanced_nonlinear',
        'model_results',
        'preprocessing_results'
    ]
    
    # 創建目錄
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
        print(f"  目錄已就緒: {directory}")
    
    print("目錄結構創建完成")

def check_dependencies():
    """檢查必要的Python包依賴"""
    print("檢查Python包依賴...")
    
    required_packages = [
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'xgboost',
        'lightgbm'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  {package} ... 已安裝")
        except ImportError:
            missing_packages.append(package)
            print(f"  {package} ... 未安裝")
    
    if missing_packages:
        print("\n缺少以下Python包，請安裝:")
        print("pip install " + " ".join(missing_packages))
        return False
    
    print("所有必要的Python包已安裝")
    return True

def create_dummy_data():
    """創建測試數據，僅用於演示"""
    import numpy as np
    import pandas as pd
    
    print("創建示例數據文件...")
    
    # 檢查是否已存在數據文件
    if os.path.exists('dataset/train.csv') and os.path.exists('dataset/test.csv'):
        print("  數據文件已存在，跳過創建")
        return
    
    # 創建訓練集
    np.random.seed(42)
    n_train = 1460
    n_test = 1459
    n_features = 80
    
    # 訓練數據
    train_features = np.random.randn(n_train, n_features)
    train_price = np.exp(np.random.randn(n_train) * 0.5 + 12)  # 創建偏斜的價格分布
    
    train_df = pd.DataFrame(train_features, columns=[f'Feature_{i}' for i in range(n_features)])
    train_df['Id'] = range(1, n_train + 1)
    train_df['SalePrice'] = train_price
    
    # 測試數據
    test_features = np.random.randn(n_test, n_features)
    test_df = pd.DataFrame(test_features, columns=[f'Feature_{i}' for i in range(n_features)])
    test_df['Id'] = range(1, n_test + 1)
    
    # 保存數據文件
    os.makedirs('dataset', exist_ok=True)
    train_df.to_csv('dataset/train.csv', index=False)
    test_df.to_csv('dataset/test.csv', index=False)
    
    print("  示例數據文件已創建")
    
    # 創建高級特徵工程結果
    adv_train_features = np.random.randn(n_train, 224)  # 224個特徵
    adv_test_features = np.random.randn(n_test, 224)
    
    adv_train_df = pd.DataFrame(adv_train_features, columns=[f'Feature_{i}' for i in range(224)])
    adv_test_df = pd.DataFrame(adv_test_features, columns=[f'Feature_{i}' for i in range(224)])
    
    # 保存高級特徵文件
    os.makedirs('feature_engineering_results/advanced_nonlinear', exist_ok=True)
    adv_train_df.to_csv('feature_engineering_results/advanced_nonlinear/train_features.csv', index=False)
    adv_test_df.to_csv('feature_engineering_results/advanced_nonlinear/test_features.csv', index=False)
    
    print("  高級特徵工程示例數據已創建")

def create_todo_md():
    """創建TODO.md文件"""
    print("創建TODO.md文件...")
    
    if os.path.exists('TODO.md'):
        print("  TODO.md文件已存在，跳過創建")
        return
    
    todo_content = """# 房價預測專案待辦事項清單

## 待辦事項

- [ ] 數據探索性分析
- [ ] 數據預處理
- [ ] 基本特徵工程
- [ ] 高級特徵工程
- [ ] 基礎模型訓練
- [ ] 優化模型參數
- [ ] 多模型集成方法嘗試
- [ ] 神經網路 (簡單深度學習模型)
- [ ] SHAP值分析
- [ ] 創建最終提交文件

## 已完成

- [x] 項目初始化
- [x] 環境配置
"""
    
    with open('TODO.md', 'w', encoding='utf-8') as f:
        f.write(todo_content)
    
    print("  TODO.md文件已創建")

def main():
    """主函數"""
    print("====== 環境設置 ======")
    
    # 創建目錄結構
    setup_directories()
    
    # 檢查依賴
    dependencies_ok = check_dependencies()
    
    if not dependencies_ok:
        print("\n警告: 缺少必要的Python包依賴，請先安裝")
    
    # 創建示例數據
    create_dummy_data()
    
    # 創建TODO.md文件
    create_todo_md()
    
    print("\n環境設置完成！請繼續執行以下腳本:")
    print("  python main_ensemble.py")

if __name__ == "__main__":
    main() 