"""
模型定義模塊
包含各類型模型的定義和初始化
"""

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import lightgbm as lgb

def get_base_models():
    """獲取基礎模型列表"""
    models = {
        # 線性模型
        'ridge': Ridge(alpha=0.5, random_state=42),
        'lasso': Lasso(alpha=0.0005, random_state=42),
        'elasticnet': ElasticNet(alpha=0.0005, l1_ratio=0.5, random_state=42),
        
        # 樹模型
        'rf': RandomForestRegressor(
            n_estimators=150, 
            max_depth=15, 
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42
        ),
        'gbm': GradientBoostingRegressor(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        ),
        'xgb': xgb.XGBRegressor(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            colsample_bylevel=0.8,
            gamma=0.1,
            min_child_weight=1,
            reg_alpha=0.01,
            reg_lambda=1,
            random_state=42
        ),
        'lgb': lgb.LGBMRegressor(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.01,
            reg_lambda=1,
            random_state=42
        ),
        
        # 其他模型
        'svr': SVR(
            kernel='rbf', 
            C=10, 
            epsilon=0.1, 
            gamma='scale'
        ),
        'knn': KNeighborsRegressor(
            n_neighbors=8,
            weights='distance',
            p=1  # Manhattan distance
        )
    }
    
    return models

def get_meta_model():
    """獲取元模型"""
    return xgb.XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        colsample_bylevel=0.8,
        gamma=0.1,
        min_child_weight=1,
        reg_alpha=0.01,
        reg_lambda=1,
        random_state=42
    )

def get_final_meta_model():
    """獲取最終元模型"""
    return lgb.LGBMRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.01,
        reg_lambda=1,
        random_state=42
    ) 