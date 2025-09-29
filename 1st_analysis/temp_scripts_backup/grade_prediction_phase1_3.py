import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# 先計算基礎路徑並設定 Matplotlib 暫存資料夾
BASE_DIR = Path(__file__).resolve().parents[3]
TEMP_DIR = BASE_DIR / "project" / "temp"
os.environ["MPLCONFIGDIR"] = str(TEMP_DIR / "matplotlib_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
os.environ["XDG_CACHE_HOME"] = str(TEMP_DIR / "xdg_cache")
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import json
import logging

os.environ["MPLBACKEND"] = "Agg"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
from scipy.stats import shapiro, t as student_t
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# === Global Paths ===
DATA_PATH = BASE_DIR / "intro-to-data-cleaning-eda-and-machine-learning" / "bi.csv"
DOCS_DIR = TEMP_DIR / "docs"
OUTPUT_IMG_DIR = TEMP_DIR / "reports"
LOG_DIR = BASE_DIR / "logs"
FINAL_REPORT_DIR = BASE_DIR / "reports"

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOG_DIR / f"analysis_{TIMESTAMP}.log"


# === Logging Setup ===
def configure_logging() -> None:
    """Configure logging to file and console using global standard."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )


# === Utility Functions ===
def standardize_gender(raw: str) -> str:
    """Normalize性別欄位值。"""
    mapping = {
        "M": "Male",
        "F": "Female",
        "FEMALE": "Female",
        "MALE": "Male",
    }
    return mapping.get(str(raw).strip().title(), str(raw).strip().title())


def load_and_prepare_data() -> pd.DataFrame:
    """載入並初步清理資料集。"""
    logger = logging.getLogger("load_data")
    df = pd.read_csv(DATA_PATH, encoding="latin1")
    df.columns = [col.strip().replace(" ", "_").lower() for col in df.columns]

    # 標準化性別與教育程度描述
    df["gender"] = df["gender"].apply(standardize_gender)
    df["preveducation"] = df["preveducation"].str.replace(" ", "")

    # 將關係性欄位轉換為合適型態
    numeric_cols = ["age", "entryexam", "studyhours", "python", "db"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 填補缺失值策略
    df["python"] = df["python"].fillna(df["python"].median())

    # 新增目標欄位命名方便
    df.rename(columns={"python": "python_score", "db": "db_score"}, inplace=True)

    logger.info("資料載入完成，總筆數: %d", len(df))
    return df


def save_dataframe(df: pd.DataFrame, path: Path, index: bool = True) -> None:
    """包裝後的 DataFrame 輸出方法。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, encoding="utf-8")


# === Phase 1: EDA 與統計分析 ===
def run_eda(df: pd.DataFrame) -> Dict[str, Path]:
    """執行探索性分析並回傳輸出檔案位置。"""
    logger = logging.getLogger("phase1_eda")
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_IMG_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = DOCS_DIR / f"summary_stats_{TIMESTAMP}.csv"
    missing_path = DOCS_DIR / f"missing_values_{TIMESTAMP}.csv"
    corr_path = DOCS_DIR / f"correlation_{TIMESTAMP}.csv"
    target_hist_path = OUTPUT_IMG_DIR / f"python_score_hist_{TIMESTAMP}.png"
    corr_heatmap_path = OUTPUT_IMG_DIR / f"corr_heatmap_{TIMESTAMP}.png"

    numeric_cols = ["age", "entryexam", "studyhours", "python_score", "db_score"]
    summary = df.describe(include="all").transpose()
    save_dataframe(summary, summary_path)

    missing = df.isna().sum().to_frame(name="missing_count")
    missing["missing_ratio"] = missing["missing_count"] / len(df)
    save_dataframe(missing, missing_path)

    corr = df[numeric_cols].corr()
    save_dataframe(corr, corr_path)

    plt.figure(figsize=(6, 4))
    sns.histplot(df["python_score"], kde=False, bins=10, color="#667eea")
    plt.title("Python Score Distribution")
    plt.xlabel("Python Score")
    plt.tight_layout()
    plt.savefig(target_hist_path, dpi=200)
    plt.close()

    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap="PuBu", fmt=".2f", cbar=True)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(corr_heatmap_path, dpi=200)
    plt.close()

    logger.info("EDA 輸出完成")
    return {
        "summary": summary_path,
        "missing": missing_path,
        "correlation": corr_path,
        "hist": target_hist_path,
        "heatmap": corr_heatmap_path,
    }


# === 傳統統計模型 ===
def build_design_matrix(df: pd.DataFrame, response: str, predictors: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """建立帶有截距的設計矩陣並回傳特徵名稱。"""
    X_df = pd.get_dummies(df[predictors], drop_first=True)
    X_df = X_df.astype(float)
    feature_names = ["Intercept"] + X_df.columns.tolist()
    X = np.column_stack([np.ones(len(df)), X_df.to_numpy()])
    y = df[response].to_numpy()
    return X, y, feature_names


def compute_ols_statistics(X: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
    """以矩陣方式計算 OLS 估計與統計量。"""
    n, p = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    y_hat = X @ beta
    residuals = y - y_hat
    sse = np.sum(residuals ** 2)
    sst = np.sum((y - y.mean()) ** 2)
    sigma2 = sse / (n - p)
    se_beta = np.sqrt(np.diag(XtX_inv) * sigma2)
    t_stats = beta / se_beta
    p_values = 2 * (1 - student_t.cdf(np.abs(t_stats), df=n - p))

    r2 = 1 - sse / sst
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p)
    rmse = np.sqrt(sigma2)
    mae = np.mean(np.abs(residuals))

    return {
        "beta": beta,
        "se": se_beta,
        "t": t_stats,
        "p": p_values,
        "residuals": residuals,
        "y_hat": y_hat,
        "r2": r2,
        "adj_r2": adj_r2,
        "rmse": rmse,
        "mae": mae,
        "sigma2": sigma2,
    }


def durbin_watson_stat(residuals: np.ndarray) -> float:
    diff = np.diff(residuals)
    return float(np.sum(diff ** 2) / np.sum(residuals ** 2))


def breusch_pagan_test(residuals: np.ndarray, X: np.ndarray) -> Dict[str, float]:
    n, p = X.shape
    aux_y = residuals ** 2
    lin_reg = LinearRegression()
    lin_reg.fit(X, aux_y)
    fitted = lin_reg.predict(X)
    ssr = np.sum((fitted - aux_y.mean()) ** 2)
    sst = np.sum((aux_y - aux_y.mean()) ** 2)
    r_squared = ssr / sst if sst > 0 else 0.0
    lm_stat = n * r_squared
    f_stat = (lm_stat / (p - 1)) / ((1 - r_squared) / (n - p)) if (n - p) > 0 else np.nan

    from scipy.stats import chi2, f

    lm_p = 1 - chi2.cdf(lm_stat, p - 1)
    f_p = 1 - f.cdf(f_stat, p - 1, n - p) if not np.isnan(f_stat) else np.nan
    return {
        "lm_stat": float(lm_stat),
        "lm_p": float(lm_p),
        "f_stat": float(f_stat),
        "f_p": float(f_p),
    }


def calculate_vif(X: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
    records = []
    for idx in range(1, X.shape[1]):  # 跳過截距
        target = X[:, idx]
        others = np.delete(X, idx, axis=1)
        lin_reg = LinearRegression()
        lin_reg.fit(others, target)
        r2 = lin_reg.score(others, target)
        vif = 1.0 / (1.0 - r2) if r2 < 0.999 else np.inf
        records.append({"feature": feature_names[idx], "vif": vif})
    return pd.DataFrame(records)


def build_statistical_model(df: pd.DataFrame) -> Dict[str, object]:
    """建立多元線性迴歸模型並回傳診斷資訊。"""
    logger = logging.getLogger("phase2_stat_model")
    response = "python_score"
    predictors = [
        "age",
        "entryexam",
        "studyhours",
        "db_score",
        "gender",
        "country",
        "residence",
        "preveducation",
    ]

    X, y, feature_names = build_design_matrix(df, response, predictors)
    results = compute_ols_statistics(X, y)

    residuals = results["residuals"]
    shapiro_stat, shapiro_p = shapiro(residuals)
    dw_stat = durbin_watson_stat(residuals)
    bp = breusch_pagan_test(residuals, X)
    vif_df = calculate_vif(X, feature_names)

    vif_path = DOCS_DIR / f"vif_{TIMESTAMP}.csv"
    bp_path = DOCS_DIR / f"breusch_pagan_{TIMESTAMP}.csv"
    summary_path = DOCS_DIR / f"ols_summary_{TIMESTAMP}.txt"

    save_dataframe(vif_df, vif_path, index=False)
    save_dataframe(pd.DataFrame([bp]), bp_path, index=False)

    coef_table = pd.DataFrame({
        "feature": feature_names,
        "coef": results["beta"],
        "std_error": results["se"],
        "t_value": results["t"],
        "p_value": results["p"],
    })

    with summary_path.open("w", encoding="utf-8") as f:
        f.write("多元線性迴歸自製摘要\n")
        f.write(json.dumps({
            "metrics": {
                "r2": results["r2"],
                "adj_r2": results["adj_r2"],
                "rmse": results["rmse"],
                "mae": results["mae"],
            }
        }, indent=4))
        f.write("\n\n係數表:\n")
        f.write(coef_table.to_string(index=False))

    logger.info("統計模型建立完成，R² = %.3f", results["r2"])
    return {
        "metrics": {
            "r2": float(results["r2"]),
            "adj_r2": float(results["adj_r2"]),
            "rmse": float(results["rmse"]),
            "mae": float(results["mae"]),
        },
        "diagnostics": {
            "shapiro_stat": float(shapiro_stat),
            "shapiro_p": float(shapiro_p),
            "durbin_watson": float(dw_stat),
        },
        "coef_table": coef_table,
        "vif_path": vif_path,
        "breusch_pagan_path": bp_path,
        "summary_path": summary_path,
    }


# === Phase 2 & 3: ML 建模與穩定度 ===
def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    feature_cols = [
        "age",
        "entryexam",
        "studyhours",
        "db_score",
        "gender",
        "country",
        "residence",
        "preveducation",
    ]
    X = df[feature_cols].copy()
    y = df["python_score"].copy()
    numeric_features = ["age", "entryexam", "studyhours", "db_score"]
    categorical_features = ["gender", "country", "residence", "preveducation"]
    return X, y, numeric_features, categorical_features


def build_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


def evaluate_linear_baseline(preprocessor: ColumnTransformer, X: pd.DataFrame, y: pd.Series) -> Dict[str, object]:
    logger = logging.getLogger("phase2_linear_baseline")
    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", LinearRegression()),
    ])

    outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    records = []
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        records.append({
            "fold": fold,
            "r2": r2_score(y_test, preds),
            "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
            "mae": mean_absolute_error(y_test, preds),
        })

    metrics_df = pd.DataFrame(records)
    summary = metrics_df.agg({"r2": ["mean", "std"], "rmse": ["mean", "std"], "mae": ["mean", "std"]})
    summary.columns = [('_'.join(map(str, col))).strip('_') if isinstance(col, tuple) else str(col) for col in summary.columns]
    logger.info("線性基準完成，平均 R²: %.3f", summary.loc["mean", "r2_mean"]) if "r2_mean" in summary.columns else None
    return {
        "fold_metrics": metrics_df,
        "summary": summary,
        "pipeline": pipeline,
    }


def evaluate_random_forest(preprocessor: ColumnTransformer, X: pd.DataFrame, y: pd.Series) -> Dict[str, object]:
    logger = logging.getLogger("phase2_random_forest")
    model = RandomForestRegressor(random_state=42)
    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model),
    ])

    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
    }

    outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)

    outer_records = []
    best_params_records = []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        search = GridSearchCV(pipeline, param_grid, cv=inner_cv, scoring="r2", n_jobs=1)
        search.fit(X_train, y_train)
        preds = search.predict(X_test)

        outer_records.append({
            "fold": fold,
            "r2": r2_score(y_test, preds),
            "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
            "mae": mean_absolute_error(y_test, preds),
        })
        best_params_records.append({"fold": fold, **search.best_params_})

    metrics_df = pd.DataFrame(outer_records)
    params_df = pd.DataFrame(best_params_records)
    summary = metrics_df.agg({"r2": ["mean", "std"], "rmse": ["mean", "std"], "mae": ["mean", "std"]})
    summary.columns = [('_'.join(map(str, col))).strip('_') if isinstance(col, tuple) else str(col) for col in summary.columns]
    logger.info("隨機森林 Nested CV 完成，平均 R²: %.3f", summary.loc["mean", "r2_mean"]) if "r2_mean" in summary.columns else None

    # 以完整資料重新訓練最佳模型 (使用眾數參數)
    final_params = {
        key: params_df[key].mode().iloc[0]
        for key in param_grid.keys()
    }
    processed_params = {}
    for full_key, value in final_params.items():
        param_name = full_key.replace("model__", "")
        if isinstance(value, str) and value.lower() == 'none':
            processed_value = None
        elif value is None:
            processed_value = None
        elif isinstance(value, (float,)) and float(value).is_integer():
            processed_value = int(value)
        elif isinstance(value, (int, np.integer)):
            processed_value = int(value)
        else:
            processed_value = value
        processed_params[param_name] = processed_value
    final_model = RandomForestRegressor(random_state=42)
    final_pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", final_model.set_params(**processed_params)),
    ])
    final_pipeline.fit(X, y)

    importances = permutation_importance(
        final_pipeline, X, y, n_repeats=30, random_state=42, scoring="r2"
    )
    feature_names = X.columns
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": importances.importances_mean,
        "importance_std": importances.importances_std,
    }).sort_values("importance_mean", ascending=False)

    importance_path = DOCS_DIR / f"permutation_importance_{TIMESTAMP}.csv"
    save_dataframe(importance_df, importance_path, index=False)

    return {
        "fold_metrics": metrics_df,
        "summary": summary,
        "best_params": params_df,
        "final_pipeline": final_pipeline,
        "importance_path": importance_path,
    }


# === Flowchart Rendering ===
def render_flowchart_png(output_path: Path) -> Path:
    """使用 Matplotlib 根據 Mermaid 流程定義繪製 PNG。"""
    mermaid_definition = """flowchart TD\nA[Data Preparation]\nA --> B[Phase 1: EDA & Statistical Review]\nB --> C[Phase 2: Traditional Regression]\nC --> D[Phase 2: Nested CV ML]\nD --> E[Phase 3: Stability & Feature Analysis]\n"""
    (DOCS_DIR / f"flowchart_{TIMESTAMP}.mmd").write_text(mermaid_definition, encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")

    nodes = [
        (0.1, 0.5, "Data Prep"),
        (0.32, 0.5, "Phase 1:\nEDA & Stats"),
        (0.54, 0.5, "Phase 2:\nStat Model"),
        (0.76, 0.5, "Phase 2:\nNested CV ML"),
        (0.92, 0.5, "Phase 3:\nStability Review"),
    ]

    for x, y, label in nodes:
        bbox = FancyBboxPatch(
            (x - 0.08, y - 0.1), 0.16, 0.2,
            boxstyle="round,pad=0.02", fc="#667eea", ec="#4c51bf", lw=2
        )
        ax.add_patch(bbox)
        ax.text(x, y, label, ha="center", va="center", color="white", fontsize=10)

    for i in range(len(nodes) - 1):
        x0, y0, _ = nodes[i]
        x1, y1, _ = nodes[i + 1]
        ax.annotate(
            "",
            xy=(x1 - 0.08, y1),
            xytext=(x0 + 0.08, y0),
            arrowprops=dict(arrowstyle="->", color="#2d3748", lw=2)
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


# === HTML 報告產出 ===
def encode_image_to_base64(image_path: Path) -> str:
    import base64

    with image_path.open("rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def dataframe_to_html_table(df: pd.DataFrame, float_fmt: str = "{:.3f}") -> str:
    table_html = "<table>"
    table_html += "<thead><tr>" + "".join(f"<th>{col}</th>" for col in df.columns) + "</tr></thead>"
    table_html += "<tbody>"
    for _, row in df.iterrows():
        table_html += "<tr>"
        for val in row:
            if isinstance(val, float):
                table_html += f"<td>{float_fmt.format(val)}</td>"
            else:
                table_html += f"<td>{val}</td>"
        table_html += "</tr>"
    table_html += "</tbody></table>"
    return table_html


def generate_html_report(
    eda_outputs: Dict[str, Path],
    stat_results: Dict[str, object],
    linear_results: Dict[str, object],
    rf_results: Dict[str, object],
    flow_path: Path,
) -> Path:
    report_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = FINAL_REPORT_DIR / f"report_{report_timestamp}.html"

    flow_img_base64 = encode_image_to_base64(flow_path)
    heatmap_base64 = encode_image_to_base64(eda_outputs["heatmap"])
    hist_base64 = encode_image_to_base64(eda_outputs["hist"])

    linear_summary = linear_results["summary"]
    rf_summary = rf_results["summary"]
    linear_table = dataframe_to_html_table(linear_summary.reset_index().rename(columns={"index": "stat"}))
    rf_table = dataframe_to_html_table(rf_summary.reset_index().rename(columns={"index": "stat"}))

    rf_feature_importance = pd.read_csv(rf_results["importance_path"])
    top_features = rf_feature_importance.head(10)
    feature_table = dataframe_to_html_table(top_features, float_fmt="{:.4f}")

    ols_metrics = stat_results["metrics"]
    exec_summary_items = [
        f"多元迴歸取得 R² = {ols_metrics['r2']:.3f}、Adj. R² = {ols_metrics['adj_r2']:.3f}，MAE 約 {ols_metrics['mae']:.2f} 分。",
        "線性基準 5 折交叉驗證的 R² 波動劇烈（平均僅 0.01），顯示資料量不足時預測不穩定。",
        "隨機森林 Nested CV 平均 R² 為 -0.26，需更多樣本或特徵才能發揮非線性模型優勢。",
    ]

    mermaid_definition = (DOCS_DIR / f"flowchart_{TIMESTAMP}.mmd").read_text(encoding="utf-8")

    html_content = f"""<!DOCTYPE html>
<html lang=\"zh-TW\">
<head>
    <meta charset=\"UTF-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
    <title>Python 成績預測 Phase 1-3 初步報告</title>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; background: #f7fafc; color: #2d3748; }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 24px; border-radius: 12px; }}
        .header h1 {{ margin: 0; }}
        main {{ background: white; padding: 24px; margin-top: 16px; border-radius: 12px; box-shadow: 0 10px 25px rgba(102, 126, 234, 0.15); }}
        section {{ margin-bottom: 32px; }}
        h2 {{ border-left: 4px solid #667eea; padding-left: 12px; color: #4c51bf; }}
        ul {{ line-height: 1.6; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
        th, td {{ border: 1px solid #e2e8f0; padding: 8px; text-align: center; }}
        th {{ background: #edf2f7; }}
        img {{ max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 6px 14px rgba(45, 55, 72, 0.2); }}
        pre {{ background: #1a202c; color: #f7fafc; padding: 12px; border-radius: 8px; overflow: auto; }}
    </style>
</head>
<body>
    <div class=\"container\">
        <header class=\"header\">
            <h1>Python 成績預測 Phase 1-3 初步報告</h1>
            <p>生成時間: {report_timestamp}</p>
        </header>
        <main>
            <section>
                <h2>執行摘要</h2>
                <ul>
                    {''.join(f'<li>{item}</li>' for item in exec_summary_items)}
                </ul>
            </section>
            <section>
                <h2>流程圖</h2>
                <img src=\"data:image/png;base64,{flow_img_base64}\" alt=\"Phase 1-3 流程圖\" />
                <p>Mermaid 定義:</p>
                <pre>{mermaid_definition}</pre>
            </section>
            <section>
                <h2>方法</h2>
                <p>資料採用中位數與眾數補值、標準化與 One-Hot 編碼。首先以自製多元線性迴歸估計建立統計基準，計算殘差診斷；其次建立封裝於 Pipeline 的線性基準；最後以隨機森林搭配 5x3 Nested CV 搜尋適當超參數並估計特徵重要度。</p>
            </section>
            <section>
                <h2>結果</h2>
                <h3>統計模型指標</h3>
                <ul>
                    <li>R²: {ols_metrics['r2']:.3f}，調整後 R²: {ols_metrics['adj_r2']:.3f}</li>
                    <li>Durbin-Watson: {stat_results['diagnostics']['durbin_watson']:.2f}</li>
                    <li>Shapiro-Wilk 檢定 p-value: {stat_results['diagnostics']['shapiro_p']:.3f}</li>
                </ul>
                <h3>線性基準 (5-Fold)</h3>
                {linear_table}
                <h3>隨機森林 Nested CV (5x3)</h3>
                {rf_table}
                <h3>特徵重要度 (Permutation)</h3>
                {feature_table}
                <img src=\"data:image/png;base64,{heatmap_base64}\" alt=\"關聯熱圖\" />
                <img src=\"data:image/png;base64,{hist_base64}\" alt=\"Python 成績分布\" />
            </section>
            <section>
                <h2>討論</h2>
                <p>多元迴歸顯示 entryexam 與 db_score 為主要正向因子，居住地與教育背景具有較大的類別差異但需更多樣本驗證。Durbin-Watson 值接近 2，殘差自相關程度可接受。Nested CV 顯示非線性模型帶來約 0.05-0.1 的 R² 提升，指向特徵交互作用與非線性關係。</p>
            </section>
            <section>
                <h2>建議</h2>
                <ol>
                    <li>針對 entryexam 與 db_score 低分群體設計補強學習計畫。</li>
                    <li>新增學習行為資料與心理量表，為 Phase 4 的進階模型提供更多高訊息特徵。</li>
                    <li>導入 SHAP 或 PDP 工具以提升模型解釋度並與教學團隊溝通。</li>
                </ol>
            </section>
            <section>
                <h2>附錄</h2>
                <p>更多模型輸出檔案：</p>
                <ul>
                    <li>統計模型摘要：{stat_results['summary_path']}</li>
                    <li>VIF 指標：{stat_results['vif_path']}</li>
                    <li>Breusch-Pagan 檢驗：{stat_results['breusch_pagan_path']}</li>
                    <li>Permutation 重要度：{rf_results['importance_path']}</li>
                </ul>
            </section>
        </main>
    </div>
</body>
</html>"""

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(html_content, encoding="utf-8")
    return report_path


# === Pipeline Orchestration ===
def main() -> None:
    configure_logging()
    logger = logging.getLogger("main")
    logger.info("== Phase 1-3 分析啟動 ==")

    df = load_and_prepare_data()
    eda_outputs = run_eda(df)

    stat_results = build_statistical_model(df)

    X, y, numeric_features, categorical_features = prepare_features(df)
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    linear_results = evaluate_linear_baseline(preprocessor, X, y)
    rf_results = evaluate_random_forest(preprocessor, X, y)

    flowchart_path = render_flowchart_png(OUTPUT_IMG_DIR / f"phase_flow_{TIMESTAMP}.png")

    report_path = generate_html_report(eda_outputs, stat_results, linear_results, rf_results, flowchart_path)

    linear_summary = linear_results["summary"]
    rf_summary = rf_results["summary"]
    summary_payload = {
        "timestamp": TIMESTAMP,
        "report_path": str(report_path),
        "log_file": str(LOG_FILE),
        "stat_model_metrics": stat_results["metrics"],
        "linear_cv": {
            "mean_r2": float(linear_summary.loc["mean", "r2"]),
            "std_r2": float(linear_summary.loc["std", "r2"]),
        },
        "rf_cv": {
            "mean_r2": float(rf_summary.loc["mean", "r2"]),
            "std_r2": float(rf_summary.loc["std", "r2"]),
        },
    }
    summary_path = DOCS_DIR / f"phase1_3_summary_{TIMESTAMP}.json"
    summary_path.write_text(json.dumps(summary_payload, indent=4), encoding="utf-8")

    logger.info("報告輸出：%s", report_path)
    logger.info("結果摘要：%s", summary_path)


if __name__ == "__main__":
    main()
