import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Configure working directories and Matplotlib cache before heavy imports
BASE_DIR = Path(__file__).resolve().parents[3]
TEMP_DIR = BASE_DIR / "project" / "temp"
os.environ["MPLCONFIGDIR"] = str(TEMP_DIR / "matplotlib_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
os.environ["XDG_CACHE_HOME"] = str(TEMP_DIR / "xdg_cache")
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
os.environ["MPLBACKEND"] = "Agg"

import json
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
from scipy.stats import shapiro, t as student_t
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
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


def configure_logging() -> None:
    """Configure logging for both file and console outputs."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )


def standardize_gender(raw: str) -> str:
    """Normalize gender values to 'Male' or 'Female'."""
    mapping = {
        "M": "Male",
        "F": "Female",
        "FEMALE": "Female",
        "MALE": "Male",
    }
    return mapping.get(str(raw).strip().title(), str(raw).strip().title())


def load_and_prepare_data() -> pd.DataFrame:
    """Load dataset and perform minimal cleaning suitable for statistical modelling."""
    logger = logging.getLogger("load_data")
    df = pd.read_csv(DATA_PATH, encoding="latin1")
    df.columns = [col.strip().replace(" ", "_").lower() for col in df.columns]

    df["gender"] = df["gender"].apply(standardize_gender)
    df["preveducation"] = df["preveducation"].str.replace(" ", "")

    numeric_cols = ["age", "entryexam", "studyhours", "python", "db"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["python"] = df["python"].fillna(df["python"].median())
    df.rename(columns={"python": "python_score", "db": "db_score"}, inplace=True)

    logger.info("資料載入完成，總筆數: %d", len(df))
    return df


def save_dataframe(df: pd.DataFrame, path: Path, index: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, encoding="utf-8")


def run_basic_eda(df: pd.DataFrame) -> Dict[str, Path]:
    """Generate descriptive statistics, missingness summary, and key visualisations."""
    logger = logging.getLogger("eda")
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_IMG_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = DOCS_DIR / f"summary_stats_{TIMESTAMP}.csv"
    missing_path = DOCS_DIR / f"missing_values_{TIMESTAMP}.csv"
    corr_path = DOCS_DIR / f"correlation_{TIMESTAMP}.csv"
    hist_path = OUTPUT_IMG_DIR / f"python_score_hist_{TIMESTAMP}.png"
    corr_fig_path = OUTPUT_IMG_DIR / f"corr_heatmap_{TIMESTAMP}.png"

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
    plt.savefig(hist_path, dpi=200)
    plt.close()

    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap="PuBu", fmt=".2f", cbar=True)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(corr_fig_path, dpi=200)
    plt.close()

    logger.info("EDA 輸出完成")
    return {
        "summary": summary_path,
        "missing": missing_path,
        "correlation": corr_path,
        "hist": hist_path,
        "heatmap": corr_fig_path,
    }


def build_design_matrix(df: pd.DataFrame, response: str, predictors: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    X_df = pd.get_dummies(df[predictors], drop_first=True).astype(float)
    feature_names = ["Intercept"] + X_df.columns.tolist()
    X = np.column_stack([np.ones(len(df)), X_df.to_numpy()])
    y = df[response].to_numpy()
    return X, y, feature_names


def compute_ols_statistics(X: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
    n, p = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    y_hat = X @ beta
    residuals = y - y_hat
    sse = np.sum(residuals ** 2)
    sst = np.sum((y - y.mean()) ** 2)
    sigma2 = sse / (n - p)

    se_beta = np.sqrt(np.diag(XtX_inv) * sigma2)
    t_stats = np.divide(beta, se_beta, out=np.zeros_like(beta), where=se_beta != 0)
    p_values = 2 * (1 - student_t.cdf(np.abs(t_stats), df=n - p))

    r2 = 1 - sse / sst if sst > 0 else 0.0
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p) if n > p else r2
    rmse = float(np.sqrt(sigma2))
    mae = float(np.mean(np.abs(residuals)))

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
    }


def durbin_watson_stat(residuals: np.ndarray) -> float:
    diff = np.diff(residuals)
    denominator = np.sum(residuals ** 2)
    return float(np.sum(diff ** 2) / denominator) if denominator != 0 else float("nan")


def breusch_pagan_test(residuals: np.ndarray, X: np.ndarray) -> Dict[str, float]:
    n, p = X.shape
    aux_y = residuals ** 2
    reg = LinearRegression()
    reg.fit(X, aux_y)
    fitted = reg.predict(X)
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
    for idx in range(1, X.shape[1]):  # skip intercept
        target = X[:, idx]
        others = np.delete(X, idx, axis=1)
        reg = LinearRegression()
        reg.fit(others, target)
        r2 = reg.score(others, target)
        vif = 1.0 / (1.0 - r2) if r2 < 0.999 else np.inf
        records.append({"feature": feature_names[idx], "vif": vif})
    return pd.DataFrame(records)


def run_ols(df: pd.DataFrame) -> Dict[str, object]:
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
    X, y, feature_names = build_design_matrix(df, "python_score", predictors)
    stats = compute_ols_statistics(X, y)
    residuals = stats["residuals"]

    shapiro_stat, shapiro_p = shapiro(residuals)
    dw = durbin_watson_stat(residuals)
    bp = breusch_pagan_test(residuals, X)
    vif_df = calculate_vif(X, feature_names)

    coef_table = pd.DataFrame({
        "feature": feature_names,
        "coef": stats["beta"],
        "std_error": stats["se"],
        "t_value": stats["t"],
        "p_value": stats["p"],
    })

    vif_path = DOCS_DIR / f"vif_{TIMESTAMP}.csv"
    bp_path = DOCS_DIR / f"breusch_pagan_{TIMESTAMP}.csv"
    coef_path = DOCS_DIR / f"ols_coefficients_{TIMESTAMP}.csv"
    save_dataframe(vif_df, vif_path, index=False)
    save_dataframe(pd.DataFrame([bp]), bp_path, index=False)
    save_dataframe(coef_table, coef_path, index=False)

    return {
        "metrics": {
            "r2": float(stats["r2"]),
            "adj_r2": float(stats["adj_r2"]),
            "rmse": stats["rmse"],
            "mae": stats["mae"],
        },
        "diagnostics": {
            "shapiro_stat": float(shapiro_stat),
            "shapiro_p": float(shapiro_p),
            "durbin_watson": float(dw),
        },
        "bp": bp,
        "vif_path": vif_path,
        "coef_path": coef_path,
        "coef_table": coef_table,
    }


def prepare_features_for_lasso(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
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


def run_lasso(df: pd.DataFrame) -> Dict[str, object]:
    logger = logging.getLogger("lasso")
    X, y, numeric_features, categorical_features = prepare_features_for_lasso(df)

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

    lasso_model = LassoCV(cv=5, random_state=42, n_alphas=100, max_iter=10000)
    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", lasso_model),
    ])

    pipeline.fit(X, y)
    preds = pipeline.predict(X)

    r2 = r2_score(y, preds)
    rmse = float(np.sqrt(mean_squared_error(y, preds)))
    mae = float(mean_absolute_error(y, preds))

    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    coef_series = pd.Series(pipeline.named_steps["model"].coef_, index=feature_names)
    coef_df = coef_series.reset_index()
    coef_df.columns = ["feature", "coefficient"]

    sparsity = (coef_series == 0).mean()
    logger.info("Lasso 完成：最佳 alpha = %.6f", pipeline.named_steps["model"].alpha_)

    coef_path = DOCS_DIR / f"lasso_coefficients_{TIMESTAMP}.csv"
    save_dataframe(coef_df, coef_path, index=False)

    return {
        "metrics": {
            "r2": float(r2),
            "rmse": rmse,
            "mae": mae,
            "alpha": float(pipeline.named_steps["model"].alpha_),
            "zero_coef_ratio": float(sparsity),
        },
        "coefficients_path": coef_path,
        "coefficients": coef_df,
    }


def render_flowchart(output_path: Path) -> Path:
    definition = """flowchart TD\nA[Data Preparation]\nA --> B[Descriptive Review]\nB --> C[OLS Diagnostics]\nC --> D[Lasso Regularization]\n"""
    (DOCS_DIR / f"flowchart_{TIMESTAMP}.mmd").write_text(definition, encoding="utf-8")

    fig, ax = plt.subplots(figsize=(6.5, 2.5))
    ax.axis("off")

    nodes = [
        (0.12, 0.5, "Data Prep"),
        (0.37, 0.5, "Descriptive\nReview"),
        (0.62, 0.5, "OLS\nDiagnostics"),
        (0.86, 0.5, "Lasso\nRegularization"),
    ]

    for x, y, label in nodes:
        bbox = FancyBboxPatch(
            (x - 0.09, y - 0.12), 0.18, 0.24,
            boxstyle="round,pad=0.02", fc="#4f46e5", ec="#312e81", lw=2
        )
        ax.add_patch(bbox)
        ax.text(x, y, label, ha="center", va="center", color="white", fontsize=10)

    for i in range(len(nodes) - 1):
        x0, y0, _ = nodes[i]
        x1, y1, _ = nodes[i + 1]
        ax.annotate(
            "",
            xy=(x1 - 0.09, y1),
            xytext=(x0 + 0.09, y0),
            arrowprops=dict(arrowstyle="->", color="#1f2937", lw=2)
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def encode_image_base64(image_path: Path) -> str:
    import base64
    with image_path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def dataframe_to_html_table(df: pd.DataFrame, float_fmt: str = "{:.3f}") -> str:
    header = "<thead><tr>" + ''.join(f"<th>{col}</th>" for col in df.columns) + "</tr></thead>"
    body_rows = []
    for _, row in df.iterrows():
        cells = []
        for val in row:
            if isinstance(val, float):
                cells.append(f"<td>{float_fmt.format(val)}</td>")
            else:
                cells.append(f"<td>{val}</td>")
        body_rows.append("<tr>" + ''.join(cells) + "</tr>")
    body = "<tbody>" + ''.join(body_rows) + "</tbody>"
    return f"<table>{header}{body}</table>"


def generate_report(
    eda_outputs: Dict[str, Path],
    ols_results: Dict[str, object],
    lasso_results: Dict[str, object],
    flowchart_path: Path,
) -> Path:
    report_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = FINAL_REPORT_DIR / f"report_{report_timestamp}.html"

    flow_img = encode_image_base64(flowchart_path)
    hist_img = encode_image_base64(eda_outputs["hist"])
    corr_img = encode_image_base64(eda_outputs["heatmap"])

    ols_coef_table = ols_results["coef_table"].copy()
    ols_coef_table["coef"] = ols_coef_table["coef"].astype(float)
    ols_table_html = dataframe_to_html_table(ols_coef_table.head(15))

    lasso_coef = lasso_results["coefficients"].copy().sort_values("coefficient", key=np.abs, ascending=False)
    lasso_table_html = dataframe_to_html_table(lasso_coef.head(15), float_fmt="{:.4f}")

    exec_summary = [
        f"多元迴歸 R² = {ols_results['metrics']['r2']:.3f} (Adj. R² = {ols_results['metrics']['adj_r2']:.3f}), MAE ≈ {ols_results['metrics']['mae']:.2f}。",
        f"LassoCV 最佳 α = {lasso_results['metrics']['alpha']:.4f}，全樣本 R² = {lasso_results['metrics']['r2']:.3f}，MAE ≈ {lasso_results['metrics']['mae']:.2f}。",
        "Lasso 係數稀疏化約 {:.0%}，提供特徵過濾基準。".format(lasso_results['metrics']['zero_coef_ratio']),
    ]

    mermaid_definition = (DOCS_DIR / f"flowchart_{TIMESTAMP}.mmd").read_text(encoding="utf-8")

    html = f"""<!DOCTYPE html>
<html lang=\"zh-TW\">
<head>
    <meta charset=\"UTF-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
    <title>Python 成績預測：統計 + Lasso 分析</title>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; background: #f8fafc; color: #1f2937; }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%); color: white; padding: 24px; border-radius: 12px; }}
        .header h1 {{ margin: 0; }}
        main {{ background: white; padding: 24px; margin-top: 18px; border-radius: 12px; box-shadow: 0 12px 30px rgba(79, 70, 229, 0.12); }}
        section {{ margin-bottom: 32px; }}
        h2 {{ border-left: 4px solid #4f46e5; padding-left: 12px; color: #4338ca; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 0.95rem; }}
        th, td {{ border: 1px solid #e2e8f0; padding: 8px; text-align: center; }}
        th {{ background: #ede9fe; color: #4c1d95; }}
        img {{ max-width: 100%; height: auto; border-radius: 10px; box-shadow: 0 6px 16px rgba(30, 64, 175, 0.25); margin-top: 12px; }}
        pre {{ background: #111827; color: #f9fafb; padding: 12px; border-radius: 8px; overflow: auto; }}
    </style>
</head>
<body>
    <div class=\"container\">
        <header class=\"header\">
            <h1>Python 成績預測：統計 + Lasso 分析</h1>
            <p>生成時間：{report_timestamp}</p>
        </header>
        <main>
            <section>
                <h2>執行摘要</h2>
                <ul>
                    {''.join(f'<li>{item}</li>' for item in exec_summary)}
                </ul>
            </section>
            <section>
                <h2>流程圖</h2>
                <img src=\"data:image/png;base64,{flow_img}\" alt=\"流程圖\" />
                <p>Mermaid 定義：</p>
                <pre>{mermaid_definition}</pre>
            </section>
            <section>
                <h2>方法</h2>
                <p>先進行資料清理與描述統計，接著以自訂 OLS 求解獲得係數與診斷，再運用 LassoCV 為主要正規化手段。數值欄位標準化、類別欄位 One-Hot 編碼皆封裝在 Pipeline 中以避免資訊洩漏。</p>
            </section>
            <section>
                <h2>結果</h2>
                <h3>OLS 指標與診斷</h3>
                <ul>
                    <li>R²: {ols_results['metrics']['r2']:.3f}, 調整後 R²: {ols_results['metrics']['adj_r2']:.3f}</li>
                    <li>Durbin-Watson: {ols_results['diagnostics']['durbin_watson']:.2f}</li>
                    <li>Shapiro-Wilk p-value: {ols_results['diagnostics']['shapiro_p']:.3e}</li>
                </ul>
                {ols_table_html}
                <h3>LassoCV（5-fold）結果</h3>
                <ul>
                    <li>最佳 α: {lasso_results['metrics']['alpha']:.4f}</li>
                    <li>R²: {lasso_results['metrics']['r2']:.3f}, RMSE: {lasso_results['metrics']['rmse']:.2f}, MAE: {lasso_results['metrics']['mae']:.2f}</li>
                    <li>零係數比例：約 {lasso_results['metrics']['zero_coef_ratio']:.0%}</li>
                </ul>
                {lasso_table_html}
                <img src=\"data:image/png;base64,{corr_img}\" alt=\"Correlation heatmap\" />
                <img src=\"data:image/png;base64,{hist_img}\" alt=\"Python score distribution\" />
            </section>
            <section>
                <h2>討論</h2>
                <p>OLS 顯示入學考試與自修時數為主要正向因子，但殘差常態性不足，後續可評估變數轉換或強健迴歸。Lasso 提供自動化特徵收斂框架，雖然在現有樣本下 R² 較 OLS 低，但能凸顯可淘汰變數，為後續維度縮減與偏差-變異折衷提供依據。</p>
            </section>
            <section>
                <h2>建議</h2>
                <ol>
                    <li>針對 Lasso 保留的關鍵特徵（如 studyhours、entryexam、db_score）設計指標監控。</li>
                    <li>蒐集更多樣本後再次比較 Lasso 與 Ridge，並評估交叉驗證分層策略。</li>
                    <li>嘗試 Box-Cox 或 Yeo-Johnson 轉換改善殘差常態性，驗證診斷指標。</li>
                </ol>
            </section>
            <section>
                <h2>附錄</h2>
                <ul>
                    <li>OLS 係數：{ols_results['coef_path']}</li>
                    <li>VIF 指標：{ols_results['vif_path']}</li>
                    <li>Breusch-Pagan 檢驗：{DOCS_DIR / f"breusch_pagan_{TIMESTAMP}.csv"}</li>
                    <li>Lasso 係數：{lasso_results['coefficients_path']}</li>
                </ul>
            </section>
        </main>
    </div>
</body>
</html>"""

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(html, encoding="utf-8")
    return report_path


def main() -> None:
    configure_logging()
    logger = logging.getLogger("main")
    logger.info("統計 + Lasso 分析啟動")

    df = load_and_prepare_data()
    eda_outputs = run_basic_eda(df)
    ols_results = run_ols(df)
    lasso_results = run_lasso(df)
    flowchart_path = render_flowchart(OUTPUT_IMG_DIR / f"phase_flow_{TIMESTAMP}.png")

    report_path = generate_report(eda_outputs, ols_results, lasso_results, flowchart_path)

    summary_payload = {
        "timestamp": TIMESTAMP,
        "report_path": str(report_path),
        "log_file": str(LOG_FILE),
        "ols_metrics": ols_results["metrics"],
        "ols_diagnostics": ols_results["diagnostics"],
        "lasso_metrics": lasso_results["metrics"],
        "eda_outputs": {key: str(path) for key, path in eda_outputs.items()},
    }

    summary_path = DOCS_DIR / f"stat_lasso_summary_{TIMESTAMP}.json"
    summary_path.write_text(json.dumps(summary_payload, indent=4), encoding="utf-8")

    logger.info("報告輸出：%s", report_path)
    logger.info("摘要輸出：%s", summary_path)


if __name__ == "__main__":
    main()
