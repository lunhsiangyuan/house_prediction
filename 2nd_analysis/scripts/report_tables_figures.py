import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

BASE_DIR = Path(__file__).resolve().parents[3]
TEMP_DIR = BASE_DIR / "project" / "temp"
LOG_DIR = BASE_DIR / "logs"
REPORT_DIR = BASE_DIR / "reports"
DATA_PATH = BASE_DIR / "intro-to-data-cleaning-eda-and-machine-learning" / "bi.csv"

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
from scipy.stats import iqr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
DOCS_DIR = TEMP_DIR / "docs"
FIG_DIR = TEMP_DIR / "reports"
LOG_FILE = LOG_DIR / f"table_figures_{TIMESTAMP}.log"


def configure_logging() -> None:
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
    mapping = {"M": "Male", "F": "Female", "MALE": "Male", "FEMALE": "Female"}
    return mapping.get(str(raw).strip(), str(raw).strip().title())


def load_data() -> pd.DataFrame:
    logger = logging.getLogger("load")
    df = pd.read_csv(DATA_PATH, encoding="latin1")
    df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]
    df["gender"] = df["gender"].apply(standardize_gender)
    df["preveducation"] = df["preveducation"].str.replace(" ", "")
    numeric_cols = ["age", "entryexam", "studyhours", "python", "db"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["python"] = df["python"].fillna(df["python"].median())
    df.rename(columns={"python": "python_score", "db": "db_score"}, inplace=True)
    logger.info("資料筆數：%d", len(df))
    return df


def build_table1(df: pd.DataFrame) -> pd.DataFrame:
    continuous_vars = {
        "Age (years)": "age",
        "Entry exam score": "entryexam",
        "Study hours": "studyhours",
        "Python score": "python_score",
        "Database score": "db_score",
    }
    records: List[Dict[str, str]] = []
    for label, col in continuous_vars.items():
        series = df[col].dropna()
        mean = series.mean()
        std = series.std()
        med = series.median()
        spread = iqr(series)
        records.append({
            "Variable": label,
            "Statistic": "Mean ± SD",
            "Value": f"{mean:.2f} ± {std:.2f}",
        })
        records.append({
            "Variable": label,
            "Statistic": "Median (IQR)",
            "Value": f"{med:.2f} ({(med - spread/2):.2f}-{(med + spread/2):.2f})",
        })

    categorical_vars = {
        "Gender": "gender",
        "Country": "country",
        "Residence": "residence",
        "Previous education": "preveducation",
    }
    for label, col in categorical_vars.items():
        counts = df[col].fillna("Unknown").value_counts()
        for level, count in counts.items():
            pct = count / len(df) * 100
            records.append({
                "Variable": label,
                "Statistic": level,
                "Value": f"{count} ({pct:.1f}%)",
            })
    table1 = pd.DataFrame(records)
    return table1


def run_lasso(df: pd.DataFrame) -> pd.DataFrame:
    numeric_features = ["age", "entryexam", "studyhours", "db_score"]
    categorical_features = ["gender", "country", "residence", "preveducation"]
    X = df[numeric_features + categorical_features].copy()
    y = df["python_score"].copy()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
            ("cat", Pipeline([( "imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
        ]
    )

    model = LassoCV(cv=5, random_state=42, n_alphas=100, max_iter=10000)
    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
    pipeline.fit(X, y)

    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    coefs = pipeline.named_steps["model"].coef_
    df_coef = pd.DataFrame({"feature": feature_names, "coefficient": coefs})
    df_coef["abs_coefficient"] = df_coef["coefficient"].abs()
    important = df_coef[df_coef["abs_coefficient"] > 1e-6].sort_values("abs_coefficient", ascending=False)
    important.drop(columns="abs_coefficient", inplace=True)
    important.reset_index(drop=True, inplace=True)
    return important, float(model.alpha_), float(pipeline.score(X, y))


def plot_flow(fig_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 2.5))
    ax.axis("off")
    nodes = [
        (0.1, 0.5, "Dataset\nCollected"),
        (0.32, 0.5, "Data Cleaning\n& EDA"),
        (0.55, 0.5, "OLS Modelling"),
        (0.78, 0.5, "Lasso CV"),
        (0.93, 0.5, "Reporting"),
    ]
    for x, y, label in nodes:
        box = FancyBboxPatch((x - 0.09, y - 0.12), 0.18, 0.24,
                             boxstyle="round,pad=0.02", fc="#2563eb", ec="#1e3a8a", lw=2)
        ax.add_patch(box)
        ax.text(x, y, label, ha="center", va="center", color="white", fontsize=10)
    for i in range(len(nodes) - 1):
        x0, y0, _ = nodes[i]
        x1, y1, _ = nodes[i + 1]
        ax.annotate("", xy=(x1 - 0.09, y1), xytext=(x0 + 0.09, y0),
                    arrowprops=dict(arrowstyle="->", color="#111827", lw=2))
    legend_proxy = plt.Line2D([0], [0], color="#2563eb", lw=6)
    ax.legend([legend_proxy], ["Sequential workflow"], loc="lower center", bbox_to_anchor=(0.5, -0.25))
    fig.tight_layout()
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_distributions(df: pd.DataFrame, fig_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.histplot(df["python_score"], bins=10, color="#7c3aed", alpha=0.8, ax=axes[0])
    axes[0].set_title("Python Score Distribution")
    axes[0].set_xlabel("Python score")
    axes[0].set_ylabel("Count")
    axes[0].legend(["Python score"], loc="upper right")

    sns.histplot(df["entryexam"], bins=10, color="#22c55e", alpha=0.8, ax=axes[1])
    axes[1].set_title("Entry Exam Score Distribution")
    axes[1].set_xlabel("Entry exam score")
    axes[1].set_ylabel("Count")
    axes[1].legend(["Entry exam"], loc="upper right")

    fig.suptitle("Figure 2. Score Distributions", fontsize=13)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)


def plot_correlation(df: pd.DataFrame, fig_path: Path) -> None:
    corr = df[["age", "entryexam", "studyhours", "python_score", "db_score"]].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f", ax=ax)
    ax.set_title("Figure 3. Correlation Matrix")
    colorbar = ax.collections[0].colorbar
    colorbar.set_label("Pearson correlation", rotation=90)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)


def dataframe_to_html(df: pd.DataFrame) -> str:
    return df.to_html(index=False, classes="table", border=0)


def encode_image(path: Path) -> str:
    import base64
    with path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def generate_html(
    table1: pd.DataFrame,
    table2: pd.DataFrame,
    flow_path: Path,
    dist_path: Path,
    corr_path: Path,
    lasso_alpha: float,
    lasso_r2: float,
) -> Path:
    report_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORT_DIR / f"report_tables_figures_{report_timestamp}.html"

    flow_img = encode_image(flow_path)
    dist_img = encode_image(dist_path)
    corr_img = encode_image(corr_path)

    html = f"""<!DOCTYPE html>
<html lang='zh-TW'>
<head>
  <meta charset='UTF-8'>
  <meta name='viewport' content='width=device-width, initial-scale=1.0'>
  <title>Python 成績分析補充報告</title>
  <style>
    body {{ font-family: 'Segoe UI', sans-serif; background: #f9fafb; color: #111827; }}
    .container {{ max-width: 1180px; margin: 0 auto; padding: 24px; }}
    header {{ background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%); color: white; padding: 24px; border-radius: 12px; }}
    h1 {{ margin: 0; }}
    section {{ background: white; border-radius: 12px; padding: 24px; margin-top: 20px; box-shadow: 0 12px 30px rgba(37, 99, 235, 0.15); }}
    h2 {{ color: #1f2937; border-left: 4px solid #2563eb; padding-left: 12px; }}
    .table {{ width: 100%; border-collapse: collapse; margin-top: 16px; }}
    .table th, .table td {{ border: 1px solid #e5e7eb; padding: 8px; text-align: left; }}
    .table thead th {{ background: #eff6ff; color: #1d4ed8; }}
    figure {{ margin: 0; text-align: center; }}
    figure img {{ max-width: 100%; border-radius: 10px; box-shadow: 0 8px 20px rgba(76, 29, 149, 0.2); }}
    figcaption {{ margin-top: 8px; font-size: 0.95rem; color: #4b5563; }}
  </style>
</head>
<body>
  <div class='container'>
    <header>
      <h1>Python 成績分析補充報告</h1>
      <p>生成時間：{report_timestamp}</p>
    </header>
    <section>
      <h2>Table 1. 學生背景概述 (Demographics)</h2>
      {dataframe_to_html(table1)}
    </section>
    <section>
      <h2>Figure 1. Study Flow Design</h2>
      <figure>
        <img src='data:image/png;base64,{flow_img}' alt='Study flow design' />
        <figcaption>Legend: 自左至右為資料蒐集、清理與探索、OLS 建模、Lasso 交叉驗證、報告彙整等流程；藍色節點表示主要步驟，箭頭顯示順序。</figcaption>
      </figure>
    </section>
    <section>
      <h2>Figure 2. Python 與入學考試分數分布</h2>
      <figure>
        <img src='data:image/png;base64,{dist_img}' alt='Score distributions' />
        <figcaption>Legend: 左側直方圖為 Python 成績分布，右側為入學考試分數分布；圖例標示各分數來源。</figcaption>
      </figure>
    </section>
    <section>
      <h2>Figure 3. 主要變項相關矩陣</h2>
      <figure>
        <img src='data:image/png;base64,{corr_img}' alt='Correlation matrix' />
        <figcaption>Legend: 熱圖顯示年齡、入學考試、自修時數、Python 與資料庫成績間的 Pearson 相關係數；顏色代表相關方向與強度。</figcaption>
      </figure>
    </section>
    <section>
      <h2>Table 2. Lasso 重要特徵</h2>
      <p>最佳 α = {lasso_alpha:.4f}，模型在全資料的 R² = {lasso_r2:.3f}。</p>
      {dataframe_to_html(table2)}
      <p style='font-size:0.95rem;color:#4b5563;'>Legend: 僅列出 Lasso 交叉驗證後保留的非零係數；正值代表與 Python 成績正向關聯，負值代表反向關聯。</p>
    </section>
  </div>
</body>
</html>"""

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path.write_text(html, encoding="utf-8")
    return report_path


def main() -> None:
    configure_logging()
    logger = logging.getLogger("main")
    logger.info("補充報告流程啟動")

    df = load_data()
    table1 = build_table1(df)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    table1_path = DOCS_DIR / f"table1_demographics_{TIMESTAMP}.csv"
    table1.to_csv(table1_path, index=False, encoding="utf-8")

    lasso_table, lasso_alpha, lasso_r2 = run_lasso(df)
    table2_path = DOCS_DIR / f"table2_lasso_{TIMESTAMP}.csv"
    lasso_table.to_csv(table2_path, index=False, encoding="utf-8")

    flow_path = FIG_DIR / f"figure1_flow_{TIMESTAMP}.png"
    dist_path = FIG_DIR / f"figure2_distributions_{TIMESTAMP}.png"
    corr_path = FIG_DIR / f"figure3_correlation_{TIMESTAMP}.png"

    plot_flow(flow_path)
    plot_distributions(df, dist_path)
    plot_correlation(df, corr_path)

    report_path = generate_html(table1, lasso_table, flow_path, dist_path, corr_path, lasso_alpha, lasso_r2)

    summary = {
        "timestamp": TIMESTAMP,
        "report_path": str(report_path),
        "table1_path": str(table1_path),
        "table2_path": str(table2_path),
        "figure_paths": {
            "flow": str(flow_path),
            "distribution": str(dist_path),
            "correlation": str(corr_path),
        },
        "lasso_alpha": lasso_alpha,
        "lasso_r2": lasso_r2,
        "log_file": str(LOG_FILE),
    }
    summary_path = DOCS_DIR / f"tables_figures_summary_{TIMESTAMP}.json"
    summary_path.write_text(json.dumps(summary, indent=4), encoding="utf-8")

    logger.info("報告完成：%s", report_path)
    logger.info("Table1：%s", table1_path)
    logger.info("Table2：%s", table2_path)


if __name__ == "__main__":
    main()
