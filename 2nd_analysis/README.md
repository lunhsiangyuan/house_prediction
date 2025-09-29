# 2nd Analysis Package

本資料夾彙整第二版統計與 Lasso 建模結果，內容涵蓋產出報告、原始指標表格、圖像與分析腳本，方便臨床／教學團隊一次查閱。

## 結構總覽
- `reports/`
  - `report_20250929_103827.html`：Phase 1-3 統計 + Lasso 主報告。
  - `report_tables_figures_20250929_105529.html`：補充報告（Table 1、Figure 1-3、Table 2）。
- `docs/`
  - `*_summary_*.json`：分析摘要（含指標、輸出路徑、Lasso α 與 R² 等）。
  - `table1_demographics_*.csv` / `table2_lasso_*.csv`：表格 1/2 的原始資料。
  - `summary_stats_*`, `missing_values_*`, `correlation_*`, `vif_*`, `breusch_pagan_*`, `ols_coefficients_*`, `lasso_coefficients_*`：統計診斷指標。
  - `flowchart_*.mmd`：Phase 流程圖 Mermaid 定義。
- `figures/`
  - `figure1_flow_*`, `figure2_distributions_*`, `figure3_correlation_*`：補充報告 Figure 1-3。
  - `phase_flow_*`, `python_score_hist_*`, `corr_heatmap_*`：Phase 1-3 主報告圖。
- `scripts/`
  - `stat_lasso_phase_analysis.py`：主分析流程。
  - `report_tables_figures.py`：補充表格／圖像生成。
- `logs/`
  - `analysis_*.log`, `table_figures_*.log`：執行紀錄與診斷訊息。

## 使用建議
1. 若需重建結果，以 `scripts/` 中對應檔案執行即可；輸出將覆寫於 `reports/`、`docs/`、`figures/`、`logs/`。
2. 與臨床分享時，可直接提供 HTML 報告或 CSV/PNG 檔案；表格與圖檔皆附 Legend。
3. 需新增分析時，建議複製腳本至新目錄，以保留此版本的可追溯性。
