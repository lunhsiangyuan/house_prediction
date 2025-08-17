import os
import glob
from typing import List, Dict

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

MODEL_RESULTS_DIR = "model_results"

app = FastAPI(title="House Price Prediction API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"]
)


def get_submission_files(directory: str) -> List[str]:
    pattern = os.path.join(directory, "*_submission.csv")
    return sorted(glob.glob(pattern))


@app.get("/api/files")
def list_files() -> Dict[str, List[str]]:
    files = [os.path.basename(f) for f in get_submission_files(MODEL_RESULTS_DIR)]
    return {"files": files}


@app.get("/api/predictions")
def get_predictions(file: str) -> Dict[str, List[Dict[str, float]]]:
    file_path = os.path.join(MODEL_RESULTS_DIR, file)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    df = pd.read_csv(file_path)
    preview = df.head().to_dict(orient="records")

    if "SalePrice" in df.columns:
        hist = df["SalePrice"].value_counts(bins=50, sort=False)
        histogram = {
            "bins": [f"{int(bin.left)}-{int(bin.right)}" for bin in hist.index],
            "counts": hist.values.tolist(),
        }
    else:
        histogram = {}

    return {"preview": preview, "histogram": histogram}
