import argparse
import json
import tarfile
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import xgboost as xgb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-artifacts", type=str, default="/opt/ml/processing/model")
    parser.add_argument("--evaluation-data", type=str, default="/opt/ml/processing/evaluation")
    parser.add_argument("--output", type=str, default="/opt/ml/processing/output")
    parser.add_argument("--input-filename", type=str, default="validation.csv")
    parser.add_argument("--prediction-threshold", type=float, default=0.5)
    return parser.parse_args()


def extract_model(model_dir: Path) -> Path:
    model_tar = model_dir / "model.tar.gz"
    if not model_tar.exists():
        raise FileNotFoundError(f"Could not find model artifact at {model_tar}")

    with tarfile.open(model_tar) as tar:
        tar.extractall(path=model_dir)

    for candidate in model_dir.rglob("*"):
        if candidate.is_file() and candidate.name in {"xgboost-model", "model.bin"}:
            return candidate
        if candidate.suffix in {".json", ".bst", ".bin", ".model", ".xgb"}:
            return candidate

    raise FileNotFoundError("Could not locate a supported XGBoost model file after extraction")


def load_dataset(data_dir: Path, filename: str) -> xgb.DMatrix:
    data_path = data_dir / filename
    if not data_path.exists():
        raise FileNotFoundError(f"Evaluation data not found at {data_path}")

    df = pd.read_csv(data_path, header=None)
    if df.empty:
        raise ValueError("Evaluation dataset is empty")

    labels = df.iloc[:, 0].to_numpy()
    features = df.iloc[:, 1:]
    return xgb.DMatrix(features, label=labels)


def compute_metrics(labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray) -> Dict[str, Optional[float]]:
    if labels.size == 0:
        raise ValueError("No labels provided for evaluation")

    accuracy = float(np.mean(predictions == labels))

    positives = predictions == 1
    negatives = predictions == 0
    true_positives = float(np.sum(positives & (labels == 1)))
    true_negatives = float(np.sum(negatives & (labels == 0)))
    false_positives = float(np.sum(positives & (labels == 0)))
    false_negatives = float(np.sum(negatives & (labels == 1)))

    precision = float(true_positives / (true_positives + false_positives)) if (true_positives + false_positives) else 0.0
    recall = float(true_positives / (true_positives + false_negatives)) if (true_positives + false_negatives) else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    roc_auc = None
    try:
        from sklearn.metrics import roc_auc_score

        roc_auc = float(roc_auc_score(labels, probabilities))
    except Exception:
        # sklearn may not be available in the container - skip AUC if import or computation fails
        roc_auc = None

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "true_positives": true_positives,
        "true_negatives": true_negatives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def to_report(metrics: Dict[str, Optional[float]]) -> Dict[str, object]:
    def metric_entry(value: Optional[float]) -> Dict[str, object]:
        return {"value": value if value is None else round(value, 4), "standard_deviation": "NaN"}

    return {
        "binary_classification_metrics": {
            "accuracy": metric_entry(metrics.get("accuracy")),
            "precision": metric_entry(metrics.get("precision")),
            "recall": metric_entry(metrics.get("recall")),
            "f1": metric_entry(metrics.get("f1")),
            "roc_auc": metric_entry(metrics.get("roc_auc")),
            "confusion_matrix": {
                "true_positives": metrics.get("true_positives"),
                "true_negatives": metrics.get("true_negatives"),
                "false_positives": metrics.get("false_positives"),
                "false_negatives": metrics.get("false_negatives"),
            },
        }
    }


def main() -> None:
    args = parse_args()

    model_dir = Path(args.model_artifacts)
    evaluation_dir = Path(args.evaluation_data)
    output_dir = Path(args.output)

    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = extract_model(model_dir)
    booster = xgb.Booster()
    booster.load_model(str(model_path))

    dmatrix = load_dataset(evaluation_dir, args.input_filename)
    probabilities = booster.predict(dmatrix)
    predictions = (probabilities >= args.prediction_threshold).astype(int)
    labels = dmatrix.get_label().astype(int)

    metrics = compute_metrics(labels, predictions, probabilities)
    report = to_report(metrics)

    report_path = output_dir / "evaluation.json"
    with report_path.open("w") as fp:
        json.dump(report, fp)

    detailed_results = pd.DataFrame({
        "label": labels,
        "prediction": predictions,
        "score": probabilities,
    })
    detailed_results.to_csv(output_dir / "detailed_predictions.csv", index=False)

    print(f"Saved evaluation report to {report_path}")


if __name__ == "__main__":
    main()
