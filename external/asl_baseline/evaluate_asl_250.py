import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "model.tflite"
DEFAULT_LABEL_MAP = ROOT / "sign_to_prediction_index_map.json"


def load_label_map(path):
    with open(path, "r", encoding="utf-8") as f:
        sign_to_id = json.load(f)
    id_to_sign = {idx: sign for sign, idx in sign_to_id.items()}
    return sign_to_id, id_to_sign


def load_manifest(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = {"path", "label"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest must contain columns: path,label. Missing: {sorted(missing)}")
        return list(reader)


def load_landmarks(path):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
    elif path.suffix.lower() == ".npz":
        data = np.load(path)
        key = "landmarks" if "landmarks" in data.files else data.files[0]
        arr = data[key]
    else:
        raise ValueError(f"Unsupported landmark file: {path}. Use .npy or .npz.")

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[1:] != (543, 3):
        raise ValueError(f"{path} must have shape (frames, 543, 3), got {arr.shape}")
    return arr


def softmax_if_needed(scores):
    scores = scores.astype(np.float32)
    total = float(np.sum(scores))
    if np.all(scores >= 0) and 0.95 <= total <= 1.05:
        return scores
    exp = np.exp(scores - np.max(scores))
    return exp / np.sum(exp)


class TFLitePredictor:
    def __init__(self, model_path):
        self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
        self.input_index = self.interpreter.get_input_details()[0]["index"]
        self.output_index = self.interpreter.get_output_details()[0]["index"]

    def predict_scores(self, landmarks):
        start = time.perf_counter()
        self.interpreter.resize_tensor_input(self.input_index, landmarks.shape, strict=False)
        self.interpreter.allocate_tensors()
        self.interpreter.set_tensor(self.input_index, landmarks)
        self.interpreter.invoke()
        scores = self.interpreter.get_tensor(self.output_index)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return softmax_if_needed(scores), elapsed_ms


def top_k_indices(scores, k):
    return np.argsort(scores)[-k:][::-1]


def compute_metrics(y_true, topk_predictions, num_classes, latencies_ms):
    y_pred = [preds[0] for preds in topk_predictions]
    n = len(y_true)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_id, pred_id in zip(y_true, y_pred):
        confusion[true_id, pred_id] += 1

    supports = confusion.sum(axis=1)
    precisions = []
    recalls = []
    f1s = []
    present_classes = np.where(supports > 0)[0]
    for class_id in present_classes:
        tp = confusion[class_id, class_id]
        fp = confusion[:, class_id].sum() - tp
        fn = confusion[class_id, :].sum() - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    f1s = np.array(f1s, dtype=np.float64)
    supports_present = supports[present_classes].astype(np.float64)
    latency = np.array(latencies_ms, dtype=np.float64)
    return {
        "samples": n,
        "top1_accuracy": float(np.mean([t == p[0] for t, p in zip(y_true, topk_predictions)])),
        "top3_accuracy": float(np.mean([t in p[:3] for t, p in zip(y_true, topk_predictions)])),
        "top5_accuracy": float(np.mean([t in p[:5] for t, p in zip(y_true, topk_predictions)])),
        "macro_precision": float(np.mean(precisions)) if precisions else 0.0,
        "macro_recall": float(np.mean(recalls)) if recalls else 0.0,
        "macro_f1": float(np.mean(f1s)) if len(f1s) else 0.0,
        "weighted_f1": float(np.sum(f1s * supports_present) / np.sum(supports_present)) if np.sum(supports_present) else 0.0,
        "latency_mean_ms": float(np.mean(latency)) if len(latency) else 0.0,
        "latency_p95_ms": float(np.percentile(latency, 95)) if len(latency) else 0.0,
    }, confusion


def write_confusion_matrix(path, confusion, id_to_sign):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        labels = [id_to_sign[i] for i in range(len(id_to_sign))]
        writer.writerow(["true\\pred", *labels])
        for i, row in enumerate(confusion):
            writer.writerow([id_to_sign[i], *row.tolist()])


def write_predictions(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["path", "label", "top1", "top1_score", "top3", "top5", "latency_ms"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def benchmark_only(predictor, lengths, runs, warmup):
    results = []
    for length in lengths:
        latencies = []
        sample = np.random.random((length, 543, 3)).astype(np.float32)
        for idx in range(warmup + runs):
            _, elapsed_ms = predictor.predict_scores(sample)
            if idx >= warmup:
                latencies.append(elapsed_ms)
        latencies = np.array(latencies, dtype=np.float64)
        results.append(
            {
                "sequence_length": length,
                "runs": runs,
                "latency_mean_ms": float(np.mean(latencies)),
                "latency_p95_ms": float(np.percentile(latencies, 95)),
                "fps_equivalent": float(1000.0 / np.mean(latencies)),
            }
        )
    return {"benchmark": results}


def main():
    parser = argparse.ArgumentParser(description="Evaluate the pretrained ASL-250 TFLite recognizer.")
    parser.add_argument("--manifest", type=Path, help="CSV with columns path,label for labeled landmark samples.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--label-map", type=Path, default=DEFAULT_LABEL_MAP)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "eval_results")
    parser.add_argument("--benchmark-only", action="store_true", help="Measure latency on synthetic landmark tensors.")
    parser.add_argument("--benchmark-lengths", type=int, nargs="+", default=[16, 32, 60])
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    sign_to_id, id_to_sign = load_label_map(args.label_map)
    predictor = TFLitePredictor(args.model)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.benchmark_only:
        metrics = benchmark_only(predictor, args.benchmark_lengths, args.runs, args.warmup)
        out = args.output_dir / "recognition_latency_benchmark.json"
        out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return

    if not args.manifest:
        raise SystemExit("Provide --manifest or use --benchmark-only.")

    rows = load_manifest(args.manifest)
    y_true = []
    topk_predictions = []
    latencies = []
    prediction_rows = []
    base_dir = args.manifest.parent

    for row in rows:
        label = row["label"].strip()
        if label not in sign_to_id:
            raise ValueError(f"Unknown label '{label}' in {args.manifest}")
        sample_path = Path(row["path"])
        if not sample_path.is_absolute():
            sample_path = base_dir / sample_path
        scores, elapsed_ms = predictor.predict_scores(load_landmarks(sample_path))
        top5 = top_k_indices(scores, 5).tolist()

        y_true.append(sign_to_id[label])
        topk_predictions.append(top5)
        latencies.append(elapsed_ms)
        prediction_rows.append(
            {
                "path": str(sample_path),
                "label": label,
                "top1": id_to_sign[top5[0]],
                "top1_score": f"{scores[top5[0]]:.6f}",
                "top3": ";".join(id_to_sign[i] for i in top5[:3]),
                "top5": ";".join(id_to_sign[i] for i in top5),
                "latency_ms": f"{elapsed_ms:.3f}",
            }
        )

    metrics, confusion = compute_metrics(y_true, topk_predictions, len(sign_to_id), latencies)
    metrics_path = args.output_dir / "recognition_metrics.json"
    predictions_path = args.output_dir / "recognition_predictions.csv"
    confusion_path = args.output_dir / "recognition_confusion_matrix.csv"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_predictions(predictions_path, prediction_rows)
    write_confusion_matrix(confusion_path, confusion, id_to_sign)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

