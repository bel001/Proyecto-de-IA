from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.gestures import build_feature_vector  # noqa: E402


def _load_landmarks(csv_path: Path) -> Tuple[np.ndarray, List[str]]:
    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise ValueError(f"El archivo {csv_path} no tiene columna 'label'.")
    labels = df["label"].tolist()
    coords = df.drop(columns=["label"]).to_numpy().astype(np.float32)
    if coords.shape[1] != 63:
        raise ValueError(f"Se esperaban 63 columnas de landmarks y label en {csv_path}, hay {coords.shape[1]}.")
    landmarks = coords.reshape(-1, 21, 3)
    return landmarks, labels


def load_dataset(data_dir: Path) -> Tuple[np.ndarray, List[str]]:
    files = sorted(data_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No se encontraron CSV en {data_dir}")

    all_features: List[np.ndarray] = []
    all_labels: List[str] = []

    for csv_path in files:
        landmarks, labels = _load_landmarks(csv_path)
        for lm, label in zip(landmarks, labels):
            feats = build_feature_vector(lm)
            all_features.append(feats)
            all_labels.append(label)
    X = np.vstack(all_features)
    return X, all_labels


def main() -> int:
    parser = argparse.ArgumentParser(description="Entrena un clasificador de gestos con RandomForest.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"), help="Carpeta con CSV etiquetados.")
    parser.add_argument("--output", type=Path, default=Path("models/gesture_classifier.pkl"), help="Ruta del modelo joblib.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proporcion del set de prueba.")
    parser.add_argument("--estimators", type=int, default=200, help="Arboles del RandomForest.")
    parser.add_argument("--cv-folds", type=int, default=5, help="Folds para cross-validation estratificada (0 para omitir).")
    args = parser.parse_args()

    X, y = load_dataset(args.data_dir)
    base_params = dict(
        n_estimators=args.estimators,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    if args.cv_folds and args.cv_folds > 1:
        cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(RandomForestClassifier(**base_params), X, y, cv=cv, n_jobs=-1)
        print(f"Cross-validation ({args.cv_folds} folds) accuracy: mean={cv_scores.mean():.3f}, std={cv_scores.std():.3f}, scores={np.round(cv_scores,3)}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42, stratify=y)

    clf = RandomForestClassifier(**base_params)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"Accuracy test: {acc:.3f}")
    print(classification_report(y_test, preds))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, args.output)
    print(f"Modelo guardado en {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
