"""Optional local text classifiers used as traditional-model baselines."""
from __future__ import annotations

from typing import Any, Dict, List


def record_text(record: Dict[str, Any]) -> str:
    """Flatten the human-readable fields of a benchmark record."""
    return " ".join(str(record[key]) for key in ("text", "query", "passage", "task") if key in record)


class TraditionalModel:
    """TF-IDF plus a gradient-boosting classifier.

    Dependencies are imported lazily so the API can still serve the workbench
    when optional ML wheels are unavailable. In that case ``available`` is
    false and callers can show an explicit unavailable status.
    """

    def __init__(self, name: str):
        self.name = name
        self.available = False
        self.error = ""
        self._vectorizer = None
        self._encoder = None
        self._model = None
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.preprocessing import LabelEncoder
            self._vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=1, max_features=3000)
            self._encoder = LabelEncoder()
            if name == "xgboost":
                from xgboost import XGBClassifier
                self._model_factory = lambda classes: XGBClassifier(
                    n_estimators=80, max_depth=4, learning_rate=0.08,
                    subsample=0.9, colsample_bytree=0.9, objective="multi:softprob",
                    num_class=classes, eval_metric="mlogloss", tree_method="hist",
                    random_state=42, n_jobs=1,
                )
            elif name == "lightgbm":
                from lightgbm import LGBMClassifier
                self._model_factory = lambda classes: LGBMClassifier(
                    n_estimators=80, num_leaves=15, learning_rate=0.08,
                    objective="multiclass", num_class=classes,
                    random_state=42, verbosity=-1, n_jobs=1,
                )
            else:
                raise ValueError(f"unknown traditional model: {name}")
            self.available = True
        except Exception as exc:  # pragma: no cover - depends on host wheels
            self.error = f"{type(exc).__name__}: {exc}"

    def fit(self, records: List[Dict[str, Any]]) -> None:
        if not self.available:
            raise RuntimeError(f"{self.name} unavailable: {self.error}")
        texts = [record_text(record) for record in records]
        labels = [str(record.get("label", "")) for record in records]
        matrix = self._vectorizer.fit_transform(texts)
        encoded = self._encoder.fit_transform(labels)
        self._model = self._model_factory(len(self._encoder.classes_))
        self._model.fit(matrix, encoded)

    def predict(self, record: Dict[str, Any]) -> Dict[str, Any]:
        if self._model is None:
            raise RuntimeError(f"{self.name} is not fitted")
        matrix = self._vectorizer.transform([record_text(record)])
        probabilities = self._model.predict_proba(matrix)[0]
        index = int(probabilities.argmax())
        return {
            "prediction": str(self._encoder.inverse_transform([index])[0]),
            "confidence": float(probabilities[index]),
            "probabilities": {str(label): float(probabilities[i]) for i, label in enumerate(self._encoder.classes_)},
        }


def local_model_status() -> List[Dict[str, Any]]:
    """Return availability metadata for the local traditional models."""
    return [{"model": name, "available": TraditionalModel(name).available} for name in ("xgboost", "lightgbm")]
