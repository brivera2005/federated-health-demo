from __future__ import annotations

import math
import random
from dataclasses import dataclass

FEATURES = ["age_norm", "comorbidity_score", "prior_admissions", "los_days_norm", "ed_visits"]

SITES = {
  "site-a": {
    "name": "Riverview Medical Center",
    "seed": 42,
    "n": 240,
    "bias_shift": 0.15,
  },
  "site-b": {
    "name": "Summit Community Hospital",
    "seed": 99,
    "n": 180,
    "bias_shift": -0.1,
  },
}


@dataclass
class PatientRow:
  features: list[float]
  label: int


def _sigmoid(z: float) -> float:
  if z >= 0:
    ez = math.exp(-z)
    return 1 / (1 + ez)
  ez = math.exp(z)
  return ez / (1 + ez)


def generate_site_data(site_id: str) -> list[PatientRow]:
  cfg = SITES[site_id]
  rng = random.Random(cfg["seed"])
  rows: list[PatientRow] = []
  for _ in range(cfg["n"]):
    age_norm = rng.uniform(0.2, 1.0)
    comorb = rng.uniform(0, 1)
    prior = rng.randint(0, 4) / 4
    los = rng.uniform(0.1, 1.0)
    ed = rng.randint(0, 3) / 3
    x = [age_norm, comorb, prior, los, ed]
    logit = -1.2 + 1.8 * comorb + 1.4 * prior + 0.9 * los + cfg["bias_shift"]
    label = 1 if _sigmoid(logit + rng.gauss(0, 0.35)) > 0.5 else 0
    rows.append(PatientRow(features=x, label=label))
  return rows


class LogisticModel:
  def __init__(self, n_features: int) -> None:
    self.weights = [0.0] * n_features
    self.bias = 0.0

  def predict_proba(self, x: list[float]) -> float:
    z = self.bias + sum(w * xi for w, xi in zip(self.weights, x))
    return _sigmoid(z)

  def fit(self, data: list[PatientRow], epochs: int = 80, lr: float = 0.08) -> None:
    n = len(data)
    for _ in range(epochs):
      for row in data:
        pred = self.predict_proba(row.features)
        err = pred - row.label
        for i in range(len(self.weights)):
          self.weights[i] -= lr * err * row.features[i] / n
        self.bias -= lr * err / n

  def evaluate(self, data: list[PatientRow]) -> dict[str, float]:
    tp = fp = tn = fn = 0
    scores: list[tuple[float, int]] = []
    for row in data:
      p = self.predict_proba(row.features)
      scores.append((p, row.label))
      pred = 1 if p >= 0.5 else 0
      if pred == 1 and row.label == 1:
        tp += 1
      elif pred == 1:
        fp += 1
      elif row.label == 1:
        fn += 1
      else:
        tn += 1
    total = max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    accuracy = (tp + tn) / total
    # Simple AUC proxy via Mann-Whitney style rank
    pos = [s for s, y in scores if y == 1]
    neg = [s for s, y in scores if y == 0]
    if not pos or not neg:
      auc = 0.5
    else:
      wins = sum(1 for p in pos for n_ in neg if p > n_)
      ties = sum(1 for p in pos for n_ in neg if p == n_)
      auc = (wins + 0.5 * ties) / (len(pos) * len(neg))
    return {
      "accuracy": round(accuracy, 3),
      "precision": round(precision, 3),
      "recall": round(recall, 3),
      "auc_proxy": round(auc, 3),
    }

  def copy(self) -> LogisticModel:
    m = LogisticModel(len(self.weights))
    m.weights = list(self.weights)
    m.bias = self.bias
    return m
