from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class SiteInfo(BaseModel):
  id: str
  name: str
  patient_count: int
  features: list[str]
  data_stays_local: bool = True


class WeightVector(BaseModel):
  site_id: str
  weights: list[float]
  bias: float
  sample_count: int
  transmitted_fields: list[str] = Field(
    default_factory=lambda: ["weights", "bias", "sample_count"]
  )
  raw_data_exported: bool = False


class Metrics(BaseModel):
  accuracy: float
  precision: float
  recall: float
  auc_proxy: float


class SiteTrainResult(BaseModel):
  site: SiteInfo
  local_weights: WeightVector
  local_metrics: Metrics
  message: str


class FederatedRoundResult(BaseModel):
  round_number: int
  contributing_sites: list[str]
  aggregated_weights: list[float]
  aggregated_bias: float
  global_metrics: Metrics
  privacy_note: str


class DemoState(BaseModel):
  step: int
  step_label: str
  sites: list[SiteInfo]
  site_a_result: Optional[SiteTrainResult] = None
  site_b_result: Optional[SiteTrainResult] = None
  federated_result: Optional[FederatedRoundResult] = None
  audit_log: list[str] = Field(default_factory=list)
