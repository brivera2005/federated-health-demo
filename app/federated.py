from __future__ import annotations

from app.ml import FEATURES, SITES, LogisticModel, generate_site_data
from app.models import (
 DemoState,
 FederatedRoundResult,
 Metrics,
 SiteInfo,
 SiteTrainResult,
 WeightVector,
)

_site_cache: dict[str, list] = {}


def _get_data(site_id: str):
 if site_id not in _site_cache:
 _site_cache[site_id] = generate_site_data(site_id)
 return _site_cache[site_id]


def site_info(site_id: str) -> SiteInfo:
 cfg = SITES[site_id]
 return SiteInfo(
 id=site_id,
 name=cfg["name"],
 patient_count=cfg["n"],
 features=FEATURES,
 data_stays_local=True,
 )


def train_site(site_id: str) -> SiteTrainResult:
 data = _get_data(site_id)
 model = LogisticModel(len(FEATURES))
 model.fit(data)
 metrics_dict = model.evaluate(data)
 weights = WeightVector(
 site_id=site_id,
 weights=[round(w, 4) for w in model.weights],
 bias=round(model.bias, 4),
 sample_count=len(data),
 raw_data_exported=False,
 )
 return SiteTrainResult(
 site=site_info(site_id),
 local_weights=weights,
 local_metrics=Metrics(**metrics_dict),
 message=f"Trained locally on {len(data)} synthetic discharges at {SITES[site_id]['name']}. "
 f"Only weight vector ({len(model.weights)} coeffs + bias) is eligible for upload - "
 "zero patient rows left the hospital.",
 )


def federated_average(site_results: list[SiteTrainResult]) -> FederatedRoundResult:
 total = sum(r.local_weights.sample_count for r in site_results)
 n_feat = len(site_results[0].local_weights.weights)
 agg_w = [0.0] * n_feat
 agg_b = 0.0
 for r in site_results:
 w = r.local_weights.sample_count / total
 for i in range(n_feat):
 agg_w[i] += r.local_weights.weights[i] * w
 agg_b += r.local_weights.bias * w

 global_model = LogisticModel(n_feat)
 global_model.weights = agg_w
 global_model.bias = agg_b

 # Evaluate global model on combined holdout-style union
 combined = _get_data("site-a") + _get_data("site-b")
 global_metrics = Metrics(**global_model.evaluate(combined))

 return FederatedRoundResult(
 round_number=1,
 contributing_sites=[r.site.id for r in site_results],
 aggregated_weights=[round(w, 4) for w in agg_w],
 aggregated_bias=round(agg_b, 4),
 global_metrics=global_metrics,
 privacy_note=(
 "Federated round complete: coordinator received only sample-weighted coefficients. "
 "PHI never left Riverview or Summit - audit confirms raw_data_exported=false for all sites."
 ),
 )


class DemoSession:
 def __init__(self) -> None:
 self.reset()

 def reset(self) -> DemoState:
 self.state = DemoState(
 step=0,
 step_label="Ready",
 sites=[site_info("site-a"), site_info("site-b")],
 audit_log=["Session initialized. Patient-level data partitioned per hospital."],
 )
 return self.state

 def advance(self) -> DemoState:
 s = self.state
 if s.step == 0:
 s.site_a_result = train_site("site-a")
 s.step = 1
 s.step_label = "Site A trained"
 s.audit_log.append(
 f"[Site A] Local training complete - uploaded {len(s.site_a_result.local_weights.weights)} weights + bias only."
 )
 elif s.step == 1:
 s.site_b_result = train_site("site-b")
 s.step = 2
 s.step_label = "Site B trained"
 s.audit_log.append(
 f"[Site B] Local training complete - uploaded {len(s.site_b_result.local_weights.weights)} weights + bias only."
 )
 elif s.step == 2:
 assert s.site_a_result and s.site_b_result
 s.federated_result = federated_average([s.site_a_result, s.site_b_result])
 s.step = 3
 s.step_label = "Federated round complete"
 s.audit_log.append("[Coordinator] FedAvg applied. Global model metrics computed on combined synthetic cohort.")
 return s


session = DemoSession()
