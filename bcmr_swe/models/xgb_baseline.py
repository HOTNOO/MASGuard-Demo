"""Optional XGBoost baseline wrapper."""

from __future__ import annotations

import json
from pathlib import Path

from bcmr_swe.models.gcrv import GraphConditionedRecoveryValueModel
from bcmr_swe.types import ActionScore, CandidateAction, FailedState, RecoveryBudget


class XGBoostRecoveryRanker:
    """Train an XGBoost ranker when the dependency is available."""

    def __init__(self):
        try:
            import xgboost as xgb  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "xgboost is not available. Install it into .pydeps or the active Python environment to use the XGB baseline."
            ) from exc
        self.xgb = xgb
        self.model = None
        self.feature_encoder = GraphConditionedRecoveryValueModel()
        self.feature_keys: list[str] = []

    def fit(self, groups, *, n_estimators: int = 100, learning_rate: float = 0.08):
        rows = []
        labels = []
        qids = []
        row_index = 0
        for group_index, (failed_state, actions, outcomes) in enumerate(groups):
            outcome_by_action = {outcome.action_id: outcome for outcome in outcomes}
            for action in actions:
                outcome = outcome_by_action.get(action.action_id)
                if outcome is None:
                    continue
                features = self.feature_encoder._features(failed_state, action)
                rows.append(features)
                labels.append(
                    (1.0 if outcome.recover_success else 0.0)
                    - 1e-4 * outcome.token_cost
                    - 5e-4 * outcome.latency_sec
                    - 0.2 * outcome.secondary_risk
                )
                qids.append(group_index)
                row_index += 1

        if not rows:
            raise ValueError("No training rows were generated for XGBoost.")

        self.feature_keys = sorted({key for row in rows for key in row})
        dense_rows = [self._flatten(row) for row in rows]
        dtrain = self.xgb.DMatrix(dense_rows, label=labels)
        dtrain.set_group(self._group_sizes(qids))
        params = {
            "objective": "rank:pairwise",
            "eta": learning_rate,
            "max_depth": 6,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "eval_metric": "ndcg",
        }
        self.model = self.xgb.train(params, dtrain, num_boost_round=n_estimators)
        return self

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise ValueError("Model has not been trained.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))
        meta_path = self._meta_path(path)
        meta_path.write_text(
            json.dumps({"feature_keys": self.feature_keys}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "XGBoostRecoveryRanker":
        path = Path(path)
        ranker = cls()
        ranker.model = ranker.xgb.Booster()
        ranker.model.load_model(str(path))
        meta_path = ranker._meta_path(path)
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        ranker.feature_keys = [str(item) for item in payload.get("feature_keys", [])]
        return ranker

    def rank(self, failed_state: FailedState, actions: list[CandidateAction], budget: RecoveryBudget) -> list[ActionScore]:
        if self.model is None:
            raise ValueError("Model has not been trained.")
        rows = [self._flatten(self.feature_encoder._features(failed_state, action)) for action in actions]
        dmatrix = self.xgb.DMatrix(rows)
        scores = self.model.predict(dmatrix)
        ranked: list[ActionScore] = []
        for action, utility in zip(actions, scores.tolist()):
            ranked.append(
                ActionScore(
                    action_id=action.action_id,
                    utility=float(utility),
                    estimated_recover_prob=action.estimated_recover_prob,
                    estimated_token_cost=action.estimated_token_cost,
                    estimated_latency_sec=action.estimated_latency_sec,
                    estimated_risk=action.estimated_risk,
                    explanation="xgboost_pairwise_ranker",
                )
            )
        ranked.sort(key=lambda item: item.utility, reverse=True)
        return ranked

    def _flatten(self, features: dict[str, float]) -> list[float]:
        return [float(features.get(key, 0.0)) for key in self.feature_keys]

    def _group_sizes(self, qids: list[int]) -> list[int]:
        counts: list[int] = []
        current = None
        current_count = 0
        for qid in qids:
            if current is None or qid == current:
                current = qid
                current_count += 1
            else:
                counts.append(current_count)
                current = qid
                current_count = 1
        if current_count:
            counts.append(current_count)
        return counts

    def _meta_path(self, path: Path) -> Path:
        return path.with_name(path.name + ".meta.json")
