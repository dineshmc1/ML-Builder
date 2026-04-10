"""
unified_memory.py
=================
Phase 4.3 — Unified Meta-Learning Memory (ML + DL)

Implements a unified FAISS memory store linking dataset embeddings to both
Machine Learning (ML) and Deep Learning (DL) best-performing configurations.
Includes a paradigm-aware selection algorithm prioritizing top architectural
candidates for a cold-start meta-learning routing algorithm.

Public API
----------
- ``MemoryEntry``         — Pydantic-like dataclass specifying unified schema
- ``UnifiedMemoryStore``  — Manages a single shared FAISS index for all entries
- ``paradigm_aware_selection`` - Evaluation/Ranking execution logic
- ``unified_cold_start``  — The Phase 4.2 integrated routing wrapper
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure Phase 4.2 components are accessible
from cold_start import (
    ColdStartConfig,
    build_faiss_index,
    search_faiss,
    compute_similarity_scores,
    compute_adaptive_threshold,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core Unified Schema
# ---------------------------------------------------------------------------

@dataclass
class PerformanceMetrics:
    accuracy: float = 0.0
    f1: float = 0.0
    auc: float = 0.0
    
    # Optional regression
    rmse: Optional[float] = None


@dataclass
class MemoryEntry:
    """Unified entry structure accommodating ML and DL models identically."""
    dataset_embedding: List[float]
    paradigm: str  # "ML" or "DL"
    
    # ML-specific
    model_name: Optional[str] = None
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    
    # DL-specific
    architecture_config: Optional[Dict[str, Any]] = None
    
    # Shared metrics
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    training_time: float = 0.0
    
    # Metadata
    dataset_id: str = "unknown"
    task_type: str = "classification"

    def to_dict(self) -> Dict[str, Any]:
        """Convert entry to pure JSON-compatible dict."""
        d = asdict(self)
        return d
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MemoryEntry:
        """Instantiate from a raw dictionary."""
        perf_data = data.pop("performance", {})
        perf = PerformanceMetrics(**perf_data)
        return cls(performance=perf, **data)


# ---------------------------------------------------------------------------
# Unified FAISS Store
# ---------------------------------------------------------------------------

class UnifiedMemoryStore:
    """Manages ONE shared FAISS index routing both ML and DL signatures."""
    
    def __init__(self) -> None:
        self.entries: List[MemoryEntry] = []
        self._index: Optional[Any] = None

    def add(self, entry: MemoryEntry) -> None:
        """Adds contiguous new representations."""
        if entry.paradigm not in ("ML", "DL"):
            raise ValueError("Paradigm must be explicitly 'ML' or 'DL'")
        self.entries.append(entry)
        self._index = None  # invalidate cache

    def build_index(self) -> None:
        """Projects all memory embeddings into single FAISS L2 space."""
        if not self.entries:
            raise ValueError("UnifiedMemoryStore is empty.")
        matrix = np.array([e.dataset_embedding for e in self.entries], dtype=np.float32)
        self._index = build_faiss_index(matrix, use_ip=False)

    @property
    def index(self) -> Any:
        if self._index is None:
            self.build_index()
        return self._index

    @property
    def embeddings(self) -> np.ndarray:
        return np.array([e.dataset_embedding for e in self.entries], dtype=np.float32)

    def save(self, filepath: str) -> None:
        """Persist JSON representation mapping to FAISS indexing."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        raw_list = [e.to_dict() for e in self.entries]
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(raw_list, f, indent=2)
        logger.info("[UnifiedMemory] Saved %d mixed paradigm entries to %s", len(self.entries), filepath)

    def load(self, filepath: str) -> None:
        """Reload representation."""
        with open(filepath, "r", encoding="utf-8") as f:
            raw_list = json.load(f)
        self.entries = [MemoryEntry.from_dict(d) for d in raw_list]
        self._index = None
        logger.info("[UnifiedMemory] Loaded %d mixed paradigm entries from %s", len(self.entries), filepath)

    def __len__(self) -> int:
        return len(self.entries)


# ---------------------------------------------------------------------------
# Paradigm-Aware Selection
# ---------------------------------------------------------------------------

def paradigm_aware_selection(
    neighbor_indices: np.ndarray,
    similarities: np.ndarray,
    store: UnifiedMemoryStore,
    top_n: int = 5,
    performance_weight: float = 1.0,
    time_penalty_lambda: float = 0.01
) -> Tuple[List[MemoryEntry], str, Dict[str, float]]:
    """Calculates paradigm weighting given neighborhood query.
    
    Formula: score = similarity * performance_weight - time_penalty
    Uses F1 for classification and inverse RMSE for regression.
    """
    candidates = []
    
    for idx, similarity in zip(neighbor_indices, similarities):
        if idx >= len(store):
            continue
        entry = store.entries[int(idx)]
        
        # Approximate baseline proxy for unified score evaluating ML against DL
        perf = entry.performance.f1 if entry.task_type == "classification" else entry.performance.accuracy
        if perf == 0.0:  # fallback mapping
            perf = 0.5 
            
        # time penalty linearly scaled per second (e.g. DL takes longer)
        time_penalty = entry.training_time * time_penalty_lambda
        
        # Multi-variable objective mapping
        score = (similarity * performance_weight * perf) - time_penalty
        
        candidates.append({"entry": entry, "score": score, "similarity": similarity})

    # Decouple identical model types avoiding repetitive redundant sweeps
    candidates.sort(key=lambda x: x["score"], reverse=True)
    
    selected_entries = []
    seen_strategies = set()
    
    for c in candidates:
        if len(selected_entries) >= top_n:
            break
        ent = c["entry"]
        # Use simple string hashing for strategy uniqueness (e.g., 'ML-RandomForest' vs 'DL-3LayerTransformer')
        strat_key = f"{ent.paradigm}-{ent.model_name if ent.paradigm == 'ML' else json.dumps(ent.architecture_config)}"
        if strat_key not in seen_strategies:
            seen_strategies.add(strat_key)
            selected_entries.append(ent)
            
    # Extraction telemetry parameters
    ml_count = sum(1 for e in selected_entries if e.paradigm == "ML")
    dl_count = sum(1 for e in selected_entries if e.paradigm == "DL")
    
    total = len(selected_entries) or 1
    dist = {"ML": ml_count / total, "DL": dl_count / total}
    best_chosen = selected_entries[0].paradigm if selected_entries else "ML"
    
    return selected_entries, best_chosen, dist


# ---------------------------------------------------------------------------
# Integrated Entry Router
# ---------------------------------------------------------------------------

def unified_cold_start(
    query_embedding: np.ndarray,
    store: UnifiedMemoryStore,
    config: Optional[ColdStartConfig] = None,
    task_type: str = "classification",
    log_dir: str = "reports/unified_memory.json"
) -> Dict[str, Any]:
    """Phase 4.3 Integrated Routing Output."""
    
    if config is None:
        config = ColdStartConfig()
        
    start_time = time.time()
    query_embedding = np.asarray(query_embedding, dtype=np.float32)
    
    # 1. Edge Case Handler
    if len(store) == 0:
        return {
            "decision": "cold_start",
            "selected_models": ["logistic", "rf", "simple_mlp", "xgboost", "gb"],
            "paradigm_distribution": {"ML": 1.0, "DL": 0.0},
            "best_model_type": "ML",
            "estimated_performance": 0.0,
            "epsilon": 0.0,
            "similarity_score": 0.0
        }
        
    # 2. Phase 4.2 Core Logic (Unified Shared FAISS Search)
    k = min(config.k_neighbors, len(store))
    _, indices = search_faiss(store.index, query_embedding, k=k)
    neighbor_idx = indices[0]
    
    sims = compute_similarity_scores(query_embedding, store.embeddings, neighbor_idx)
    epsilon, mu_s, sigma_s = compute_adaptive_threshold(sims, config.lambda_sensitivity)
    
    # robust mean metric (recommendation phase 7)
    s_dm = float(np.mean(np.sort(sims)[::-1][:config.top_k_for_score]))
    
    # 3. Decision Logic Integrations
    if s_dm >= epsilon:
        decision = "memory"
        # 4. Phase 4.3 Paradigm-Aware Memory Selection
        top_entries, best_type, dist = paradigm_aware_selection(
            neighbor_idx, sims, store, top_n=config.memory_models_count,
        )
        
        # Assemble standard extractable configurations
        selected_configs = []
        for e in top_entries:
            if e.paradigm == "ML":
                selected_configs.append(f"ML:{e.model_name}")
            else:
                layers = e.architecture_config.get("layers", 3) if e.architecture_config else 3
                selected_configs.append(f"DL:MLP_{layers}L")
                
        estimated_f1 = float(np.mean([e.performance.f1 for e in top_entries])) if top_entries else 0.5
        
    else:
        # LLM Broader Routing
        decision = "cold_start"
        # Dummy mock representing the fallback
        selected_configs = [
            "ML:logistic", "ML:rf", "ML:xgboost", "DL:MLP_2L", "DL:MLP_4L"
        ]
        best_type = "ML"
        dist = {"ML": 0.6, "DL": 0.4}
        estimated_f1 = 0.0
        
    # Phase 7 Log Append 
    time_saved = 0.0
    if decision == "memory":
        time_saved = sum(e.training_time for idx in neighbor_idx if idx < len(store) for e in [store.entries[int(idx)]])

    log_entry = {
        "timestamp": time.time(),
        "query_time": time.time() - start_time,
        "decision": decision,
        "ML_pct": dist["ML"],
        "DL_pct": dist["DL"],
        "best_chosen": best_type,
        "time_saved_s": time_saved,
        "estimated_perf": estimated_f1
    }
    
    # Log dump (paper trace requirement)
    try:
        os.makedirs(os.path.dirname(log_dir) or ".", exist_ok=True)
        if os.path.exists(log_dir):
            with open(log_dir, "r") as f:
                logs = json.load(f)
        else:
            logs = []
        logs.append(log_entry)
        with open(log_dir, "w") as f:
            json.dump(logs, f, indent=2)
    except Exception as exc:
        logger.warning("Paper logging failed: %s", exc)

    # 4. Target Specified JSON Output 
    return {
        "decision": decision,
        "epsilon": round(epsilon, 4),
        "similarity_score": round(s_dm, 4),
        "selected_models": selected_configs,
        "paradigm_distribution": dist,
        "best_model_type": best_type,
        "estimated_performance": round(estimated_f1, 4)
    }
