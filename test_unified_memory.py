"""
test_unified_memory.py
======================
Unit testing verification for Phase 4.3 (Unified Metadata ML/DL Schema).
Tests unified FAISS single-index constraint, performance weighting evaluation,
multi-modal architecture integration, and routing extraction logic.
"""

import os
import unittest
import numpy as np

from unified_memory import (
    MemoryEntry,
    PerformanceMetrics,
    UnifiedMemoryStore,
    paradigm_aware_selection,
    unified_cold_start
)
from cold_start import ColdStartConfig

class TestUnifiedMemory(unittest.TestCase):
    
    def setUp(self):
        self.store = UnifiedMemoryStore()
        
        # Build deterministic dataset embedding
        self.emb_base = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float32)
        
        self.ml_entry = MemoryEntry(
            dataset_embedding=list(self.emb_base),
            paradigm="ML",
            model_name="rf",
            performance=PerformanceMetrics(f1=0.85),
            training_time=10.0,
            dataset_id="ds_1"
        )
        
        self.dl_entry = MemoryEntry(
            dataset_embedding=list(self.emb_base), # EXACT same underlying dataset 
            paradigm="DL",
            architecture_config={"layers": 4, "units_per_layer": 128},
            performance=PerformanceMetrics(f1=0.90), # Better but
            training_time=120.0, # Much slower
            dataset_id="ds_1"
        )
        
    def test_schema_assignment(self):
        """Validates dataclass accurately parses complex layered paradigm constraints."""
        self.assertEqual(self.ml_entry.paradigm, "ML")
        self.assertEqual(self.ml_entry.model_name, "rf")
        self.assertIsNone(self.ml_entry.architecture_config)
        self.assertEqual(self.ml_entry.performance.f1, 0.85)

        self.assertEqual(self.dl_entry.paradigm, "DL")
        self.assertEqual(self.dl_entry.architecture_config["layers"], 4)
        self.assertIsNone(self.dl_entry.model_name)
        
    def test_single_faiss_index(self):
        """Ensure FAISS embeds exactly one matrix mapping to both paradigms."""
        self.store.add(self.ml_entry)
        self.store.add(self.dl_entry)
        self.store.build_index()
        
        # Crucially: Index ntotal must equal all elements combined.
        self.assertEqual(self.store.index.ntotal, 2)
        self.assertEqual(self.store.embeddings.shape, (2, 9))
    
    def test_paradigm_aware_ranking(self):
        """Verifies score scaling logic directly balances similarity vs time."""
        self.store.add(self.ml_entry)
        self.store.add(self.dl_entry)
        
        # Simulate perfect hit
        indices = np.array([0, 1])
        sims = np.array([1.0, 1.0]) 
        
        # Test 1: High time penalty favors ML (despite DL having higher f1)
        top_entries, best_par, dist = paradigm_aware_selection(
            indices, sims, self.store, top_n=2, time_penalty_lambda=0.01
        )
        # ML f1=0.85 * 1.0 - (10 * 0.01) = 0.85 - 0.10 = 0.75
        # DL f1=0.90 * 1.0 - (120 * 0.01) = 0.90 - 1.20 = -0.30
        self.assertEqual(best_par, "ML")
        self.assertEqual(top_entries[0].paradigm, "ML")
        
        # Test 2: Zero time penalty favors DL purely based on F1
        top_unpenalized, bp_unpen, _ = paradigm_aware_selection(
            indices, sims, self.store, top_n=2, time_penalty_lambda=0.0
        )
        self.assertEqual(bp_unpen, "DL")
        self.assertEqual(top_unpenalized[0].paradigm, "DL")

    def test_unified_cold_start_routing(self):
        """Test final integrated output structure maps to required schema exactly."""
        self.store.add(self.ml_entry)
        self.store.add(self.dl_entry)
        self.store.build_index()
        
        res = unified_cold_start(self.emb_base, self.store, log_dir="test_unified_log.json")
        
        # Exact expected Output Validation
        self.assertIn("decision", res)
        self.assertIn("selected_models", res)
        self.assertIn("paradigm_distribution", res)
        self.assertIn("best_model_type", res)
        self.assertIn("estimated_performance", res)
        
        self.assertEqual(res["decision"], "memory")
        self.assertEqual(len(res["selected_models"]), 2)
        
        # DL 4L parsing logic
        self.assertIn("DL:MLP_4L", res["selected_models"])
        self.assertIn("ML:rf", res["selected_models"])
        
        # Both got picked, ML is rank 1 because of time penalty evaluation
        self.assertEqual(res["best_model_type"], "ML")
        self.assertEqual(res["paradigm_distribution"], {"ML": 0.5, "DL": 0.5})

    def tearDown(self):
        if os.path.exists("test_unified_log.json"):
            os.remove("test_unified_log.json")

if __name__ == "__main__":
    unittest.main()
