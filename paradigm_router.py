import litellm
import os
import json
import numpy as np

def calculate_heuristics_d(num_rows, num_cols, problem_type):
    """Rule-based complexity score (0 to 1)"""
    score = 0.0
    # Large datasets favor DL
    if num_rows > 50000: score += 0.4
    elif num_rows > 10000: score += 0.2
    
    # High dimensionality favors DL
    if num_cols > 100: score += 0.3
    elif num_cols > 50: score += 0.1
    
    # Non-linear/Complex tasks (e.g., multiclass with many classes)
    # For now, we cap it at 1.0
    return min(score, 1.0)

def calculate_llm_d(dataset_profile):
    """Ask LLM for probability that DL is better than ML"""
    prompt = f"""You are an expert AutoML system. Analyze this dataset profile:
    - Rows: {dataset_profile['num_samples']}
    - Features: {dataset_profile['num_features']}
    - Problem: {dataset_profile['problem_type']}
    - Sample Columns: {dataset_profile['sample_columns']}
    
    What is the probability (from 0.0 to 1.0) that a Deep Learning approach (Neural Networks) 
    will outperform Classical ML (XGBoost, LightGBM, Random Forest) on this specific dataset?
    Return ONLY a JSON object with the key "probability". Example: {{"probability": 0.2}}
    """
    
    try:
        response = litellm.completion(
            model=os.getenv("LLM_MODEL", "openai/gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        content = json.loads(response.choices[0].message.content)
        return float(content.get("probability", 0.2)) # Default to ML (0.2) if parsing fails
    except Exception as e:
        print(f"[Router] LLM(D) failed: {e}. Defaulting to 0.2")
        return 0.2

def calculate_memory_d(faiss_store, query_embedding, k=5):
    """Check if top-K similar past datasets used DL as the winner"""
    # Since our current memory store only has ML models, this will naturally be 0.0
    # In the future, if metadata contains 'is_dl': True, this will activate.
    try:
        if faiss_store._index is None or len(faiss_store.records) == 0:
            return 0.0
        # FAISS expects 2D array for search
        q_vec = np.ascontiguousarray(np.reshape(query_embedding, (1, -1)), dtype=np.float32)
        distances, indices = faiss_store._index.search(q_vec, k)
        dl_count = 0
        valid_neighbors = 0
        for idx in indices[0]:
            if idx != -1:
                valid_neighbors += 1
                meta = faiss_store.records[idx].metadata
                if meta.get('is_dl', False): # Check if past winner was DL
                    dl_count += 1
        return (dl_count / valid_neighbors) if valid_neighbors > 0 else 0.0
    except Exception as e:
        print(f"[Router] Memory(D) failed: {e}")
        return 0.0

def route_paradigm(dataset_profile, faiss_store, query_embedding, lambda1=0.5, lambda2=0.2, lambda3=0.3, tau=0.5):
    """
    R(D) = λ₁ · LLM(D) + λ₂ · Memory(D) + λ₃ · Heuristics(D)
    """
    llm_score = calculate_llm_d(dataset_profile)
    memory_score = calculate_memory_d(faiss_store, query_embedding)
    heuristics_score = calculate_heuristics_d(dataset_profile['num_samples'], dataset_profile['num_features'], dataset_profile['problem_type'])
    
    r_d = (lambda1 * llm_score) + (lambda2 * memory_score) + (lambda3 * heuristics_score)
    
    decision = "AutoDL" if r_d > tau else "AutoML"
    
    print(f"[Paradigm Router] LLM: {llm_score:.2f} | Memory: {memory_score:.2f} | Heuristics: {heuristics_score:.2f}")
    print(f"[Paradigm Router] R(D) = {r_d:.3f} (Threshold τ = {tau}) -> Decision: {decision}")
    
    return decision, r_d, llm_score, memory_score, heuristics_score
