import sqlite3
import json
import os

DB_PATH = "dl_memory.db"

def init_db():
    """Creates the database and table if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS dl_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_name TEXT,
            modality TEXT,
            best_params TEXT,
            final_accuracy REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_dl_result(dataset_name, modality, best_params, final_accuracy):
    """Saves a new DL result to the database."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    params_json = json.dumps(best_params)
    c.execute('''
        INSERT INTO dl_history (dataset_name, modality, best_params, final_accuracy)
        VALUES (?, ?, ?, ?)
    ''', (dataset_name, modality, params_json, final_accuracy))
    conn.commit()
    conn.close()
    print(f"[DL Memory] ✅ Saved result for '{dataset_name}' ({modality}) to SQLite database.")

def get_similar_dl_results(modality, limit=3):
    """
    Queries the database for the best past models of a specific modality.
    Returns a list of dicts: {dataset, best_params, accuracy}
    """
    if not os.path.exists(DB_PATH):
        return []

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        SELECT dataset_name, best_params, final_accuracy
        FROM dl_history
        WHERE modality = ?
        ORDER BY final_accuracy DESC
        LIMIT ?
    ''', (modality, limit))
    results = c.fetchall()
    conn.close()

    return [
        {
            "dataset": row[0],
            "best_params": json.loads(row[1]),
            "accuracy": row[2]
        }
        for row in results
    ]

def warm_start_from_memory(modality):
    """
    Returns the best_params dict from the highest-accuracy past run for this modality,
    or None if no history exists. Used to warm-start Optuna NAS.
    """
    results = get_similar_dl_results(modality, limit=1)
    if results:
        best = results[0]
        print(f"[DL Memory] 🧠 Warm-starting from past '{modality}' run: "
              f"'{best['dataset']}' (acc={best['accuracy']:.4f})")
        return best["best_params"]
    print(f"[DL Memory] No prior '{modality}' history found. Starting cold.")
    return None
