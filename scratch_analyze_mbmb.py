import pandas as pd
import numpy as np

file_path = r'c:\Users\admin\Downloads\raw\MBMB_TAX.csv'
try:
    df = pd.read_csv(file_path, low_memory=False)
    
    cols_to_check = ['original_cukai', 'original_denda', 'original_jumlah']
    missing_cols = [c for c in cols_to_check if c not in df.columns]
    
    if missing_cols:
        print(f"Missing columns: {missing_cols}")
        print("Available columns:", df.columns.tolist()[:30])
    else:
        df['original_cukai'] = pd.to_numeric(df['original_cukai'], errors='coerce').fillna(0)
        df['original_denda'] = pd.to_numeric(df['original_denda'], errors='coerce').fillna(0)
        df['original_jumlah'] = pd.to_numeric(df['original_jumlah'], errors='coerce').fillna(0)

        print("--- Correlation Matrix ---")
        corr = df[cols_to_check].corr()
        print(corr)
        
        expected_jumlah = df['original_cukai'] + df['original_denda']
        mismatch = ~np.isclose(expected_jumlah, df['original_jumlah'], atol=1e-4)
        
        mismatch_count = mismatch.sum()
        total_count = len(df)
        mismatch_percent = (mismatch_count / total_count) * 100
        
        print("\n--- Equation Check (original_jumlah = original_cukai + original_denda) ---")
        print(f"Total rows: {total_count}")
        print(f"Rows that DO NOT follow the equation: {mismatch_count}")
        print(f"Percentage of mismatch: {mismatch_percent:.2f}%")

except Exception as e:
    print(f"Error: {e}")
