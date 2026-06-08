import re

with open('phase4_pipeline.py', 'r', encoding='utf-8') as f:
    code = f.read()

# Replace run_single_dataset_pipeline definition
old_def = 'def run_single_dataset_pipeline(X, y, problem_type, store, encoder, did="local", validate=False):'
new_def = 'def run_single_dataset_pipeline(X, y, problem_type, store, encoder, did="local", validate=False, modality="tabular"):'
code = code.replace(old_def, new_def)

# Add modality override block and indent the tabular logic
tabular_logic_start = code.find('    # 1. Extract meta-features')
tabular_logic_end = code.find('    if paradigm_decision == "AutoML":')

tabular_block = code[tabular_logic_start:tabular_logic_end]
indented_tabular = '\n'.join(['    ' + line if line.strip() else line for line in tabular_block.split('\n')])

override_block = """    # 🚨 MULTI-MODAL OVERRIDE LOGIC 🚨
    if modality in ['vision', 'audio', 'text', 'video']:
        print("\\n" + "="*50)
        print("🚨 MULTI-MODAL OVERRIDE DETECTED")
        print("="*50)
        print("🛑 Bypassing Tabular FAISS Memory (Incompatible Vector Space)")
        print("🛑 Bypassing Tabular R(D) Router")
        print("🧠 Forcing Paradigm Decision: AutoDL NAS")
        print("="*50 + "\\n")
        
        paradigm_decision = "AutoDL"
        r_d_score = 1.0
        llm_score, memory_score, heuristics_score = 1.0, 0.0, 1.0
        warm_params = {}
        best_retrieved_models = []
        w1_def, w2_def, w3_def = (0.6, 0.3, 0.1)
        
    else:
"""

code = code[:tabular_logic_start] + override_block + indented_tabular + code[tabular_logic_end:]

# Add modality=modality to the call inside main()
old_call = """            run_single_dataset_pipeline(
                X, y, problem_type, store, encoder, 
                did=os.path.basename(user_input), 
                validate=args.validate
            )"""
new_call = """            run_single_dataset_pipeline(
                X, y, problem_type, store, encoder, 
                did=os.path.basename(user_input), 
                validate=args.validate,
                modality=modality
            )"""
code = code.replace(old_call, new_call)

with open('phase4_pipeline.py', 'w', encoding='utf-8') as f:
    f.write(code)
print("Patched phase4_pipeline.py successfully!")
