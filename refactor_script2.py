import sys
import re

with open('phase4_pipeline.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 1. Remove old load_local_dataset
start_remove = -1
end_remove = -1
for i, line in enumerate(lines):
    if line.startswith('def load_local_dataset(csv_path, target_column):'):
        start_remove = i
        break

if start_remove != -1:
    for i in range(start_remove, len(lines)):
        if "return None, None" in lines[i]:
            end_remove = i
            break
    if end_remove != -1:
        del lines[start_remove:end_remove+1]

# 2. Add import for load_local_dataset
for i, line in enumerate(lines):
    if "from data_cleaner import clean" in line:
        lines.insert(i, "from data_loader import load_local_dataset\n")
        break

# 3. Find main() and extract loop body
main_idx = -1
for i, line in enumerate(lines):
    if line.startswith('def main():'):
        main_idx = i
        break

start_loop_idx = -1
end_loop_idx = -1
for i in range(main_idx, len(lines)):
    if 'for did in test_ids:' in lines[i]:
        start_loop_idx = i
        break

for i in range(start_loop_idx, len(lines)):
    if 'print(f"  [Phase 5.6 Report] Failed: {e}")' in lines[i]:
        end_loop_idx = i + 2
        break

loop_body = lines[start_loop_idx+10:end_loop_idx+1]
loop_body_unindented = [line[8:] if line.startswith('        ') else (line[4:] if line.startswith('    ') else line) for line in loop_body]

func_def = [
    'def run_single_dataset_pipeline(X, y, problem_type, store, encoder, did="local"):\n',
    '    DEBUG = False\n',
    '    RUN_WEIGHT_SENSITIVITY = False\n',
    '    from wandb_logger import log\n',
    '    all_query_vecs = {} # Dummy\n'
] + ['    ' + line for line in loop_body_unindented] + [
    '    return similarity, threshold, decision, selected_models, cs_score, full_score, score_gap, models_saved, c_sim, c_cons, c_agree\n'
]

# 4. Insert conditional logic for local dataset before the loop
local_dataset_logic = [
    '    # Check if this is a LOCAL dataset run\n',
    '    if "csv_path" in config and config["csv_path"]:\n',
    '        print(f"\\n🚀 Processing LOCAL dataset: {config[\'csv_path\']}")\n',
    '        X, y, problem_type = load_local_dataset(\n',
    '            config["csv_path"], \n',
    '            config["target_column"]\n',
    '        )\n',
    '        if X is not None:\n',
    '            run_single_dataset_pipeline(X, y, problem_type, store, encoder)\n',
    '            print("\\n✅ Local dataset processing complete!")\n',
    '            return  # Exit after processing local dataset\n',
    '        else:\n',
    '            print("\\n❌ Local dataset failed to load.")\n',
    '            return\n\n',
    '    print("\\n📊 Running OpenML Benchmark Suite...")\n'
]

call_str = [
    '        try:\n',
    '            similarity, threshold, decision, selected_models, cs_score, full_score, score_gap, models_saved, c_sim, c_cons, c_agree = run_single_dataset_pipeline(X, y, problem_type, store, encoder, did=did)\n',
    '        except Exception as e:\n',
    '            print(f"Pipeline failed for {did}: {e}")\n',
    '            continue\n'
]

new_lines = lines[:main_idx] + func_def + ['\n'] + lines[main_idx:start_loop_idx] + local_dataset_logic + lines[start_loop_idx:start_loop_idx+10] + call_str + lines[end_loop_idx+1:]

with open('phase4_pipeline.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
