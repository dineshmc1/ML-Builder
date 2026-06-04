import sys
with open('phase4_pipeline.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# find def main():
for i, line in enumerate(lines):
    if line.startswith('def main():'):
        main_idx = i
        break

# find the loop inside main
start_loop_idx = -1
end_loop_idx = -1
for i in range(main_idx, len(lines)):
    if 'for did in test_ids:' in lines[i]:
        start_loop_idx = i
        break

# find the end of the try block for report generator
for i in range(start_loop_idx, len(lines)):
    if 'print(f"  [Phase 5.6 Report] Failed: {e}")' in lines[i]:
        end_loop_idx = i + 2  # include traceback.print_exc()
        break

loop_body = lines[start_loop_idx+10:end_loop_idx+1] # start from # 1. Extract meta-features
# adjust indentation (remove 8 spaces)
loop_body_unindented = [line[8:] if line.startswith('        ') else (line[4:] if line.startswith('    ') else line) for line in loop_body]

func_def = [
    'def run_single_dataset_pipeline(X, y, problem_type, store, encoder, did="local"):\n',
    '    DEBUG = False\n',
    '    from wandb_logger import log\n',
    '    all_query_vecs = {} # Dummy\n'
] + ['    ' + line for line in loop_body_unindented] + [
    '    return similarity, threshold, decision, selected_models, cs_score, full_score, score_gap, models_saved, c_sim, c_cons, c_agree\n'
]

# replace loop body in main with a function call
call_str = [
    '        try:\n',
    '            similarity, threshold, decision, selected_models, cs_score, full_score, score_gap, models_saved, c_sim, c_cons, c_agree = run_single_dataset_pipeline(X, y, problem_type, store, encoder, did=did)\n',
    '        except Exception as e:\n',
    '            print(f"Pipeline failed for {did}: {e}")\n',
    '            continue\n'
]

new_lines = lines[:main_idx] + func_def + ['\n'] + lines[main_idx:start_loop_idx+10] + call_str + lines[end_loop_idx+1:]

with open('phase4_pipeline_new.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
