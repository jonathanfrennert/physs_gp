import os
import json
import pickle
import csv

results = []

# Collect data from each run directory
for dirpath in os.listdir('runs/'):
    run_dirpath = os.path.join('runs/', dirpath)
    if dirpath.endswith('_sources') or not os.path.isdir(run_dirpath):
        continue

    config_fp = os.path.join(run_dirpath, 'config.json')
    if not os.path.exists(config_fp):
        continue

    run_fns = os.listdir(run_dirpath)
    result_files = [x for x in run_fns if x.endswith('.pickle')]
    if not result_files:
        continue

    result_fp = os.path.join(run_dirpath, result_files[0])

    with open(config_fp, 'r') as f:
        run_config = json.load(f)

    model = run_config['model']
    kernel = run_config['kernel']
    whiten = run_config['whiten']
    num_colocation = run_config['num_colocation']

    with open(result_fp, 'rb') as f:
        run_result = pickle.load(f)

    rmse = run_result['metrics']['test_0']['rmse']
    nlpd = run_result['metrics']['test_0_callback']['test_0_nlpd']
    time = run_result['meta']['training_time']

    results.append({
        'model': model,
        'kernel': kernel,
        'whiten': whiten,
        'C': num_colocation,
        'time': f"{time:.2f}",
        'rmse': f"{rmse:.2f}",
        'nlpd': f"{nlpd:.2f}"
    })

# Write to CSV
output_csv = 'run_results.csv'
with open(output_csv, 'w', newline='') as csvfile:
    fieldnames = ['model', 'kernel', 'whiten', 'C', 'time', 'rmse', 'nlpd']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(results)

print(f"Results written to {output_csv}")
