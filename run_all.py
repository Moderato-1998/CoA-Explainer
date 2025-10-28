import os
import itertools
import subprocess
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
root = r'd:\OneDrive\1-Code\GNN_Exp\my model\Multi_Agent_Reinforcement_Learning_Explainer\25-1004'
os.chdir(root)

py = r'C:\Users\liyon\anaconda3\envs\pyg_2.6.1_graphxai\python.exe'
script = os.path.join('results','1_explanation_accuracy','run_rc_explainer.py')   # run_gnnex_subx, run_coa_explainer

datasets = ['Benzene', 'FluorideCarbonyl', 'AlkaneCarbonyl', 'Mutagenicity', 'house_triangle', 'grid_triangle', 'house_cycle']
models   = ['GIN_3layer']   # 'GAT_2layer', 'GIN_3layer','GCN_3layer'
methods  = ['rcex']   # 'gnnex','subx', 'coaex', 'pgex', 'rcex'

# 如需过滤某些组合，可在这里加 if 判断
for d, m, e in itertools.product(datasets, models, methods):
    tag = f'{d}_{m}_{e}'
    print(f'==> Running {tag}')
    logdir = os.path.join('results','1_explanation_accuracy','logs')
    os.makedirs(logdir, exist_ok=True)
    logfile = os.path.join(logdir, f'{tag}.log')

    cmd = [py, script, '--dataset_name', d, '--model_type', m, '--exp_name', e]
    with open(logfile, 'w', encoding='utf-8') as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            print(f'  [!] Failed: {tag} (see {logfile})')
        else:
            print(f'  [OK] Done: {tag} -> {logfile}')

