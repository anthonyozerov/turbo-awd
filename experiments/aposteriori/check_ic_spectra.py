import os
import yaml
from turboawd.utils import load_data
from turboawd.aposteriori import load_online_data, aposteriori
import matplotlib.pyplot as plt

with open('../apriori/data_defs.yaml', 'r') as f:
    data_defs = yaml.safe_load(f)

# IC's:
dns_runs = {
    'A': '2025-03-01-a-filtered',
    'B': '2025-03-01-b-filtered',
    'C': '2025-03-01-c-filtered'
}

for k, v in data_defs.items():
    print(k)
    if k not in dns_runs:
        continue

    omegas_true = load_data(os.path.join(v['root'],v['input']), keys=['omega']).squeeze()
    omegas_fdns, _ = load_online_data('../online/results/'+dns_runs[k]+'/data')
    omegas_fdns = omegas_fdns[50:]

    result_true = aposteriori(omegas_true)
    result_fdns = aposteriori(omegas_fdns)

    which = 'tke_spectrum'

    plt.loglog(result_true[which], color='black', linestyle='-', label='Old')
    plt.loglog(result_fdns[which], color='red', linestyle='--', label='New')

    plt.legend()
    plt.show()