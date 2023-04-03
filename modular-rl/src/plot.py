import matplotlib.pyplot as plt
import pickle
import numpy as np
import os


methods = ['ConsistentMLP', 'MLP', 'amorpheus']
colors = ['blue', 'orange', 'green']
envs = ['humanoid', 'hopper', 'walker']
seeds = [0, 1, 2]

# plot separate figures for each environment (morphology type)
for env in envs:
    all_results = dict()
    for method in methods:
        all_results[method] = dict()
        for seed in seeds:
            if f'{method}_{env}_{seed}.pkl' not in os.listdir('results/eval'):
                continue
            with open(f'results/eval/{method}_{env}_{seed}.pkl', 'rb') as f:
                results = pickle.load(f)
            for morphology in results:
                if morphology not in all_results[method]:
                    all_results[method][morphology] = np.zeros([3, len(results[morphology].keys())])
                for model_idx in results[morphology]:
                    try:
                        all_results[method][morphology][seed, int(model_idx.split('_')[1])] = results[morphology][model_idx]
                    except:
                        pass
    
    for morphology in all_results['MLP']:
        plt.figure()
        for method, color in zip(methods, colors):
            # if method == 'amorpheus':
            #     all_results[method][morphology] = all_results[method][morphology][:2]
            mean = all_results[method][morphology].mean(0)
            std = all_results[method][morphology].std(0)
            # plt.plot(mean, label=method)
            # plt.fill_between(np.arange(mean.shape[0]), mean - std, mean + std, alpha=0.25)
            for seed in seeds:
                plt.plot(all_results[method][morphology][seed], label=method, c=color)
        plt.legend()
        plt.savefig(f'figures/{morphology}.png')
        plt.close()
    
    plt.figure()
    for method, color in zip(methods, colors):
        all_morphology_results = np.stack([all_results[method][morphology] for morphology in all_results[method]]).mean(0)
        mean = all_morphology_results.mean(0)
        std = all_morphology_results.std(0)
        # plt.plot(mean, label=method)
        # plt.fill_between(np.arange(mean.shape[0]), mean - std, mean + std, alpha=0.25)
        for seed in seeds:
            plt.plot(all_morphology_results[seed], label=method, c=color)
    plt.legend()
    plt.savefig(f'figures/{env}_avg.png')
    plt.close()
