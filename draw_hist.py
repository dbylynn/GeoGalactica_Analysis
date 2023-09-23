import numpy as np
import argparse
import matplotlib.pyplot as plt


orange = (214/225, 169/225, 101/225)
blue = (112/225, 157/225, 198/225)

def plot_hist(data1, data2, exp_name):
    fig, ax = plt.subplots(1)
    min_value = min(data1.min(), data2.min())
    max_value = max(data1.max(), data2.max())
    num_bins = 50  # 或者根据需要指定bin的数量
    bin_edges = np.linspace(min_value, max_value, num_bins + 1)

    value = [10, 1000, 100000, 10000000]

    ax.set_title(exp_name)
    ax.set_xlim(-1, 0, 1)
    ax.set_xticks([-1, 0, 1])

    bar_width = 0.05

    bin_edges1, _, patches1 = ax.hist(data1, bins=bin_edges, alpha=1.0, color=blue, label='Histogram 1',log=True)
    bin_edges2, _, patches2 = ax.hist(data2, bins=bin_edges, alpha=0.8, color=orange, label='Histogram 2',log=True)

    ax.set_yticks([10, 1000, 100000, 10000000])
    plt.savefig('./output/weight_hist.png')

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight_type", default='qkv', type=str)
    parser.add_argument("--layer", default=0, type=int)
    args = parser.parse_args()
    
    weight_type = args.weight_type

    qkv0 = np.load(f"./output/galactica_qkv_{layer}.npy")
    out = np.load(f"./output/galactica_out_{layer}.npy")
    fc1 = np.load(f"./output/galactica_fc1_{layer}.npy")
    fc2 = np.load(f"./output/galactica_fc2_{layer}.npy")
    mlp = np.load(f"./output/llama_mlp_{layer}.npy")
    qkv = np.load(f"./output/llama_qkv_{layer}.npy")
    dense = np.load(f"./output/llama_dense_{layer}.npy")

    fc = np.concatenate((fc1.reshape(1,-1), fc2.reshape(1,-1)), axis=0)
    fc = np.mean(fc, axis=0).reshape(-1)

    exp_name = f'{weight_type}_{layer}'
    if weight_type == 'qkv':
        plot_hist(qkv, qkv0, exp_name)

    elif weight_type == 'out':
        plot_hist(dense, out, exp_name)

    elif weight_type == 'fc'
        plot_hist(mlp, fc, exp_name)



