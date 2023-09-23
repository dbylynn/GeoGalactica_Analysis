import matplotlib.pyplot as plt
import torch
import numpy as np
import argparse
from transformers import AutoModel, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM

def get_galactica_weight(model, layer): 
    q0 = model.state_dict()[f'model.decoder.layers.{layer}.self_attn.q_proj.weight'].flatten()
    k0 = model.state_dict()[f'model.decoder.layers.{layer}.self_attn.k_proj.weight'].flatten()
    v0 = model.state_dict()[f'model.decoder.layers.{layer}.self_attn.v_proj.weight'].flatten()
    qkv0 = torch.stack([q0, k0, v0], dim=0).mean(dim=0)
    out = model.state_dict()[f'model.decoder.layers.{layer}.self_attn.out_proj.weight'].flatten()
    fc1 = model.state_dict()[f'model.decoder.layers.{layer}.fc1.weight'].flatten()
    fc2 = model.state_dict()[f'model.decoder.layers.{layer}.fc2.weight'].flatten()
    return qkv0, out, fc1, fc2

def get_llama_weight(model, layer):
    q0 = model.state_dict()[f'model.layers.{layer}.self_attn.q_proj.weight'].flatten()
    k0 = model.state_dict()[f'model.layers.{layer}.self_attn.k_proj.weight'].flatten()
    v0 = model.state_dict()[f'model.layers.{layer}.self_attn.v_proj.weight'].flatten()

    qkv0 = torch.stack([q0, k0, v0], dim=0).mean(dim=0)
    dense = model.state_dict()[f'model.layers.{layer}.self_attn.o_proj.weight'].flatten()

    gate = model.state_dict()[f'model.layers.{layer}.mlp.gate_proj.weight'].flatten()
    up = model.state_dict()[f'model.layers.{layer}.mlp.up_proj.weight'].flatten()
    down = model.state_dict()[f'model.layers.{layer}.mlp.down_proj.weight'].flatten()
    mlp = torch.stack([gate, up, down], dim=0).mean(dim=0)

    return qkv0, dense, mlp

def save_llama_weight(model, layer):
    qkv, dense, mlp = get_llama_weight(model, layer)
    model_name = "llama"

    exp_name = "qkv"
    save_path = f'./weight_npy/{model_name}_{exp_name}_{layer}.npy'
    np.save(save_path, qkv)
    print(f"Aready save qkv in path: {save_path}")

    exp_name = "dense"
    save_path = f'./weight_npy/{model_name}_{exp_name}_{layer}.npy'
    np.save(save_path, dense)
    print(f"Aready save dense in path: {save_path}")


    exp_name = "mlp"
    save_path = f'./weight_npy/{model_name}_{exp_name}_{layer}.npy'
    np.save(save_path, mlp)
    print(f"Aready save mlp in path: {save_path}")

    
def save_galactica_weight(model, layer):
    qkv, out, fc1, fc2 = get_galactica_weight(model, layer)

    model_name = "galactica"
    exp_name = "qkv"
    save_path = f'./weight_npy/{model_name}_{exp_name}_{layer}.npy'
    np.save(save_path, qkv)
    print(f"Aready save qkv in path: {save_path}")


    exp_name = "out"
    save_path = f'./weight_npy/{model_name}_{exp_name}_{layer}.npy'
    np.save(save_path, out)
    print(f"Aready save out in path: {save_path}")


    fc = np.concatenate((fc1.reshape(1,-1), fc2.reshape(1,-1)), axis=0)
    fc = np.mean(fc, axis=0).reshape(-1)
    exp_name = "fc"
    save_path = f'./weight_npy/{model_name}_{exp_name}_{layer}.npy'
    np.save(save_path, fc)
    print(f"Aready save fc in path: {save_path}")



def plot_hist(data, save_path, exp_name):
    # bin_width = 0.05
    # num_bins = int((max(data) - min(data)) / bin_width)

    plt.bar(data, alpha=0.8, log=True)
    plt.title(exp_name)
    plt.xlim(-1, 0, 1)
    plt.xticks([-1, 0, 1])
    plt.yticks([10, 1000, 100000, 10000000])

    plt.savefig(save_path)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lla_model_path", default='/home/daven/research/GeoLLaMA-PT/weights/llama-7b', type=str)
    parser.add_argument("--gla_model_path", default='/home/hantek/galactica-30b/galactica-30b', type=str)
    parser.add_argument("--layer", default=0, type=int)

    args = parser.parse_args()

    layer = args.layer

    model_path = ""
    lla_model = AutoModelForCausalLM.from_pretrained(args.lla_model_path)
    model_name = "llama"
    save_llama_weight(lla_model, args.layer)

    gla_model = AutoModelForCausalLM.from_pretrained(args.gla_model_path)
    model_name = "galactica"
    save_galactica_weight(gla_model, args.layer)

