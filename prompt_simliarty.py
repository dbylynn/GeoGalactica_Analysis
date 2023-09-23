import torch
import json
import argparse
import numpy as np
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM

def get_embed(sentence):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    inputs['input_ids'] = inputs['input_ids'][:,:512]
    inputs['token_type_ids'] = inputs['token_type_ids'][:,:512]
    inputs['attention_mask'] = inputs['attention_mask'][:,:512]

    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # 第一个句子的嵌入表示
    # cls_embedding = outputs[0][:, 0, :]  # 取第一个位置的[CLS]嵌入
    # pool_embedding = outputs.pooler_output
    return embeddings
def get_categorys(categorys):
    categorys_index = []
    categorys_embed = []
    for k, v in categorys.items():
        cur_embeds = []
        cur_strs = []
        for item in v[:10]:
            cur_str = item['instruction'] + item['context'] + item['response']
            cur_strs.append(cur_str)
        cur_embeds = get_embed(cur_strs)

        mean_embedding = torch.mean(cur_embeds, dim=0)
        categorys_embed.append(mean_embedding)
        categorys_index.append(k)
    return categorys_embed, categorys_index

def cal_similarity(categorys_embed, categorys_index):
    similarity_matrix = np.zeros((len(categorys_index), len(categorys_index)))

    for i in range(len(categorys_embed)):
        for j in range(i, len(categorys_embed)):
            similarity = F.cosine_similarity(categorys_embed[i].unsqueeze(0), categorys_embed[j].unsqueeze(0), dim=1)
            similarity_matrix[i, j] = similarity.item()
            similarity_matrix[j, i] = similarity.item()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default='./datas/databricks-dolly-15k.jsonl', type=str)
    args = parser.parse_args()

    with open(args.prompt, 'r') as file:
        lines = file.readlines()
    data = []
    for line in lines:
        d = json.loads(line)
        data.append(d)
        
    categorys = {}
    for d in data:
        if d['category'] not in categorys:
            categorys[d['category']] = [d]
        else:
            categorys[d['category']].append(d)
    
    
    categorys_embed, categorys_index = get_categorys(categorys)
    similarity_matrix = cal_similarity(categorys_embed, categorys_index)

    with open('./output/categorys_index_dolly.txt', 'w') as file:
        for item in categorys_index:
            file.write(item + "\n")

    np.save('./output/similarity_matrix_dolly.npy', similarity_matrix)
