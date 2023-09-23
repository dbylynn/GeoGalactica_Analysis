import argparser
import numpy as np
import matplotlib.pyplot as plt


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--similarity_matrix", default='./similarity_matrix_dolly.npy', type=str)
    args = parser.parse_args()

    similarity_matrix = np.load(args.similarity_matrix)
    with open(args.similarity_matrix, "r") as file:
        categorys_index = [line.strip() for line in file]

    plt.figure(figsize=(8, 6))
    plt.imshow(similarity_matrix, cmap='magma', interpolation='nearest')
    plt.colorbar()  
    plt.title('Dolly')
    plt.xticks(range(len(categorys_index)), categorys_index, rotation='vertical', fontname='Times New Roman', fontsize=15)
    plt.yticks(range(len(categorys_index)), categorys_index, fontname='Times New Roman', fontsize=15)
    plt.savefig('./output/heat_map.png')
