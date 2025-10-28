import torch
from graphxai.datasets import Benzene
from graphxai.visualization import visualize_edge_explanation
from data.utils import print_dataset_info, get_dataset

seed = 42
device = "cuda" if torch.cuda.is_available() else "cpu"

dataset_name = 'Benzene'    # Benzene, FluorideCarbonyl, AlkaneCarbonyl, Mutagenicity, house_triangle, triangle_grid, house_cycle
dataset = get_dataset(dataset_name, seed, device)


ave_exp_size = []
for exp in dataset.explanations:
    if len(exp) > 0:
        exp_size = sum(exp[-1].edge_imp)
        ave_exp_size.append(exp_size)
ave_exp_size = sum(ave_exp_size)/len(ave_exp_size)
print(ave_exp_size)


# print(f'dataset.train_index: {dataset.train_index}')
# print(f'dataset.val_index: {dataset.val_index}')
# print(f'dataset.test_index: {dataset.test_index}')




# for i in dataset.train_index:
#     data, exp = dataset[i]
#     # print(f'data.x: {data.x}')
#     # print(f'data.edge_index: {data.edge_index}')
#     # print(f'data.y: {data.y}')
#     print(data.num_edges)
#     for e in exp:
#         print(f'e.edge_imp: {e.edge_imp}')
#         print(f'e.node_imp: {e.node_imp}')
#         e.visualize_graph(show=True)
#     # break