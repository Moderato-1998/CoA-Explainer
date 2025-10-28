import os
import random
from graphxai.utils import Explanation
import torch

from graphxai.datasets.real_world.extract_google_datasets import load_graphs
from graphxai.datasets import GraphDataset
import numpy as np


from torch_geometric.data import Data
# fc_data_dir = os.path.join(os.path.dirname(__file__), 'ac_data')
# fc_smiles_df = 'AC_smiles.csv'


class BAHouse(GraphDataset):

    def __init__(
        self,
        split_sizes=(0.7, 0.2, 0.1),
        seed=None,
        device=None,
        downsample=True,
        downsample_seed=None,
        shuffle=True
    ):
        '''
        Args:
            split_sizes (tuple): 
            seed (int, optional):
            data_path (str, optional):
        '''

        self.device = device
        self.downsample = downsample
        self.downsample_seed = downsample_seed

        process_path = os.path.join(os.path.dirname(__file__), 'BA-House.npy')
        # process_path = os.path.join(os.path.dirname(__file__), 'BA-NewHouse-2.npy')
        edge_index_list, feature_list, label_list, ground_truth_list = np.load(process_path, allow_pickle=True)
        Vs = 0
        Es = 0
        GTs = 0
        count = 0
        self.graphs = []
        self.explanations = []
        for idx, (edge_index, x, y, ground_truth) in enumerate(zip(edge_index_list, feature_list, label_list, ground_truth_list)):
            # print(y)
            # print(x)
            count += 1
            Vs += edge_index.size(1)/2
            Es += x.shape[0]
            GTs += y
            # edge_index = torch.from_numpy(edge_index)
            node_idx = torch.unique(edge_index)
            assert node_idx.max() == node_idx.size(0) - 1
            edge_attr = torch.ones(edge_index.size(1), 1)
            y = torch.tensor(int(y), dtype=torch.torch.long).unsqueeze(dim=0)
            # ys.append(y)
            # fix bug for torch > 1.6
            # if idx == 0:
            #     print(y)
            #     print(edge_index)
            #     print(x)
            #     print(ground_truth)
            #     print("")
            # num_index.append(idx)
            data = Data(
                x=x,
                y=y,
                edge_attr=edge_attr,
                edge_index=edge_index
            )
            self.graphs.append(data)

            exp = Explanation(
                feature_imp=None,  # No feature importance - everything is one-hot encoded
                node_imp=None,
                edge_imp=ground_truth,
            )

            exp.set_whole_graph(data)

            self.explanations.append([exp])
        # print(index)

        # print(Vs/count)
        # print(Es/count)
        # print(GTs)
        # new_graphs=[]
        # new_exp=[]
        # random.shuffle(num_index)
        # for idx in num_index:
        #    new_graphs.append(self.graphs[idx])
        #    new_exp.append(self.explanations[idx])
        # print(num_index)
        # self.graphs=new_graphs
        # self.explanations=new_exp
        super().__init__(name='BAHouse', seed=seed, split_sizes=split_sizes, device=device, shuffle=shuffle)
# BAHouse()


class BADiamond(GraphDataset):

    def __init__(
        self,
        split_sizes=(0.7, 0.2, 0.1),
        seed=None,
        device=None,
        downsample=True,
        downsample_seed=None,
        shuffle=True
    ):
        '''
        Args:
            split_sizes (tuple): 
            seed (int, optional):
            data_path (str, optional):
        '''

        self.device = device
        self.downsample = downsample
        self.downsample_seed = downsample_seed

        process_path = os.path.join(os.path.dirname(__file__), 'BA-Diamond.npy')
        edge_index_list, feature_list, label_list, ground_truth_list = np.load(process_path, allow_pickle=True)
        Vs = 0
        Es = 0
        GTs = 0
        count = 0
        self.graphs = []
        self.explanations = []
        for idx, (edge_index, x, y, ground_truth) in enumerate(zip(edge_index_list, feature_list, label_list, ground_truth_list)):
            # print(edge_index)
            count += 1
            Vs += edge_index.size(1)/2
            Es += x.shape[0]
            GTs += y
            # edge_index = torch.from_numpy(edge_index)
            node_idx = torch.unique(edge_index)
            assert node_idx.max() == node_idx.size(0) - 1
            edge_attr = torch.ones(edge_index.size(1), 1)
            y = torch.tensor(int(y), dtype=torch.torch.long).unsqueeze(dim=0)
            # ys.append(y)
            # fix bug for torch > 1.6
            # if idx == 0:
            #     print(y)
            #     print(edge_index)
            #     print(x)
            #     print(ground_truth)
            #     print("")
            # num_index.append(idx)
            data = Data(
                x=x,
                y=y,
                edge_attr=edge_attr,
                edge_index=edge_index
            )
            self.graphs.append(data)

            exp = Explanation(
                feature_imp=None,  # No feature importance - everything is one-hot encoded
                node_imp=None,
                edge_imp=ground_truth,
            )

            exp.set_whole_graph(data)

            self.explanations.append([exp])
        # print(index)
        # print(Vs/count)
        # print(Es/count)
        # print(GTs)

        # new_graphs=[]
        # new_exp=[]
        # random.shuffle(num_index)
        # for idx in num_index:
        #    new_graphs.append(self.graphs[idx])
        #    new_exp.append(self.explanations[idx])
        # print(num_index)
        # self.graphs=new_graphs
        # self.explanations=new_exp
        super().__init__(name='BADiamond', seed=seed, split_sizes=split_sizes, device=device, shuffle=shuffle)


class BAWheel(GraphDataset):

    def __init__(
        self,
        split_sizes=(0.7, 0.2, 0.1),
        seed=None,
        device=None,
        downsample=True,
        downsample_seed=None,
        shuffle=True
    ):
        '''
        Args:
            split_sizes (tuple): 
            seed (int, optional):
            data_path (str, optional):
        '''

        self.device = device
        self.downsample = downsample
        self.downsample_seed = downsample_seed

        process_path = os.path.join(os.path.dirname(__file__), 'BA-Wheel.npy')
        edge_index_list, feature_list, label_list, ground_truth_list = np.load(process_path, allow_pickle=True)
        Vs = 0
        Es = 0
        GTs = 0
        count = 0
        self.graphs = []
        self.explanations = []
        for idx, (edge_index, x, y, ground_truth) in enumerate(zip(edge_index_list, feature_list, label_list, ground_truth_list)):
            # print(edge_index)
            count += 1
            Vs += edge_index.size(1)/2
            Es += x.shape[0]
            GTs += y
            # edge_index = torch.from_numpy(edge_index)
            node_idx = torch.unique(edge_index)
            assert node_idx.max() == node_idx.size(0) - 1
            edge_attr = torch.ones(edge_index.size(1), 1)
            y = torch.tensor(int(y), dtype=torch.torch.long).unsqueeze(dim=0)
            # ys.append(y)
            # fix bug for torch > 1.6
            # if idx == 0:
            #     print(y)
            #     print(edge_index)
            #     print(x)
            #     print(ground_truth)
            #     print("")
            # num_index.append(idx)
            data = Data(
                x=x,
                y=y,
                edge_attr=edge_attr,
                edge_index=edge_index
            )
            self.graphs.append(data)

            exp = Explanation(
                feature_imp=None,  # No feature importance - everything is one-hot encoded
                node_imp=None,
                edge_imp=ground_truth,
            )

            exp.set_whole_graph(data)

            self.explanations.append([exp])
        # print(index)
        # print(Vs/count)
        # print(Es/count)
        # print(GTs)
        # new_graphs=[]
        # new_exp=[]
        # random.shuffle(num_index)
        # for idx in num_index:
        #    new_graphs.append(self.graphs[idx])
        #    new_exp.append(self.explanations[idx])
        # print(num_index)
        # self.graphs=new_graphs
        # self.explanations=new_exp
        super().__init__(name='BAWheel', seed=seed, split_sizes=split_sizes, device=device, shuffle=shuffle)


class BACycle(GraphDataset):

    def __init__(
        self,
        split_sizes=(0.7, 0.2, 0.1),
        seed=None,
        device=None,
        downsample=True,
        downsample_seed=None,
        shuffle=True
    ):
        '''
        Args:
            split_sizes (tuple): 
            seed (int, optional):
            data_path (str, optional):
        '''

        self.device = device
        self.downsample = downsample
        self.downsample_seed = downsample_seed

        process_path = os.path.join(os.path.dirname(__file__), 'BA-Cycle.npy')
        edge_index_list, feature_list, label_list, ground_truth_list = np.load(process_path, allow_pickle=True)

        self.graphs = []
        self.explanations = []
        for idx, (edge_index, x, y, ground_truth) in enumerate(zip(edge_index_list, feature_list, label_list, ground_truth_list)):
            # print(edge_index)
            # edge_index = torch.from_numpy(edge_index)
            node_idx = torch.unique(edge_index)
            assert node_idx.max() == node_idx.size(0) - 1
            edge_attr = torch.ones(edge_index.size(1), 1)
            y = torch.tensor(int(y), dtype=torch.torch.long).unsqueeze(dim=0)
            # ys.append(y)
            # fix bug for torch > 1.6
            # num_index.append(idx)
            data = Data(
                x=x,
                y=y,
                edge_attr=edge_attr,
                edge_index=edge_index
            )
            self.graphs.append(data)

            exp = Explanation(
                feature_imp=None,  # No feature importance - everything is one-hot encoded
                node_imp=None,
                edge_imp=ground_truth,
            )

            exp.set_whole_graph(data)

            self.explanations.append([exp])
        # print(index)

        # new_graphs=[]
        # new_exp=[]
        # random.shuffle(num_index)
        # for idx in num_index:
        #    new_graphs.append(self.graphs[idx])
        #    new_exp.append(self.explanations[idx])
        # print(num_index)
        # self.graphs=new_graphs
        # self.explanations=new_exp
        super().__init__(name='BACycle', seed=seed, split_sizes=split_sizes, device=device, shuffle=shuffle)
