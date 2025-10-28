import os, random
from graphxai.utils import Explanation
import torch

from graphxai.datasets.real_world.extract_google_datasets import load_graphs 
from graphxai.datasets import GraphDataset
import numpy as np


from torch_geometric.data import Data
# fc_data_dir = os.path.join(os.path.dirname(__file__), 'ac_data')
# fc_smiles_df = 'AC_smiles.csv'

ba_datapath = os.path.join(os.path.dirname(__file__), 'BA-3motif.npy')


    
process_path = os.path.join(os.path.dirname(__file__), 'BA-3motif-p.npy')
def pre_process(data_path):
    edge_index_list, label_list, ground_truth_list, role_id_list, pos = np.load(ba_datapath, allow_pickle=True)
    #print(edge_index_list)
    length = label_list.shape[0]
    num_index = [i for i in range(length)]
    random.shuffle(num_index)
    new_edge_index_list = []
    new_label_list = []
    new_ground_truth_list = []
    new_role_id_list = []
    new_pos = []
    for i in range(length):
        new_edge_index_list.append(edge_index_list[num_index[i]])
        new_label_list.append(label_list[num_index[i]])
        new_ground_truth_list.append(ground_truth_list[num_index[i]])
        new_role_id_list.append(role_id_list[num_index[i]])
        new_pos.append(pos[num_index[i]])
    out=[new_edge_index_list,new_label_list,new_ground_truth_list,new_role_id_list,new_pos]
    np.save(process_path,arr=np.array(out,dtype=object),allow_pickle=True)
    """
    np.savez(out_path,
                 arr_0=np.array(new_edge_index_list),
                 arr_1=np.array(new_label_list),
                 arr_2=np.array(new_ground_truth_list),
                 arr_3=np.array(new_role_id_list),
                 arr_4=np.array(new_pos),
                 allow_pickle=True)
    """
#pre_process(ba_datapath)

class BA3Motif(GraphDataset):

    def __init__(
            self,
            split_sizes = (0.7, 0.2, 0.1),
            seed = None,
            data_path: str = ba_datapath,
            device = None,
            downsample = True,
            downsample_seed = None
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




        edge_index_list, label_list, ground_truth_list, role_id_list, pos = np.load(process_path, allow_pickle=True)
        
        self.graphs = []
        self.explanations = []
        alpha = 0.25
        ys=[]
        num_index= []
        for idx, (edge_index, y, ground_truth, z, p) in enumerate(zip(edge_index_list, label_list, ground_truth_list, role_id_list, pos)):
            num_index.append(idx)
            edge_index = torch.from_numpy(edge_index)
            new_edge_index = []
            new_ground = []
            for i in range(edge_index.size(1)):
                new_edge_index.append([edge_index[0,i],edge_index[1,i]])
                new_edge_index.append([edge_index[1,i],edge_index[0,i]])
                new_ground.append(ground_truth[i])
                new_ground.append(ground_truth[i])
                
            edge_index = torch.tensor(new_edge_index).transpose(1,0)
            node_idx = torch.unique(edge_index)
            assert node_idx.max() == node_idx.size(0) - 1
            x = torch.zeros(node_idx.size(0), 4)
            index = [i for i in range(node_idx.size(0))]
            x[index, z] = 1
            x = alpha * x + (1 - alpha) * torch.rand((node_idx.size(0), 4))
            edge_attr = torch.ones(edge_index.size(1), 1)
            y = torch.tensor(int(y), dtype=torch.torch.long).unsqueeze(dim=0)
            #ys.append(y)
            if y not in ys:
                ys.append(y)
            # fix bug for torch > 1.6
            p = np.array(list(p.values())) 
            new_ground=torch.tensor(new_ground)
            if idx==0:
                print(y)
                print(edge_index)
                print(x)
                print(new_ground)
                print("")
            
            data = Data(
                x = x,
                y = y,
                edge_attr = edge_attr,
                edge_index = edge_index
            )
            self.graphs.append(data)

            exp = Explanation(
                feature_imp = None, # No feature importance - everything is one-hot encoded
                node_imp = None,
                edge_imp = new_ground,
            )
            
            exp.set_whole_graph(data)
            
            self.explanations.append([exp])
        #print(index)
    
        new_graphs=[]
        new_exp=[]
        #random.shuffle(num_index)
        for idx in num_index:
            new_graphs.append(self.graphs[idx])
            new_exp.append(self.explanations[idx])
        #print(num_index)
        self.graphs=new_graphs
        self.explanations=new_exp
        super().__init__(name = 'BA3Motif', seed = seed, split_sizes = split_sizes, device = device)