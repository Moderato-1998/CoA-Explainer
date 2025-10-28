from re import T
import torch
import os
from torch_geometric.datasets import ExplainerDataset
from torch_geometric.datasets.graph_generator import BAGraph
from torch_geometric.datasets.motif_generator import HouseMotif, CycleMotif, GridMotif
from torch_geometric.data import Data

from torch_geometric.datasets import BA2MotifDataset, BAMultiShapesDataset

from graphxai.datasets import GraphDataset
from graphxai.utils import Explanation
import networkx as nx
from graphxai.datasets.utils.feature_generators import gaussian_lv_generator
import random


class BA2Motif(GraphDataset):
    def __init__(
        self,
        split_sizes=(0.7, 0.2, 0.1),
        seed=None,
        data_path: str = None,
        device=None,
        shuffle=True,
        shape1 = 'house',
        shape2 = 'cycle',
    ):
        '''
        Args:
            split_sizes (tuple): 
            seed (int, optional):
            data_path (str, optional):
        '''

        self.device = device
        self.graphs = []
        self.explanations = []

        # --- Disk cache: try load graphs/explanations to avoid regeneration ---
        # Default cache dir: <this_folder>/processed unless data_path provided
        cache_dir = data_path if data_path is not None else os.path.join(os.path.dirname(__file__), 'processed')
        os.makedirs(cache_dir, exist_ok=True)

        # Cache key is based on shapes and generation hyperparams we fix in this class
        # 由于我们在生成时让 BA 图的“每个新节点的边数 m”随机，缓存文件名改为 eVar_m1to5 以避免误用旧的固定 e1 缓存
        cache_file = os.path.join(
            cache_dir,
            f"ba2motif_{shape1}_{shape2}_n20_eVar_m1to5_500each.pt"
        )

        loaded_from_cache = False
        if os.path.isfile(cache_file):
            try:
                cache_obj = torch.load(cache_file, map_location='cpu')
                graphs_cached = cache_obj.get('graphs')
                exps_cached = cache_obj.get('explanations')
                if isinstance(graphs_cached, list) and isinstance(exps_cached, list) and len(graphs_cached) == 1000:
                    self.graphs = graphs_cached
                    self.explanations = exps_cached
                    loaded_from_cache = True
            except Exception:
                # Ignore cache errors and fall back to regeneration
                loaded_from_cache = False

        if loaded_from_cache:
            super().__init__(name=f'{shape1}_{shape2}', seed=seed, split_sizes=split_sizes, device=device, shuffle=shuffle)
            return

        if shape1 == 'house':
            motif1 = HouseMotif()
        elif shape1 == 'cycle':
            motif1 = CycleMotif(5)
        elif shape1 == 'triangle':
            motif1 = CycleMotif(3)
        elif shape1 == 'grid':
            motif1 = GridMotif()

        if shape2 == 'house':
            motif2 = HouseMotif()
        elif shape2 == 'cycle':
            motif2 = CycleMotif(5)
        elif shape2 == 'triangle':
            motif2 = CycleMotif(3)
        elif shape2 == 'grid':
            motif2 = GridMotif()
        

        # --- helpers: quality check & on-demand regeneration until target count ---
        def is_connected_no_isolates(data_item) -> bool:
            try:
                num_nodes = int(data_item.num_nodes) if getattr(data_item, 'num_nodes', None) is not None else int(data_item.edge_index.max().item()) + 1
            except Exception:
                num_nodes = int(data_item.edge_index.max().item()) + 1

            # Build undirected simple graph and ensure all nodes are present
            G = nx.Graph()
            G.add_nodes_from(range(num_nodes))
            ei = data_item.edge_index
            for i in range(ei.size(1)):
                u = int(ei[0, i].item()); v = int(ei[1, i].item())
                if u == v:
                    continue
                G.add_edge(min(u, v), max(u, v))

            if num_nodes == 0:
                return False
            # No isolated nodes
            if any(d == 0 for _, d in G.degree()):
                return False
            # Connected
            try:
                return nx.is_connected(G)
            except nx.NetworkXPointlessConcept:
                # Happens for graphs with no edges; treat as invalid
                return False

        def collect_graphs_for_motif(motif_generator, target_count=500, batch_size=10, max_batches=20):
            collected = []
            batches = 0
            while len(collected) < target_count and batches < max_batches:
                for b in range(batch_size):
                    # 随机节点数与 BA 参数 m（每个新节点连接的边数）
                    num_nodes = random.randint(20, 25)
                    # m 至少为 1，且不超过 num_nodes-1；上限再限制到 5 以控制稠密度
                    # m_max = max(1, min(2, num_nodes - 1))
                    # m = random.randint(1, m_max)

                    ds = ExplainerDataset(
                        graph_generator=BAGraph(num_nodes=num_nodes, num_edges=1),
                        motif_generator=motif_generator,
                        num_motifs=1,
                        num_graphs=10,
                    )
                    for item in ds:
                        if is_connected_no_isolates(item):
                            collected.append(item)
                            if len(collected) >= target_count:
                                break
                batches += 1
            if len(collected) < target_count:
                raise RuntimeError(
                    f"Failed to collect {target_count} valid graphs (connected, no isolates) after {batches} attempts for motif {type(motif_generator).__name__}."
                )
            return collected[:target_count]

        # Collect at least 500 valid graphs for each dataset; regenerate batches until satisfied
        dataset1_items = collect_graphs_for_motif(motif1, target_count=500, batch_size=1000)
        dataset2_items = collect_graphs_for_motif(motif2, target_count=500, batch_size=1000)
        
        # Helper to build x using gaussian_lv_generator similar to syn_graph
        def build_x_via_gaussian(data_item, rng_seed=1234, n_features=10, n_informative=4, class_sep=1.0, n_clusters_per_class=2):
            # Determine number of nodes
            num_nodes = int(data_item.edge_index.max().item()) + 1

            # Build undirected NX graph from edge_index
            G = nx.Graph()
            G.add_nodes_from(range(num_nodes))
            ei = data_item.edge_index
            # add undirected edges (dedup)
            for i in range(ei.size(1)):
                u = int(ei[0, i].item()); v = int(ei[1, i].item())
                if u == v:
                    continue
                G.add_edge(min(u, v), max(u, v))

            # Node-level labels for feature generation: prefer node_mask if present
            if hasattr(data_item, 'node_mask') and data_item.node_mask is not None:
                node_mask = data_item.node_mask
                # ensure length
                if node_mask.numel() == num_nodes:
                    yvals = node_mask.long()
                else:
                    yvals = torch.zeros(num_nodes, dtype=torch.long)
            else:
                yvals = torch.zeros(num_nodes, dtype=torch.long)

            gen_features, _feature_imp_true = gaussian_lv_generator(
                G, yvals, seed=rng_seed,
                n_features=n_features,
                class_sep=class_sep,
                n_informative=n_informative,
                n_clusters_per_class=n_clusters_per_class,
            )
            x = torch.stack([gen_features(i) for i in range(num_nodes)]).float()
            return x

        for data in dataset1_items:
            x = build_x_via_gaussian(data_item=data, rng_seed=seed if seed is not None else 1234)
            graph = Data(
                x=x,
                edge_index=data.edge_index,
                y=torch.tensor([0], dtype=torch.long)
            )
            edge_mask = getattr(data, 'edge_mask', None)
            edge_imp = edge_mask.float() if edge_mask is not None else torch.zeros(graph.edge_index.size(1), dtype=torch.float)
            exp = Explanation(
                feature_imp=None,
                node_imp=None,
                edge_imp=edge_imp,
            )
            exp.set_whole_graph(graph)
            self.graphs.append(graph)
            self.explanations.append([exp])

        for data in dataset2_items:
            x = build_x_via_gaussian(data_item=data, rng_seed=seed if seed is not None else 1234)
            graph = Data(
                x=x,
                edge_index=data.edge_index,
                y=torch.tensor([1], dtype=torch.long)
            )
            edge_mask = getattr(data, 'edge_mask', None)
            edge_imp = edge_mask.float() if edge_mask is not None else torch.zeros(graph.edge_index.size(1), dtype=torch.float)
            exp = Explanation(
                feature_imp=None,
                node_imp=None,
                edge_imp=edge_imp,
            )
            exp.set_whole_graph(graph)
            self.graphs.append(graph)
            self.explanations.append([exp])

        # Save to cache for future fast loads
        try:
            torch.save({
                'graphs': self.graphs,
                'explanations': self.explanations,
                'meta': {
                    'shape1': shape1,
                    'shape2': shape2,
                    'num_nodes': '20-25',
                    'num_edges': 'random_1to5',
                    'num_motifs': 1,
                    'per_class': 500,
                }
            }, cache_file)
        except Exception:
            # If saving fails, continue without blocking usage
            pass

        super().__init__(name=f'{shape1}_{shape2}', seed=seed, split_sizes=split_sizes, device=device, shuffle=shuffle)

# triangle_grid = BA2Motif(shape1='triangle', shape2='grid',seed=42, device='cuda')

# print(f'house_triangle num_features: {triangle_grid.num_features}')
# print(f'house_triangle num_classes: {triangle_grid.num_classes}')

# idx = triangle_grid.train_index
# for i in idx:
#     data, exp = triangle_grid[i]
#     print(f'data.y: {data.y}')
#     print(f'data.x: {data.x}')
#     print(f'exp[0].edge_imp: {exp[0].edge_imp}')
#     print(f'exp[0].node_imp: {exp[0].node_imp}')
#     exp[0].visualize_graph(show=True)