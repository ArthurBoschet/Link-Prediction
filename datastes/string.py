import pandas as pd
import networkx as nx

import torch
import numpy as np

from deepsnap.graph import Graph
from deepsnap.dataset import GraphDataset

species = "human"
glob_dir = "drive/MyDrive/protein_network/string/"+species
data = glob_dir + '/data'
embeddings = glob_dir + '/embeddings'


def load_string_network(
    probability_threshold=0.7, 
    dir_edges=data +"/9606.protein.links.detailed.v11.5.txt", 
    dir_node_info=data +"/human_string_dataset Full Network default node.csv"
    ):
  '''
  Function used to load the string network as a networkx graph
  Parameters:
    probability_threshold (float): probability necessary to keep edge from STRING network (kept if p>probability_threshold)
    dir_node_info (str): directory of the string dataset node info data
  Returns:
    G_string (networkx.classes.graph.Graph): string network
  '''

  #get node info
  node_info = pd.read_csv(dir_node_info)

  #get description to name dictionary
  description_to_name = {k:v for k,v in zip(node_info.Description, node_info.name)}

  #get string dataset
  string_edges = pd.read_csv(dir_edges, sep=' ')

  #only keepwith 70% of above certainty
  string_edges_70 = string_edges[string_edges['combined_score'].values >= int(1000*probability_threshold)]

  #get interaction with names
  interactions = [(description_to_name[p1], description_to_name[p2]) for p1, p2 in zip(string_edges_70.protein1.values, string_edges_70.protein2.values)]

  #create full graph
  G_string = nx.Graph()
  G_string.add_edges_from(interactions)

  return G_string



def create_snap_graph(
    G ,
    functions=[lambda G: nx.betweenness_centrality(G, k=100), lambda G: dict(G.degree)]
    ):
  '''
  Function that converts network x graph to DeepSnap graph while also computing relevant features
  Parameters:
    G (networkx.classes.graph.Graph): network to be converted
    functions (list(function(networkx.classes.graph.Graph -> dict(str, float))): feature generators on the graph G
  Returns:
    dg (deepsnap.graph.Graph): DeepSnap graph returned
  '''

  #iterate over feature generators
  features_dicts = [f(G) for f in functions]

  #create a node feature dictionary
  node_feature = {}
  for k,v in features_dicts[0].items():
    node_feature[k] = torch.from_numpy(np.array([v] + [feat[k] for feat in features_dicts[1:]])).type(dtype=torch.float32)

  #add node features to the network
  nx.set_node_attributes(G, node_feature, "node_feature")

  #create DeepSnap graph
  dg = Graph(G)

  return dg


def create_deepsnap_datasets(
    dg, 
    edge_train_mode = "disjoint", 
    edge_negative_sampling_ratio = 1.0, 
    edge_message_ratio=0.9,
    split_ratio=[0.8, 0.1, 0.1],
    ):
  '''
  Create a link prediction dataset.
  Parameters:
    dg (deepsnap.graph.Graph): graph to be turned into a dataset
    edge_train_mode (str): "disjoint" or "all" how to split the training edges (supervision/message)
    edge_negative_sampling_ratio (float): neg_edges/pos_edges
    edge_message_ratio (float): ratio of edges which are edge passing (the rest are supervision edges)
    split_ratio (list(float)): train/val/test split 
  Returns:
    datasets (deepsnap.datasetGraphDataset): network transductive dataset split into train/val/test
  '''
  datasets = {}
  dataset = GraphDataset(
      [dg], 
      task="link_pred", 
      edge_train_mode=edge_train_mode, 
      edge_message_ratio=edge_message_ratio, 
      edge_negative_sampling_ratio=edge_negative_sampling_ratio
      )

  datasets['train'], datasets['val'], datasets['test'] = dataset.split(
      transductive=True, 
      split_ratio=split_ratio
      )  
  return datasets



def get_string_dataset(
    probability_threshold=0.7, 
    dir_edges=data +"/9606.protein.links.detailed.v11.5.txt", 
    dir_node_info=data +"/human_string_dataset Full Network default node.csv",
    functions=[lambda G: nx.betweenness_centrality(G, k=100), lambda G: dict(G.degree)],
    edge_train_mode = "disjoint", 
    edge_negative_sampling_ratio = 1.0, 
    edge_message_ratio=0.9,
    split_ratio=[0.8, 0.1, 0.1]
    ):
    '''
    Function get a string dataset
    Parameters:
        probability_threshold (float): probability necessary to keep edge from STRING network (kept if p>probability_threshold)
        dir_node_info (str): directory of the string dataset node info data
        functions (list(function(networkx.classes.graph.Graph -> dict(str, float))): feature generators on the graph G
        edge_train_mode (str): "disjoint" or "all" how to split the training edges (supervision/message)
        edge_negative_sampling_ratio (float): neg_edges/pos_edges
        edge_message_ratio (float): ratio of edges which are edge passing (the rest are supervision edges)
        split_ratio (list(float)): train/val/test split 
    Returns:
        datasets (deepsnap.datasetGraphDataset): network transductive dataset split into train/val/test
    '''
    G = load_string_network(
        probability_threshold=probability_threshold, 
        dir_edges=dir_edges, 
        dir_node_info=dir_node_info
    )
    dg = create_snap_graph(
        G,
        functions=functions
    )
    datasets =  create_deepsnap_datasets(
        dg,
        edge_train_mode = edge_train_mode, 
        edge_negative_sampling_ratio = edge_negative_sampling_ratio, 
        edge_message_ratio=edge_message_ratio,
        split_ratio=split_ratio
    )
    return datasets

