import torch
import torch.nn as nn
from torch_geometric.nn import Sequential
from torch_geometric.nn import GATConv, GATv2Conv

class GATnetwork(torch.nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, heads=1, dropout=0, gat_type='v2'):
    '''
    GAT (Graph Attention Network) that can be used for encoding the nodes
    Parameters:
      input_size (int): size of the input vector
      hidden size (int): size of the hidden layers
      num_layers (int): number of GAT layers
      dropout (float): probability of dropout at each layer during training
      gat_type (str): type of gat layer used ('v1' or 'v2')

    '''
    super(GATnetwork, self).__init__()

    #types of GAT layers compatible with this architechture
    types = {
        'v1': GATConv,
        'v2': GATv2Conv
    }
    ConvType = types[gat_type]

    assert num_layers >= 1, 'number of GNN layers must be at least 1'

    gnn_stack = []
    for i in range(num_layers):

        #apply dropout before gnn layer
        gnn_stack.append(
            (
              nn.Dropout(p=dropout), 'x -> x'
            )
        )

        #gnn layers
        gnn_stack.append(
            (
              ConvType(
                  (input_size if i==0 else hidden_size*heads), 
                  hidden_size,
                  heads=heads
                ), 
                'x, edge_index -> x'
            )
          )

        #apply dropout/non linear and relu if not last layer
        if i < num_layers-1:
          gnn_stack.append(
                (
                  nn.BatchNorm1d(hidden_size*heads), 'x -> x'
                )
              )
          gnn_stack.append(
                (
                    nn.LeakyReLU(), 'x -> x'
                )
              )

    #create graph neural network stack
    self.gnn_stack = Sequential('x, edge_index', gnn_stack)

  def forward(self, x, edge_index):
    '''
    takes vector as input and outputs a prediction
    Parameters:
      x (torch.tensor): input node features
      edge_index (torch.tensor): message passing edges

    Returns
      x (torch.tensor): ouput node embeddings
    '''
    x = self.gnn_stack(x, edge_index)
    return x