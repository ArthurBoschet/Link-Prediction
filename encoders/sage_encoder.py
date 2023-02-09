import torch
import torch.nn as nn
from torch_geometric.nn import Sequential
from torch_geometric.nn import SAGEConv

class SAGEnetwork(torch.nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, dropout=0, aggr='mean'):
    '''
    GAT (Graph Attention Network) that can be used for encoding the nodes
    Parameters:
      input_size (int): size of the input vector
      hidden size (int): size of the hidden layers
      num_layers (int): number of GAT layers
      dropout (float): probability of dropout at each layer during training
      aggr  (Optional[Union[str, List[str], Aggregation]]): for exemple 'mean', 'max', 'lstm'
    '''
    super(SAGEnetwork, self).__init__()

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
              SAGEConv(
                  (input_size if i==0 else hidden_size), 
                  hidden_size,
                  aggr=aggr
                ), 
                'x, edge_index -> x'
            )
          )

        #apply dropout/non linear and relu if not last layer
        if i < num_layers-1:
          gnn_stack.append(
                (
                  nn.BatchNorm1d(hidden_size), 'x -> x'
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