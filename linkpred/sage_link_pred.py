from linkpred.base_gnn_link_pred import BaseGNNLinkPredModel
from encoders.sage_encoder import SAGEnetwork
from predictors.mlp_predictor import MultiLayerPerceptron
from predictors.dot_product_predictor import DotProduct

class SAGELinkPredModel(BaseGNNLinkPredModel):

  def __init__(self, 
               input_size, 
               hidden_size, 
               num_layers_sage, 
               num_layers_mlp, 
               device='cpu',
               aggr='mean', 
               dropout_sage=0, 
               dropout=0
               ):
    '''
    Model with GAT network as node encoder and an MLP or "dot product" as edge predictor
    Parameters:
      input_size (int): size of the input vector
      hidden size (int): size of the hidden layers
      num_layers_sage (int): number of GAT layers
      num_layers_mlp (int): number of MLP layers (must be >= 0 and if 0 then dot product of the two nodes embeddings)
      device (torch.cuda.device or str): device used by the model (gpu or cpu)
      aggr  (Optional[Union[str, List[str], Aggregation]]): for exemple 'mean', 'max', 'lstm'
      dropout_sage (float): probability of dropout at each gat layer during training (before entering the GAT layer)
      dropout (float): probability of dropout at each MLP layer during training
    '''

    assert num_layers_mlp >= 0, "num_layers_mlp must be either 0 (dot product) or greater (MLP)"

    self.input_size = input_size
    self.hidden_size = hidden_size
    self.num_layers_sage = num_layers_sage
    self.num_layers_mlp = num_layers_mlp
    self.aggr = aggr
    self.dropout_sage = dropout_sage
    self.dropout = dropout
    
    super(SAGELinkPredModel, self).__init__(device)

  def create_node_encoder(self):
    '''creates a node encoder with a GAT Network'''
    return SAGEnetwork(self.input_size, self.hidden_size, self.num_layers_sage, aggr=self.aggr, dropout=self.dropout_sage)

  def create_edge_predictor(self):
    '''creates an edge predictor with an MLP'''
    if self.num_layers_mlp > 0:
      return MultiLayerPerceptron(self.hidden_size, self.hidden_size, self.num_layers_mlp, dropout=self.dropout)
    else:
      return DotProduct()