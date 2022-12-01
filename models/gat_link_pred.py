from base_gnn_link_pred import BaseGNNLinkPredModel
from encoders.gat_encoder import GATnetwork
from predictors.mlp_predictor import MultiLayerPerceptron
from predictors.dot_product_predictor import DotProduct

class GATLinkPredModel(BaseGNNLinkPredModel):

  def __init__(self, 
               input_size, 
               hidden_size, 
               num_layers_gat, 
               num_layers_mlp, 
               heads=1, 
               dropout_gat=0, 
               gat_type='v2', 
               dropout=0
               ):
    '''
    Model with GAT network as node encoder and an MLP or "dot product" as edge predictor
    Parameters:
      input_size (int): size of the input vector
      hidden size (int): size of the hidden layers
      num_layers_gat (int): number of GAT layers
      num_layers_mlp (int): number of MLP layers (must be >= 0 and if 0 then dot product of the two nodes embeddings)
      dropout_gat (float): probability of dropout at each gat layer during training (before entering the GAT layer)
      gat_type (str): type of gat layer used ('v1' or 'v2')
      dropout (float): probability of dropout at each MLP layer during training
    '''

    assert num_layers_mlp >= 0, "num_layers_mlp must be either 0 (dot product) or greater (MLP)"

    self.input_size = input_size
    self.hidden_size = hidden_size
    self.num_layers_gat = num_layers_gat
    self.num_layers_mlp = num_layers_mlp
    self.heads = heads
    self.dropout_gat = dropout_gat
    self.gat_type = gat_type
    self.dropout = dropout
    
    super(GATLinkPredModel, self).__init__()

  def create_node_encoder(self):
    return GATnetwork(self.input_size, self.hidden_size, self.num_layers_gat, heads=self.heads, dropout=self.dropout_gat, gat_type=self.gat_type)

  def create_edge_predictor(self):
    if self.num_layers_mlp > 0:
      return MultiLayerPerceptron(self.hidden_size*self.heads, self.hidden_size, self.num_layers_mlp, dropout=self.dropout)
    else:
      return DotProduct()