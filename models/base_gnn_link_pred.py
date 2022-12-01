from base_link_pred import BaseLinkPredModel
import torch

class BaseGNNLinkPredModel(BaseLinkPredModel, torch.nn.Module):
    '''
    Base GNN link prediction model template.
    '''
    def __init__(self):
      '''
      Sets up a binary cross entropy loss function combined with final sigmoid layer for added numerical stability
      '''
      super(BaseGNNLinkPredModel, self).__init__()

      #combines sigmoid and BCE into one function but is more numerically stable
      self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def loss(self, pred, label):
      return self.loss_fn(pred, label)

    
    def forward(self, batch):
      '''
      Message-passing through GNN for encoding the nodes followed by edge predictions
      Parameters:
         batch (deepsnap.batch.Batch): batch with the message passing edges, supervision edges and node features

      Returns:
         pred (torch.tensor)
      '''
      x, edge_index, edge_label_index = batch.node_feature, batch.edge_index, batch.edge_label_index

      #PyTorch Geometric GNN message passing step (aggregate / update) from message passing edges (edge_index)
      x = self.node_encoder(x, edge_index)

      #get node vectors for supervision (edge_label_index)
      nodes_first = torch.index_select(x, 0, edge_label_index[0,:].long())
      nodes_second = torch.index_select(x, 0, edge_label_index[1,:].long())

      #predict edges from two nodes
      pred = self.edge_predictor(nodes_first,  nodes_second)

      return pred