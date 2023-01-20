from linkpred.base_link_pred import BaseLinkPredModel
from training.trainable import TrainableModel
import torch

class BaseGNNLinkPredModel(BaseLinkPredModel, TrainableModel, torch.nn.Module):
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


    def train_step(self, batch, optimizer, device='cpu'):
      '''training step function for model (for each batch)'''
      #set the batch to the device (cpu or gpu)
      batch.to(device)

      #set the model to training mode
      self.train()

      #no gradient accumulation
      optimizer.zero_grad()

      #get predictions from training batch
      pred = self.forward(batch)

      #get loss
      loss = self.loss(pred, batch.edge_label.type(pred.dtype))

      #gradient descent step
      loss.backward()
      optimizer.step()

      return loss


    def predict(self, batch, device='cpu'):
      '''Prediction function has to be implemented'''
      #set the model to eval mode
      self.eval()

      #send batch to proper device and evaluate predictions
      batch.to(device)
      pred = self.forward(batch)
      pred = torch.sigmoid(pred)

      return pred