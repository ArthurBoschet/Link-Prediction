from linkpred.base_link_pred import BaseLinkPredModel
from encoders.node2vec_encoder import Node2VecEncode
from predictors.mlp_predictor import MultiLayerPerceptron
from predictors.dot_product_predictor import DotProduct
from training.trainable import TrainableModel
import torch


class Node2VecLinkPredModel(BaseLinkPredModel, TrainableModel, torch.nn.Module):
    '''
    Node2vec prediction model that uses node2vec as the node encoder while using a neural network as the predictor.
    '''
    def __init__(
        self, 
        graph,
        input_size,
        walk_length,
        p,
        q, 
        num_layers, 
        hidden_size,
        device,
        dropout=0,
        batch_size=128,
        directory='drive/MyDrive/protein_network/string/human/n2v_embeddings',
        species='human',
        ):
      '''
      Sets up a binary cross entropy loss function combined with final sigmoid layer for added numerical stability. 
      The node2vec model is also fitted during the instanciation
      Parameters:
        graph (deepsnap.graph.Graph): DeepSnap training graph used for embedding the nodes
        input_size (int): size of the input vector
        walk_length (int): length of the random walks
        p (float): return parameter
        q (float): inout parameter
        hidden size (int): size of the hidden layers
        device (torch.cuda.device or str): device used by the model (gpu or cpu)
        num_layers (int): number of MLP layers (must be >= 0 and if 0 then dot product of the two nodes embeddings)
        dropout (float): probability of dropout at each MLP layer during training
        directory (str): the directory where the node2vec embeddings are stored
        species (str): the species evaluated
      '''

      #attributes for embedding directory
      self.directory = directory
      self.species = species

      #predictor network
      self.num_layers = num_layers
      self.hidden_size = hidden_size
      self.dropout = dropout

      #encoder node2vec
      self.walk_length = walk_length
      self.p = p
      self.q = q
      self.graph = graph
      self.input_size = input_size

      #device and training
      self.device = device
      self.batch_size = batch_size

      super(Node2VecLinkPredModel, self).__init__()

      #combines sigmoid and BCE into one function but is more numerically stable
      self.loss_fn = torch.nn.BCEWithLogitsLoss()


    def create_node_encoder(self):
      return Node2VecEncode(self.graph, self.input_size, self.walk_length, self.p, self.q, self.directory, self.species)


    def create_edge_predictor(self):
      if self.num_layers > 0:
        return MultiLayerPerceptron(self.input_size, self.hidden_size, self.num_layers, dropout=self.dropout)
      else:
        return DotProduct()


    def loss(self, pred, label):
      return self.loss_fn(pred, label)

    
    def forward(self, batch):
      '''
      get node2vec embeddings for encoding the nodes followed by edge predictions
      Parameters:
         batch (deepsnap.batch.Batch): batch with the message passing edges, supervision edges and node features

      Returns:
         pred (torch.tensor)
      '''
      edge_label_index = batch.to('cpu').numpy().T

      #create node embeddings with node2vec
      nodes_first, nodes_second = self.node_encoder(edge_label_index)

      #convert to pytorch
      nodes_first = torch.from_numpy(nodes_first).to(self.device)
      nodes_second = torch.from_numpy(nodes_second).to(self.device)

      #predict edges from two nodes
      pred = self.edge_predictor(nodes_first,  nodes_second)

      return pred


    def train_step(self, data, optimizer):
      '''training step function for model (for each batch)'''

      #set the batch to the device (cpu or gpu)
      data.to(self.device)

      batches_edges = torch.split(data.edge_label_index, self.batch_size, dim=1)
      batches_targets = torch.split(data.edge_label, self.batch_size, dim=0)

      for i, (batch, edge_label) in enumerate(zip(batches_edges, batches_targets)):

        #set the model to training mode
        self.train()

        #no gradient accumulation
        optimizer.zero_grad()

        #get predictions from training batch
        pred = self.forward(batch)

        #get loss
        loss = self.loss(pred, edge_label.type(pred.dtype))

        #gradient descent step
        loss.backward()
        optimizer.step()

      return loss


    def predict(self, batch):
      '''Prediction function has to be implemented'''
      #set the model to eval mode
      self.eval()

      #send batch to proper device and evaluate predictions
      edge_label_index = batch.edge_label_index
      pred = self.forward(edge_label_index)
      pred = torch.sigmoid(pred)

      return pred