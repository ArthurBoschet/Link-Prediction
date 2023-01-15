from abc import abstractmethod

from training.trainable import TrainableModel

class BaseLinkPredModel(TrainableModel):
    '''
    Base link prediction model template.
    '''
    def __init__(self):
        '''Every link prediction models needs a node encoder (node -> vector) and edge predictor (vector -> (0,1))'''
        super(BaseLinkPredModel, self).__init__()

        #create node encoder
        self.node_encoder = self.create_node_encoder()

        #edge prediction module
        self.edge_predictor = self.create_edge_predictor()

    @abstractmethod
    def create_node_encoder(self):
      '''Creates the node encoder'''
      pass

    @abstractmethod
    def create_edge_predictor(self):
      '''Creates the edge predictor'''
      pass