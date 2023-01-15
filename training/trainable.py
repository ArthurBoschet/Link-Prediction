from abc import ABC, abstractmethod

class TrainableModel(ABC):

    def __init__(self):
        '''Instanciates a model that has a training step function and a testing function'''
        super(TrainableModel, self).__init__()


    @abstractmethod
    def train_step(self, batch, optimizer, device='cpu'):
      '''training step function for model (for each batch)'''
      pass

    @abstractmethod
    def predict(self, batch, device='cpu'):
      '''Prediction function has to be implemented'''
      pass