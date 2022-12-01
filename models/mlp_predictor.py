import torch
import torch.nn as nn

class MultiLayerPerceptron(torch.nn.Module):
  '''Mutli-layer perceptron that can be used to predict edges'''

  def __init__(self, input_size, hidden_size, num_layers, dropout=0):
      '''
      Parameters:
        input_size (int): size of the input vector
        hidden_size (int): size of the hidden layers
        num_layers (int): number of linear layers (number of hidden layers + 1)
        dropout (float): probability of dropout at each layer during training
      '''
      super(MultiLayerPerceptron, self).__init__()

      #ouput model from embeddings
      assert num_layers >= 1, "Number of layers must be at least 1"

      hidden_linear_relu_stack = []
      for i in range(num_layers):
          #linear layers
          hidden_linear_relu_stack.append(
              nn.Linear(
                  (input_size if i==0 else hidden_size), 
                  (hidden_size if i < num_layers-1 else 1)
                  )
              )

          #apply dropout/non linear and relu if not last layer
          if i < num_layers-1:
            hidden_linear_relu_stack.append(nn.Dropout(p=dropout))
            hidden_linear_relu_stack.append(nn.ReLU())


      #create a neural network at the ouput
      self.hidden_linear_relu_stack = nn.Sequential(*hidden_linear_relu_stack)


  def forward(self, nodes_first, nodes_second):
    '''
    takes vector as input and outputs a prediction
    Parameters:
      nodes_first (torch.tensor): first node input vector
      nodes_second (torch.tensor): second node input vector

    Returns:
      pred (torch.tensor): ouput predictions
    '''
    pred = self.hidden_linear_relu_stack(nodes_first * nodes_second).flatten()
    return pred