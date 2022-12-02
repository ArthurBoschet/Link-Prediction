import torch
from torch.utils.data import DataLoader

from linkpred.gat_link_pred import GATLinkPredModel
from training.train import train
from training.train_gnn import train_step_GNN, model_predict_GNN


from deepsnap.batch import Batch

from ray.tune.integration.wandb import (
    wandb_mixin,
)

@wandb_mixin
def train_gat(config ,datasets=None):

  assert datasets is not None, 'Data set should be passed as an argument'

  config = {
    "device": 'cuda' if torch.cuda.is_available() else 'cpu',
    "epochs": 3,
    "hidden_dim": 128, 
    "num_layers_gat": 2, 
    "num_layers_mlp": 1, 
    "heads": 3, 
    "dropout_gat" : 0, 
    "gat_type" : 'v2', 
    "dropout": 0,
    "lr": 0.01,
    "momentum": 0.9,
    "weight_decay": 5e-4,
  }

  #print which device is being used to the terminal
  device = config["device"]
  print(f"\nThe device is {device}\n")

  #get number of input dims and classes
  input_dim = 2 #datasets['train'].num_node_features
  num_classes = 2 #datasets['train'].num_edge_labels

  #GAT link prediction model
  model = GATLinkPredModel( 
      input_dim, 
      config["hidden_dim"], 
      config["num_layers_gat"], 
      config["num_layers_mlp"], 
      heads=config["heads"], 
      dropout_gat=config["dropout_gat"], 
      gat_type=config["gat_type"], 
      dropout=config["dropout"]
  ).to(config["device"])

  #print the model architecture
  print(model)

  #optimizer setup
  optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'], weight_decay=config['weight_decay'])

  #setup dataloader
  dataloaders = {split: DataLoader(
              ds, collate_fn=Batch.collate([]),
              batch_size=1, shuffle=(split=='train'))
              for split, ds in datasets.items()}

  #training
  return train(model, train_step_GNN, model_predict_GNN, dataloaders, optimizer, config, save_best=False, verbose=True)