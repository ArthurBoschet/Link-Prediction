from linkpred.gat_link_pred import GATLinkPredModel
from training.train import train
from training.train_gnn import train_step_GNN, model_predict_GNN

import torch

from ray.air import session
from ray.tune.integration.wandb import (
    wandb_mixin,
)

@wandb_mixin
def train_gat(config, dataloaders_id):

  #print which device is being used to the terminal
  device = config["device"]
  print(f"\nThe device is {device}\n")

  #get number of input dims and classes
  input_dim = 2 

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

  #training
  score_val = train(model, train_step_GNN, model_predict_GNN, dataloaders_id, optimizer, config, save_best=False, verbose=True)
  session.report({"auroc_validation": score_val})
  return