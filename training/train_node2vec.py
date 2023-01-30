from linkpred.node2vec_link_pred import Node2VecLinkPredModel
from training.train import train

import torch

from ray.air import session
from ray.tune.integration.wandb import (
    wandb_mixin,
)

@wandb_mixin
def train_node2vec(config):

  #print which device is being used to the terminal
  device = config["device"]
  print(f"\nThe device is {device}\n")

  #GAT link prediction model
  model = Node2VecLinkPredModel( 
    config["graph"],
    int(config["input_size"]),
    int(config["walk_length"]),
    config["p"],
    config["q"], 
    int(config["num_layers"]), 
    int(config["hidden_size"]),
    config["device"],
    dropout=config["dropout"],
    directory=config["directory"],
    species=config["species"],
  ).to(config["device"])

  #print the model architecture
  print(model)

  #optimizer setup
  optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'], weight_decay=config['weight_decay'])

  #training
  score_val = train(model, config['dataloaders_id'], optimizer, config, save_best=False, verbose=True)
  session.report({"auroc_validation": score_val})
  return