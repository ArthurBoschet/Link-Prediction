import numpy as np
import torch
from gat_link_pred import GATLinkPredModel
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search import ConcurrencyLimiter
from ray import tune
from ray.air import session
#from ray.tune import Trainable
from ray.air.callbacks.wandb import WandbLoggerCallback
from ray.tune.integration.wandb import (
#    WandbTrainableMixin,
    wandb_mixin,
)
import wandb



def call():
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
  return

@wandb_mixin
def decorated_objective(config, checkpoint_dir=None):
    call()
    for i in range(30):
        print("hello")
        loss = config["mean"] + config["sd"] * np.random.randn()
        session.report({"loss": loss})
        wandb.log(dict(loss=loss))


def test_tune():
    # Step 1: Specify the search space
    hyperopt_space= {
        #  "netG_lr": hp.uniform( "netG_lr", 1e-5, 1e-2),
        # "netD_lr": hp.uniform( "netD_lr", 1e-5, 1e-2),
        # "beta1":hp.choice("beta1",[0.3,0.5,0.8]),
        "mean": lambda: np.random.uniform(1e-2, 1e-5),
        "sd": lambda: np.random.uniform(1e-2, 1e-5),
        # "mean": tune.grid_search([0.0001, 0.001, 0.1]),
        # "sd": tune.grid_search([0.9, 0.99]) , 
        }


    #step 2: initialize the search_alg object and (optionally) set the number of concurrent runs
    hyperopt_alg = HyperOptSearch(space = hyperopt_space,metric="is_score",mode="max")
    hyperopt_alg = ConcurrencyLimiter(hyperopt_alg, max_concurrent=1)


    #Step 3: Start the tuner
    analysis = tune.run(
        decorated_objective,
        callbacks=[WandbLoggerCallback(project="raytune-colab")],
        search_alg = hyperopt_alg, # Specify the search algorithm
        resources_per_trial={'gpu': 1,'cpu':2},
        num_samples=10,
        # config=config
        )