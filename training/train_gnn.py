import torch

def train_step_GNN(model, batch, optimizer, args):
    #set the batch to the device (cpu or gpu)
    batch.to(args["device"])

    #set the model to training mode
    model.train()

    #no gradient accumulation
    optimizer.zero_grad()

    #get predictions from training batch
    pred = model(batch)

    #get loss
    loss = model.loss(pred, batch.edge_label.type(pred.dtype))

    #gradient descent step
    loss.backward()
    optimizer.step()

    return loss

def model_predict_GNN(model, batch, args):

    #set the model to eval mode
    model.eval()

    #send batch to proper device and evaluate predictions
    batch.to(args["device"])
    pred = model(batch)
    pred = torch.sigmoid(pred)

    return pred