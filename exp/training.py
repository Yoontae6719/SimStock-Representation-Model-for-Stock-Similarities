import torch.nn.functional as F
import torch
from tqdm import tqdm
import numpy as np
import os
#
from torch.autograd import Variable

def make_noise(shape, type="Gaussian"):
    """
    Generate random noise.
    Parameters
    ----------
    shape: List or tuple indicating the shape of the noise
    type: str, "Gaussian" or "Uniform", default: "Gaussian".
    Returns
    -------
    noise tensor
    """

    if type == "Gaussian":
        noise = Variable(torch.randn(shape))
    elif type == 'Uniform':
        noise = Variable(torch.randn(shape).uniform_(-1, 1))
    else:
        raise Exception("ERROR: Noise type {} not supported".format(type))
    return noise

########Supervised###########

def train(dataloader, optimizer, model, args, log, device, input_E, input_hidden, task_id=0):
    E = input_E
    hidden = input_hidden
    loss_list = []
    
    log("Start Training on Domain {}...".format(task_id))
    for epoch in range(args.epoches):
        with tqdm(dataloader, unit="batch") as tepoch:
            for X, S in tepoch:
                X, S  = X.float().to(device), S.long().to(device)            
                initial_noise = make_noise((1, args.noise_dim), args.noise_type).to(device)
                model.train()
                optimizer.zero_grad()
                E, hidden, loss = model(X, initial_noise, S, E, hidden, False)
                E = E.detach()
                hidden = tuple([i.detach() for i in hidden])
                
                loss.backward()
                optimizer.step()    
                loss_list.append(loss.item())
                
        log("Task_ID: {}\tEpoch: {}\tAverage Training Loss: {}".format(task_id, epoch, np.mean(loss_list)))
    
    if not os.path.isdir('weight'):
        os.makedirs('weight')
        
    torch.save(model.state_dict(), './weight/model_weights_{}.pth'.format(args.save_name))
    
    return E, hidden, model

def test(dataloader, model, args, log, device, input_E, input_hidden, is_repre):
    model.eval()    
    E = input_E
    hidden = input_hidden
    loss_l = []
    representation_l = []
    attn_l = []
    
    with tqdm(dataloader, unit="batch") as tepoch:
        for X, S in tepoch:
            X, S  = X.float().to(device), S.long().to(device)
            initial_noise = make_noise((1, args.noise_dim), args.noise_type).to(device)
            
            with torch.no_grad():
                if not is_repre:
                    _, _, loss = models(X, initial_noise, S, E, hidden, False)
                    loss_l.append(loss.item())
                else:
                    representation, attn_ = models(X, initial_noise, S, E, hidden, True)
                    representation_l.append(representation)
                    attn_l.append(attn_)
                    
    if is_repre:
        return representation_l
    else:
        log("Average Testing Error is {}".format(np.mean(loss_l)))

    

def test_only_inference(dataloader, model, args, log, device, input_E, input_hidden, is_repre):
    model.eval()    
    E = input_E
    hidden = input_hidden
    loss_l = []
    representation_l = []
    attn_l = []
    
    model.load_state_dict(torch.load(os.path.join('./weight/', 'model_weights_{}.pth'.format(args.save_name))))
    # TODO : Must save the E and hiddens
    
    with tqdm(dataloader, unit="batch") as tepoch:
        for X, S in tepoch:
            X, S  = X.float().to(device), S.long().to(device)
            initial_noise = make_noise((1, args.noise_dim), args.noise_type).to(device)
            
            with torch.no_grad():
                if not is_repre:
                    _, _, loss = models(X, initial_noise, S, E, hidden, False)
                    loss_l.append(loss.item())
                else:
                    representation, attn_ = models(X, initial_noise, S, E, hidden, True)
                    representation_l.append(representation)
                    attn_l.append(attn_)
                    
    if is_repre:
        return representation_l
    else:
        log("Average Testing Error is {}".format(np.mean(loss_l)))
