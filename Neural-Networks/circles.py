import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from tme5 import CirclesData
from random import gauss 

def init_params(nx, nh, ny):
    params = {}
    Wh = torch.tensor(np.array([gauss(0,0.3) for i in range (nh*nx)])).reshape(nh,nx)
    Wh = Wh.type(torch.FloatTensor)
    Wy = torch.tensor(np.array([gauss(0,0.3) for i in range (nh*ny)])).reshape(ny,nh)
    Wy = Wy.type(torch.FloatTensor)
    #bh = torch.tensor(np.array([gauss(0,0.3) for i in range (nh)]))
    #bh = bh.type(torch.FloatTensor)
    #by = torch.tensor(np.array([gauss(0,0.3) for i in range (ny)]))
    #by = by.type(torch.FloatTensor)
    bh = np.array([gauss(0,0.3) for i in range (nh)])
    by = np.array([gauss(0,0.3) for i in range (ny)])
    params["Wh"] = Wh
    params["Wy"] = Wy
    params["bh"] = bh
    params["by"] = by
    return params


def forward(params, X):
    outputs = {}
    
    batch = X.shape[0]
    bh = params["bh"] 
    by = params["by"] 
    By=[]
    Bh=[]
    for i in range (batch): 
        Bh.append(bh)
        By.append(by)
    Bh = torch.tensor(Bh).t()
    Bh = Bh.type(torch.FloatTensor)
    By = torch.tensor(By).t()
    By = By.type(torch.FloatTensor)
    
    # TODO remplir avec les paramètres X, htilde, h, ytilde, yhat
    
    htilde = torch.mm(X,params["Wh"].t()).t()  + Bh 
    h = torch.tanh(htilde).t()
    #print('h',h)
    #print(By)
    #print(torch.mm(h,params["Wy"].t()))
    ytilde = torch.mm(h,params["Wy"].t())+By.t()
    #print('ytilde',ytilde)
    activationF = torch.nn.Softmax(dim=1)
    yhat = activationF(ytilde)
    
    outputs['yhat'] = yhat 
    outputs['ytilde'] = ytilde 
    outputs['htilde'] = htilde 
    outputs['h'] = h 
    
    return outputs['yhat'], outputs

def loss_accuracy(Yhat, Y):
    L = 0
    acc = 0

    # TODO

    return L, acc

def backward(params, outputs, Y):
    grads = {}

    # TODO remplir avec les paramètres Wy, Wh, by, bh
    # grads["Wy"] = ...

    return grads

def sgd(params, grads, eta):
    # TODO mettre à jour le contenu de params

    return params



if __name__ == '__main__':

    # init
    data = CirclesData()
    data.plot_data()
    N = data.Xtrain.shape[0]
    Nbatch = 10
    nx = data.Xtrain.shape[1]
    nh = 10
    ny = data.Ytrain.shape[1]
    eta = 0.03

    # Premiers tests, code à modifier
    params = init_params(nx, nh, xy)
    Yhat, outs = forward(params, data.Xtrain)
    L, _ = loss_accuracy(Yhat, Y)
    grads = backward(params, outputs, Y)
    params = sgd(params, grads, eta)

    # TODO apprentissage

    # attendre un appui sur une touche pour garder les figures
    input("done")
