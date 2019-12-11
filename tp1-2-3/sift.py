import matplotlib
import matplotlib.pyplot as plt
#matplotlib inline  
import numpy as np
from tools import *
from scipy import ndimage
from math import *

def compute_grad(I):
    h_x = np.array([-0.5,0,0.5])
    h_y = np.array([0.5,1,0.5])
    Mx = conv_separable(I, h_y, h_x)
    My = conv_separable(I, h_x, h_y)
    Ix = ndimage.convolve(I, Mx)
    Iy = ndimage.convolve(I, My)
    return Ix, Iy


def compute_grad_mod_ori(I):
    Ix, Iy = compute_grad(I)
    Gn = np.sqrt( Ix * Ix + Iy * Iy) 
    Go = compute_grad_ori(Ix, Iy, Gn)
    return Gn, Go

#fonctions auxiliaires. 
def extract_patch(I,xi,yj):
    patch=[]
    if xi+16 > len(I): 
        xi = xi - ((xi+16) % len(I))
    if yj+16 > len(I[0]): 
        yj = yj - ((yj+16) % len(I[0]))
    for i in range (xi,xi+16):
        patch.append(I[i][yj:yj+16])
    return np.asarray(patch)

def pond_patch(patch,sigma):
    m =np.mean(patch) 
    for i in range(patch.shape[0]): 
        for j in range(patch.shape[1]):  
            patch[i][j]= (1/(sigma*sqrt(2*pi)))*exp(-(((patch[i][j]-m)/(2*sigma))**2))
    return patch 

def extract_regions(patch):
    regs=[]
    for i in range(0,16,4):
        for k in range(i,i+4):
            reg=[]
            for j in range(0,16,4):
                reg.append(patch[k][j:j+4])
            regs.append(reg)
    return np.asarray(regs)

def construct_hist(Gn,Go):
    regions_pond = extract_regions(Gn)
    regions_orient = extract_regions(Go)
    Renc = np.zeros((16,8))
    for k in range(16):
        for j in range(8):
            Renc[k][j]=regions_pond[k][np.where(regions_orient[k] == j)].sum()
    return Renc.reshape(16*8)

def post_processing(Penc):
    
    if np.linalg.norm(Penc) < 0.5 : 
        return np.zeros(len(Penc))
    else: 
        #normaliser le vecteur 
        Penc /= np.linalg.norm(Penc)
    Penc = np.where(Penc <= 0.2 , Penc, 0.2) 
    Penc /= np.linalg.norm(Penc)
    return Penc 
        

def compute_sift_region(Gn, Go, mask=None):
    # TODO
    Penc = construct_hist(Gn,Go)
    sift = post_processing(Penc)
    # Note: to apply the mask only when given, do:
    if mask is not None:
        width = 16 
        Gn_pond = pond_patch(Gn, mask*width)
        Penc = construct_hist(Gn_pond,Go)
        sift = post_processing(Penc)
    return sift


def compute_sift_image(I):
    x, y = dense_sampling(I)
    im = auto_padding(I)
    # TODO calculs communs aux patchs
    sifts = np.zeros((len(x), len(y), 128))
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            patch = extract_patch(I,xi,yj)
            Gn,Go = compute_grad_mod_ori(patch)
            sifts[i, j, :] = compute_sift_region(Gn, Go, mask=None) # SIFT du patch de coordonnee (xi, yj)
    return sifts