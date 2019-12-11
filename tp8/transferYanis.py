import argparse
import os
import time

import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.utils.data
import torchvision.models as models
from PIL import Image
from torch.nn import functional as F
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.nn import functional as F
import torchvision
import numpy as np
import pickle,PIL
from sklearn.svm import LinearSVC

torchvision.models.vgg.model_urls["vgg16"] = "http://webia.lip6.fr/~robert/cours/rdfia/vgg16-397923af.pth"
os.environ["TORCH_HOME"] = "/tmp/torch"
PRINT_INTERVAL = 50
CUDA = False

    

class VGG16relu7(torch.nn.Module):
    def __init__(self,vgg16):
        super(VGG16relu7,self).__init__()
        self.features = torch.nn.Sequential(*list(vgg16.features.children()))
        self.classifier = torch.nn.Sequential(*list(vgg16.classifier.children())[:-2])


    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        return x 


def get_dataset(batch_size, path):

    def duplicateChannel(img):
        # Cette fonction permet de recopier 3 fois une image qui
        # ne serait que sur 1 channel (donc image niveau de gris)
        # pour la "transformer" en image RGB. Utilisez la avec
        # transform.Lambda
        img = img.convert('L')
        np_img = np.array(img, dtype=np.uint8)
        np_img = np.dstack([np_img, np_img, np_img])
        img = Image.fromarray(np_img, 'RGB')
        return img

    mu,sigma = [0.485,0.456,0.406],[0.229,0.224,0.225]
    train_dataset = datasets.ImageFolder(path+'/train',
        transform=transforms.Compose([ # TODO Pré-traitement à faire
            transforms.Resize(256),                    #[2]
            transforms.CenterCrop(224), 
            transforms.Lambda(duplicateChannel),
            transforms.ToTensor(),
            transforms.Normalize(mu,sigma, inplace=True)
        ]))
    val_dataset = datasets.ImageFolder(path+'/test',
        transform=transforms.Compose([ # TODO Pré-traitement à faire
            transforms.Resize(256),                    #[2]
            transforms.CenterCrop(224),
            transforms.Lambda(duplicateChannel),
            transforms.ToTensor(),
            transforms.Normalize(mu,sigma, inplace=True)
        ]))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                        batch_size=batch_size, shuffle=False, pin_memory=CUDA, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                        batch_size=batch_size, shuffle=False, pin_memory=CUDA, num_workers=2)

    return train_loader, val_loader


def extract_features(data, model):
    # TODO init features matrices
    X = np.zeros((len(data),data.batch_size,4096)) # 4096 dimension de l'avant derniere couche de VGG16.
    y = np.zeros((len(data),data.batch_size))
    model.eval()
    with torch.no_grad():
        for i, (x, target) in enumerate(data):
            if i % PRINT_INTERVAL == 0:
                print('Batch {0:03d}/{1:03d}'.format(i, len(data)))
            if CUDA:
                x = x.cuda()
            
            features = model(x)
            X[i] = features.detach().numpy()
            y[i]=target.detach().numpy()
            print("i=",i)
            
    return X, y


def reshape_no_batch(X,y):
    # Enleve les batch et reshape les données numpy. 
    X = X.reshape((X.shape[0]*X.shape[1],X.shape[3]))
    y = y.reshape((y.shape[0]*y.shape[1]))

    return X,y


if __name__ == '__main__':

    # Paramètres en ligne de commande
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='15SceneData', type=str, metavar='DIR', help='path to dataset')
    parser.add_argument('--batch-size', default=20, type=int, metavar='N', help='mini-batch size (default: 8)')
    parser.add_argument('--cuda', dest='cuda', action='store_true', help='activate GPU acceleration')
    
    args = parser.parse_args()
    if args.cuda:
        CUDA = True
        cudnn.benchmark = True

    #main(args)
    params = args
    print('Instanciation de VGG16')
    vgg16 = models.vgg16(pretrained=True)

    print('Instanciation de VGG16relu7')
    model = VGG16relu7(vgg16) # TODO À remplacer par un reseau tronché pour faire de la feature extraction

    model.eval()
    if CUDA: # si on fait du GPU, passage en CUDA
        model = model.cuda()

    # On récupère les données
    print('Récupération des données')
    train, test = get_dataset(params.batch_size, params.path)

    # Extraction des features
    print('Feature extraction')
    if not os.path.exists('numpy_object/X_train.npy') : 
        X_train, y_train = extract_features(train, model)
        np.save('numpy_object/X_train.npy', X_train)
        np.save('numpy_object/y_train.npy', y_train)
    else:
        print("Chargement des fichiers X_train.npy et y_train.npy")
        X_train = np.load('numpy_object/X_train.npy')
        y_train = np.load('numpy_object/y_train.npy')

    X_train,y_train = reshape_no_batch(X_train,y_train)

    if not os.path.exists('numpy_object/X_test.npy') : 
        X_test, y_test = extract_features(test, model)
        np.save('numpy_object/X_test.npy', X_test)
        np.save('numpy_object/y_test.npy', y_test)
    else:
        X_test = np.load('numpy_object/X_test.npy')
        y_test = np.load('numpy_object/y_test.npy')

    X_test,y_test = reshape_no_batch(X_test,y_test)
    

    # TODO Apprentissage et évaluation des SVM à faire
    print('Apprentissage des SVM')
    svm = LinearSVC(C=1.0)
    svm.fit(X_train,y_train)
    accuracy = svm.score(X_test,y_test)
    print("SVM accuraccy :",accuracy)

    input("done")