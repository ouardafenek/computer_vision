import argparse
import os
import time

import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.utils.data
import torchvision.models as models
from PIL import Image
from torch.nn import functional as F
from scipy.special import softmax
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.nn import functional as F
import torchvision
import numpy as np
import pickle,PIL

torchvision.models.vgg.model_urls["vgg16"] = "http://webia.lip6.fr/~robert/cours/rdfia/vgg16-397923af.pth"
os.environ["TORCH_HOME"] = "/tmp/torch"
PRINT_INTERVAL = 50
CUDA = False

imagenet_classes= pickle.load(open('imagenet_classes.pkl','rb')) # chargement des classes


img = PIL.Image.open("chat_mignon.jpg").convert('RGB')

#img.show()
#img = img.resize((224,224),PIL.Image.BILINEAR)
#img = np.array(img,dtype=np.float32) /255

mu,sigma = [0.485,0.456,0.406],[0.229,0.224,0.225]
#img = img.transpose((2,0,1))
transform = transforms.Compose([            #[1]
 transforms.Resize(256),                    #[2]
 transforms.CenterCrop(224),                #[3]
 transforms.ToTensor(),                     #[4]
 transforms.Normalize(mu,sigma)                  #[7]
 ])
#img = np.expand_dims(img,0)
img_t = transform(img)
#x = torch.Tensor(img)
x = torch.unsqueeze(img_t, 0)
vgg = models.vgg16(pretrained=True)
vgg.eval()
y = vgg(x)
y = y.detach().numpy()
y = softmax(y)
topN = 5
top = np.argsort(-y[0])[:topN]
proba = np.sort(-y[0])[:topN]
proba = -proba
liste = [imagenet_classes[i] for i in list(top)]


print("VGG16 reconnait: ",liste," sur l'image donnée, avec des probabilités respectives de :",list(proba))




# Affichage de cartes 


carte = False
if carte : 
    class Vgg16(torch.nn.Module):
        def __init__(self):
            super(Vgg16, self).__init__()
            features = list(models.vgg16(pretrained = True).features)
            # features的第3，8，15，22层分别是: relu1_2,relu2_2,relu3_3,relu4_3
            self.features = torch.nn.ModuleList(features).eval() 
            
        def forward(self, x):
            self.results = []
            for _,model in enumerate(self.features):
                x = model(x)
                
                #print(model)
                self.results.append(x)
            return self.results


    vgg_maps = Vgg16()
    maps = vgg_maps.forward(x)

    cc1 = maps[0]
    npcc1 = cc1.detach().numpy()
    npcc1 = npcc1[0]

    for i in range(64): 
        map_conv = Image.fromarray(npcc1[i],'RGB') 
        map_conv.save('maps_conv_1/maps'+str(i)+'.png')



    ccfin = maps[-3:][0]
    npccfin = ccfin.detach().numpy()
    npccfin = npccfin[0]

    for i in range(512): 
        map_conv = Image.fromarray(npccfin[i],'RGB') 
        map_conv.save('maps_conv_fin/maps'+str(i)+'.png')