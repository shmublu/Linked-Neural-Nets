import os
import glob
import d2l
from skimage.io import imread
import torch, torchvision
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18, ResNet18_Weights
from torch.optim import SGD
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, label, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.label = label

    def __len__(self):
        return len(glob.glob(self.img_dir + '/*'));

    def __getitem__(self, idx):
        img_path = glob.glob(self.img_dir + '/*')[idx]
        image = (read_image(img_path))
        if self.transform:
            image = self.transform(image)
        return image, self.label

    


#takes in a list of [(img_dir, label, transform), ...]
def concat_datasets(list_of_3_tup):
    dsets = []
    for tup in list_of_3_tup:
        dsets.append(CustomImageDataset(tup[0], tup[1], tup[2]))
    return ConcatDataset(dsets)
    

def set_loaders(datasets, batch_size):
    train_loaders = []
    validation_loaders = []
    test_loaders = []
    for d in datasets:
       train_loaders.append(DataLoader(d[0],batch_size=batch_size, shuffle=True))
       validation_loaders.append(DataLoader(d[1],batch_size=batch_size, shuffle=False))
       test_loaders.append(DataLoader(d[2],batch_size=batch_size, shuffle=False))
    return (train_loaders, validation_loaders, test_loaders)


    
def equalize_subset(nnet, nets, layers, layer_nodes = None, divisor = 2):
    copy = nnet.state_dict()
    for net in nets:
        if net is nnet:
            continue
        for ind, layer in enumerate(layers):
            if layer_nodes:
                for node in layer_nodes[ind]:
                    net.state_dict()[layer][node].data.copy_(copy[layer][node])
                continue
            elif net.state_dict()[layer].dim():
                for node in range(len(net.state_dict()[layer]) // divisor):
                    net.state_dict()[layer][node].data.copy_(copy[layer][node])
                
                    
def calculate_accuracy(output, label):
    """Calculate the accuracy of the trained network. 
    output: (batch_size, num_output) float32 tensor
    label: (batch_size, ) int32 tensor """
    
    return (output.argmax(axis=1) == label.float()).float().mean()
                
def alt_train(nets, epochs, learning_rate, train_loaders, validation_loaders, test_loaders):
    train_loss, val_loss, train_acc, valid_acc= [],[],[],[]
    for nnet in nets:
       """Set up the NNs on the device and initalize their accuracies and loss functions"""
       nnet = nnet.to(device)
       train_loss.append(0.)
       val_loss.append(0.)
       train_acc.append(0.)
       valid_acc.append(0.)
    for epoch in range(epochs):
       traind = list(map(iter, train_loaders))
       more_evidence = True
       while(more_evidence):
           for ind,nnet in enumerate(nets):
              nnet.train()
              try:
                 data, label = next(traind[ind])
              except Exception as e:
                  more_evidence = False
                  print(e)
                  continue
              criterion = nn.CrossEntropyLoss()
              optimizer = SGD(nnet.parameters(), lr=learning_rate)
              optimizer.zero_grad()
              # Put data and label to the correct device
              data = data.to(device)
              label = label.to(device)
              # Make forward pass
              output = nnet(data)
              # Calculate loss
              loss = criterion(output, label)
              # Make backwards pass (calculate gradients)
              loss.backward()
              # Accumulate training accuracy and loss
              train_acc[ind] += calculate_accuracy(output, label).item()
              train_loss[ind] += loss.item()
              # Update weights
              optimizer.step()
              equalize_subset(nnet,nets, list(nnet.state_dict().keys())[60:78], divisor = 2)
        # Validation loop:
       for ind, net in enumerate(nets):
           net.eval() # Activate "evaluate" mode (don't use dropouts etc.)
           with torch.no_grad():
               for data, label in validation_loaders[ind]:
                   data = data.to(device)
                   label = label.to(device)
                   output = net(data)
                   valid_acc[ind] += calculate_accuracy(output, label).item()
                   val_loss[ind] += criterion(output, label).item()
                   #fix this later so different ones can have different amouns of evidence
           train_loss[ind] /= len(train_loaders[ind])
           train_acc[ind] /= len(train_loaders[ind])
           val_loss[ind] /= len(validation_loaders[ind])
           valid_acc[ind] /= len(validation_loaders[ind])
           print("Epoch %d. NN %d.: train loss %.3f, train acc %.3f, val loss %.3f, val acc %.3f" % (
epoch+1, ind, train_loss[ind], train_acc[ind], val_loss[ind], valid_acc[ind]))
           train_loss[ind] = 0
           train_acc[ind] = 0
           val_loss[ind] = 0
           valid_acc[ind] = 0
    return nets[0]
        
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

batch_size = 64
transform =  transforms.Compose([transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.Grayscale(3), transforms.ToTensor(), transforms.Normalize(mean=(0,0,0), std=(1,1,1))])
zebra = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
#zebra = resnet18().to(device)
epochs = 7
learning_rate = 0.01
horse = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
#horse = resnet18().to(device)
rnest= resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
testnet1 = resnet18().to(device)
testnet2 = resnet18().to(device)
print(len(horse.state_dict()['layer4.1.conv1.weight']))
print(list(horse.state_dict().keys())[79:90])
#fi.state_dict()['layer4.1.bn1.weight'][0].data.copy_(ci.state_dict()['layer4.1.bn1.weight'][0])
#print(fi.state_dict()['layer4.1.bn1.weight'].data)
tr1 = concat_datasets([(r"C:\Users\samue\Desktop\AI\datasets\train1\elephants", 0, transform),
                             (r'C:\Users\samue\Desktop\AI\datasets\train1\zebras', 1, transform)])
tr2 = concat_datasets([(r"C:\Users\samue\Desktop\AI\datasets\train2\horses", 1, transform),
                             (r'C:\Users\samue\Desktop\AI\datasets\train2\butterflies', 0, transform)])
v1 = concat_datasets([(r"C:\Users\samue\Desktop\AI\datasets\validate1\elephants", 0, transform),
                             (r'C:\Users\samue\Desktop\AI\datasets\validate1\zebras', 1, transform)])
v2 = concat_datasets([(r"C:\Users\samue\Desktop\AI\datasets\validate2\horses", 1, transform),
                             (r'C:\Users\samue\Desktop\AI\datasets\validate2\butterflies', 0, transform)])
ts1 = concat_datasets([(r"C:\Users\samue\Desktop\AI\datasets\test1\elephants", 0, transform),
                             (r'C:\Users\samue\Desktop\AI\datasets\test1\zebras', 1, transform)])
ts2 = concat_datasets([(r"C:\Users\samue\Desktop\AI\datasets\test2\horses", 1, transform),
                             (r'C:\Users\samue\Desktop\AI\datasets\test2\butterflies', 0, transform)])
tr3 = concat_datasets([(r"C:\Users\samue\Desktop\AI\datasets\train1\elephants", 0, transform),
                             (r'C:\Users\samue\Desktop\AI\datasets\train2\horses', 1, transform)])
v3 = concat_datasets([(r"C:\Users\samue\Desktop\AI\datasets\validate1\elephants", 0, transform),
                             (r'C:\Users\samue\Desktop\AI\datasets\validate2\horses', 1, transform)])
train_loaders, validation_loaders, test_loaders = set_loaders([(tr1,v1,ts1), (tr2,v2,ts2)], batch_size)
train_loaders1, validation_loaders1, test_loaders1 = set_loaders([(tr3,v3,ts1)], batch_size)
alt_train([zebra, horse], epochs, learning_rate, train_loaders, validation_loaders, test_loaders)
equalize_subset(rnest, [testnet1], list(zebra.state_dict().keys())[60:78], divisor = 2)
equalize_subset(zebra, [testnet2], list(zebra.state_dict().keys())[60:78], divisor = 2)
alt_train([testnet1], epochs, learning_rate, train_loaders1, validation_loaders1, test_loaders1)
alt_train([testnet2], epochs, learning_rate, train_loaders1, validation_loaders1, test_loaders1)

