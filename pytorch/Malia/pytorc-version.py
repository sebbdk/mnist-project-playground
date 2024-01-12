import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
####################################################################################

#Step 1 — Knowing The Dataset
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])
#Meaning of code: 
#transforms.ToTensor() — converts the image into numbers and separates the image into three color channels: red, green & blue. 
#Then it converts the pixels of each image to the brightness of their color between 0 and 255. 
#These values are then scaled down to a range between 0 and 1. The image is now a Torch Tensor.

#transforms.Normalize() — normalizes the tensor with a mean and standard deviation which goes as the two parameters respectively.

####################################################################################
trainset = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=transform)
valset = datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)
#Meaning of code: 
#Download the data sets, shuffle them and transform each of them.
#load them to DataLoader, which combines the data-set 
#and a sampler and provides single- or multi-process iterators over the data-set.

#######################################################################################
#Step 2 — Knowing The Dataset Better

#dataiter = iter(trainloader)
#images, labels = dataiter.next()

#print(images.shape)
#print(labels.shape)

#Meaning of code:
#The shape of images is torch.Size([64,1,28,28]) = there are 64 images in each batch and each image has a dimension of 28 x 28 pixels.
#the labels have a shape as torch.Size([64]). 

#plt.imshow(images[0].numpy().squeeze(), cmap='gray_r');
#displays one image from the training set
#########################################################################################
#display some more images:
#figure = plt.figure()
#num_of_images = 60
#for index in range(1, num_of_images + 1):
#    plt.subplot(6, 10, index)
#    plt.axis('off')
#    plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')

#####################################################################################
#Step 3 — Build The Neural Network
input_size = 784
hidden_sizes = [128, 64]
output_size = 10

model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1))
print(model)

criterion = nn.NLLLoss()
images, labels = next(iter(trainloader))
images = images.view(images.shape[0], -1)

logps = model(images) #log probabilities
loss = criterion(logps, labels) #calculate the NLL loss

print('Before backward pass: \n', model[0].weight.grad)
loss.backward()
print('After backward pass: \n', model[0].weight.grad)

# Step 5 — Core Training Process
optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
time0 = time()
epochs = 15
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)
    
        # Training pass
        optimizer.zero_grad()
        
        output = model(images)
        loss = criterion(output, labels)
        
        #This is where the model learns by backpropagating
        loss.backward()
        
        #And optimizes its weights here
        optimizer.step()
        
        running_loss += loss.item()
    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss/len(trainloader)))
print("\nTraining Time (in minutes) =",(time()-time0)/60)

#Step 6 — Testing & Evaluation
correct_count, all_count = 0, 0
for images,labels in valloader:
  for i in range(len(labels)):
    img = images[i].view(1, 784)
    with torch.no_grad():
        logps = model(img)

    
    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])
    pred_label = probab.index(max(probab))
    true_label = labels.numpy()[i]
    if(true_label == pred_label):
      correct_count += 1
    all_count += 1

print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count/all_count))
