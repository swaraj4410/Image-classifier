#importing all the necessary modules
import argparse
import os
import time
import torch
import matplotlib.pyplot as plt 
import numpy as np
import json

from torch import nn, optim
from torchvision import datasets, models, transforms
from collections import OrderedDict
from PIL import Image

def input():
    parser = argparse.ArgumentParser(description='train neural network')
    parser.add_argument('data_directory', help='its a mandatory data directory')
    parser.add_argument('--save_dir', help='directory for output.')
    parser.add_argument('--arch', help='different models to choose for ex. [vgg,resnet]')
    parser.add_argument('--hidden_units', help='number of hidden units')
    parser.add_argument('--learning_rate', help='adjust the learning rate')
    parser.add_argument('--epochs', help='epochs')
    parser.add_argument('--gpu',action='store_true', help='gpu')
    args = parser.parse_args()
    return args


def validate(args):

    if (args.gpu and not torch.cuda.is_available()):
        raise Exception('GPU demanded but its not available')
    
    if(not os.path.isdir(args.data_directory)):
        raise Exception('directory does not exist!')
    
    if args.arch not in ('vgg','densenet',None):
        raise Exception('as of now only VGG and RESNET are available') 
    

def process(data_dir):
    #train_dir, test_dir, valid_dir = data_dir
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(data_dir + '\\'+ 'train', transform=train_transforms)
    test_datasets = datasets.ImageFolder(data_dir + '\\'+ 'test', transform=test_transforms)
    valid_datasets = datasets.ImageFolder(data_dir + '\\' + 'valid', transform=test_transforms)
    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_dataloaders = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
    test_dataloaders = torch.utils.data.DataLoader(test_datasets, batch_size=64)
    valid_dataloaders = torch.utils.data.DataLoader(valid_datasets, batch_size=64)

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    loader={'train':train_dataloaders,'test':test_dataloaders,'valid':valid_dataloaders,'label':cat_to_name,'class_to_idx':train_datasets.class_to_idx}

    return loader




#testing the model
def test(model,loader,device='cuda'):
    test_accuracy = 0

    # defines tests
    for images,labels in loader['test']:
        model.eval()
        images,labels = images.to(device),labels.to(device)
        log_ps = model.forward(images)
        ps = torch.exp(log_ps)
        top_ps,top_class = ps.topk(1,dim=1)
        matches = (top_class == labels.view(*top_class.shape)).type(torch.FloatTensor)
        accuracy = matches.mean()
        test_accuracy += accuracy
        
    print(f'Test Accuracy: {test_accuracy/len(loader["test"])*100:.2f}%')  
    return None  

#make the model and do the learning steps
     
def build_learn(loader,args):
    #loading the appropriate model
    if (args.hidden_units == None):
        hidden_units = 4096
    else:
        hidden_units = int(args.hidden_units)



    if args.arch is None:
        model=models.vgg16(pretrained=True)
        input=25088
        #freeze the convolutional parameters
        for param in model.parameters():
            param.requires_grad = False

        classifier = nn.Sequential(OrderedDict([
             ('fc1', nn.Linear(input, hidden_units)),
             ('relu', nn.ReLU()),
             ('drop1', nn.Dropout(0.3)),
             ('fc2', nn.Linear(hidden_units, 102)),
             ('output', nn.LogSoftmax(dim=1))
             ]))
        model.classifier=classifier
   
    else:
        if args.arch == 'vgg':
            model=models.vgg16(pretrained = True)
            input=25088
            #freeze the gradients of hidden units

            for param in model.parameters():
                param.requires_grad = False

            classifier = nn.Sequential(OrderedDict([
             ('fc1', nn.Linear(input, hidden_units)),
             ('relu', nn.ReLU()),
             ('drop1', nn.Dropout(0.3)),
             ('fc2', nn.Linear(hidden_units, 102)),
             ('output', nn.LogSoftmax(dim=1))
             ]))
            model.classifier=classifier
 

        if args.arch == 'densenet':
            model=models.densenet121(pretrained=True)
            input=1024
            #freeze the gradients of hidden units
            for param in model.parameters():
                param.requires_grad = False
            
            classifier = nn.Sequential(OrderedDict([
             ('fc1', nn.Linear(input, 102)),
             ('output', nn.LogSoftmax(dim=1))
             ]))
            model.classifier=classifier
            

    
    



    #defining the criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    if args.gpu:
        device='cuda'
    else:
        device='cpu'
    model.to(device)

    epochs = int(args.epochs)
    steps = 0
    running_loss = 0
    print_every = 32

    for epoch in range(epochs):
        for inputs, labels in loader['train']:
            steps += 1
            # Move input and label tensors to the default device
            #print('moving to {}'.format(device))
            inputs, labels = inputs.to(device), labels.to(device)
            
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            
            optimizer.zero_grad()
            #loss.requires_grad = True
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in loader['valid']:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        
                        test_loss += batch_loss.item()
                        
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        
                print(f"Epoch {epoch+1}/{epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Test loss: {test_loss/len(loader['valid']):.3f}.. "
                    f"Test accuracy: {accuracy/len(loader['valid'])*100:.3f}%")
                running_loss = 0
                model.train()
    return model


def save_model(model,args):
    print('Saving model')

    if args.save_dir is None:
        save_dir= 'checkpoint.pth'
    else:
        save_dir=args.save_dir

    checkpoint = {
                'structure':args.arch,
                'model': model.cpu(),
                'features': model.features,
                'classifier': model.classifier,
                'state_dict': model.state_dict(),
                'class_to_idx':model.class_to_idx
    }
    torch.save(checkpoint, save_dir)


def main():
    print('training has started \n')
    args=input()
    validate(args)
    print('validation is ok and alright *****')

    loader_dict=process(args.data_directory)
    ctx=loader_dict['class_to_idx']
    model=build_learn(loader_dict,args)
    model.class_to_idx = ctx
    print('MODEL TRAINED AND IS READY TO BE TESTED')
    print('TESTING the model now')
    if args.gpu:
        device = 'cuda'
    else:
        device ='cpu'
    test(model,loader_dict,device)

    print('SAVING THE MODEL NOW')

    save_model(model,args)


    print('MODEL TRAINING FINISHED')
    
    return None

main()











