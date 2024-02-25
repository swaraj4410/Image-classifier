import numpy as np
import time
import json
import torch
import argparse

from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from PIL import Image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def parse():
    parser = argparse.ArgumentParser(description='predicting the flowers categories')
    parser.add_argument('--img', type=str, default='flowers/test/40/image_04588.jpg', help='pick the path of images to be classified')
    parser.add_argument('--check_point', type=str, default='checkpoint.pth', help='load the checkpoint')
    parser.add_argument('--gpu', type=bool, default=False, help='enable the gpu')
    parser.add_argument('--topK', type=int, default=5, help='Print tokK classes probabilities')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Load the JSON file for labels')
    args=parser.parse_args()
    return args

def load(path):
    checkpoint = torch.load(path)
    if checkpoint['structure'] == 'vgg':
        model=model = models.vgg16(pretrained=True)
    else:
     model=model = models.densenet121(pretrained=True)

    model.to(device)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model


#method to process the image to correct dimension for processing and loading
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    processed_image = Image.open(image).convert('RGB')
    processed_image.thumbnail(size=(256,256))
    width, height = processed_image.size

    # sets new dimensions for center crop
    nw,nh = 224,224 
    left = (width - nw)/2
    top = (height - nh)/2
    right = (width + nw)/2
    bottom = (height + nh)/2
    processed_image = processed_image.crop((left, top, right, bottom))

    # converts to tensor adn normalises
    to_tens = transforms.ToTensor()
    to_norm = transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    tensor = to_norm(to_tens(processed_image))
    
    # convert tensor result to numpy array
    #np_image = np.array(tensor)
    return tensor

def predict(image_path, model,args ,topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    device = torch.device('cuda' if args.gpu else 'cpu')

    model.to(device)
    i_torch = process_image(image_path)
    i_torch = i_torch.unsqueeze(0)
    i_torch = i_torch.float()
    i_torch.to(device)
    model.eval()
    with torch.no_grad():
        output = model.forward(i_torch.cuda())
        ps=torch.exp(output)
    probs, indices = torch.topk(ps, topk)
    #prob = F.softmax(output.data,dim=1)

    Probs = np.array(probs.data[0].cpu())
    Indices = np.array(indices.data[0].cpu())

    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)


    idx_to_class = {idx:Class for Class,idx in model.class_to_idx.items()}
    classes = [idx_to_class[i] for i in Indices]
    labels = [cat_to_name[Class] for Class in classes]

    return Probs,labels


args = parse()
model=load(args.check_point)
probs,classes = predict(args.img, model,args,args.topK)
print(' Possible Type    Probability')
for prob, Class in zip(probs, classes):
    print("%20s: %f" % (Class, prob))



