#Data:2019/4/23
# Author     : Pin-Hao,Chen
# Program    : ResNet Pre-Trained Model
# Purpose    : load picture and watch their output
# 
# Reference  :
# PyTorch MNIST CNN example
# PyTorch Documentation
# https://www.youtube.com/watch?v=WQwI6vBkv1g
#
# DataSets  :
# https://beerys.github.io/CaltechCameraTraps/       
import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
from   torch.autograd import Variable

import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import numpy as np
        
import pandas 

from ResNetModel import ResNetModel 

def train(epoch , show_plot = False ):
    #switch MODEL to train mode
    MODEL.train()
    #initial SUM_LOSS , total , correct , 
    SUM_LOSS    = 0.0 
    total       = 0.0
    correct     = 0.0  

    #iteration for every batch 
    for batch_index ,data in enumerate( TRAIN_LOADER ) : 
        
        picture_input , picture_class = data
        
        if torch.cuda.is_available() : 

            picture_input , picture_class = picture_input.to(DEVICE) , picture_class.to(DEVICE)
        
        OPTIMIZER.zero_grad()
        output  = MODEL(picture_input)
        loss = F.cross_entropy( output , picture_class )
        loss.backward()
        OPTIMIZER.step()
        
        print( "%15s:%2d/%2d:"%("Epoch",epoch+1,EPOCH))
        print( "%15s:%2d"%("Batch_index",batch_index))
        print( "%15s:%5.2f"%("Loss",loss))
        print( "" )
        
def test( epoch , show_batch_output = False ):
    
    MODEL.eval()

    test_loss   = 0
    correct     = 0
    total       = 0 
    for batch_index , data in enumerate( TEST_LOADER ) : 
        
        picture_input , picture_class = data 
        
        if torch.cuda.is_available() : 
            
            picture_input , picture_class = picture_input.to(DEVICE),picture_class.to(DEVICE)

        picture_input,picture_class=Variable(picture_input),Variable(picture_class)
        output = MODEL(picture_input)
        garbage , predict = torch.max( output.data , 1 )
        
        if predict == picture_class :
            correct += 1

        total += 1 

        if show_batch_output == True : 

            print( "%15s:%d"%("Batch_index",batch_index))
            print( "%15s:%15s"%("Label",TEST_DATASET.classes[picture_class]))
            print( "%15s:%15s"%("Predict",TEST_DATASET.classes[predict[0]]))
            print( "%15s:%2d"%("Correct",correct))
            print( "%15s:%2d"%("Total",total))
            print( "%15s:%5.4f%s"%("Accuracy",correct/total * 100,"%" ))
            print( "" )
        #test_loss += F.cross_entropy( output , target , size_average=False).data[0]
        #pred = output.data.max(1,keepdim=True)[1]
        #correct += pred.eq(target.data.view_as(pred) ).cpu().sum()
    print( "%15s:%2d/%2d"%("EPOCH",epoch+1,EPOCH))
    print( "%15s:%2d"%("Correct",correct))
    print( "%15s:%2d"%("Total",total))
    print( "%15s:%5.4f%s"%("Accuracy",correct/total * 100 ,"%"))
    print( "" )
    """
    test_loss /= len(test_loader.dataset )
    print( "Test set: Average loss: {:.6f}, Accuracy: {}/{} ({:.0f}%) " %(test_loss,
        correct,
        len(test_loader.dataset),
        100. *correct/len(test_loader.dataset)) )
    """
    return None 

SHOW_PLOT   = False 
SHOW_BATCH_OUTPUT = True 

LOAD_MODEL  = True
LOAD_MODEL_PATH = "./Epoch20.pkl"

SAVE_MODEL  = True
SAVE_MODEL_PATH = "./Epoch40.pkl"

USE_GPU     = True

ONLY_TEST   = False 

MODEL = ResNetModel()

if USE_GPU == True and torch.cuda.is_available() :    
    DEVICE = torch.device( "cuda:0" )
    print( "Use GPU for training")
    
    if torch.cuda.device_count() > 1 :
        print( "Multiple GPU" )
        MODEL = torch.nn.DataParallel( MODEL )
    
else : 
    DEVICE = torch.device( "cpu" )
    print( "Use CPU for training")

MODEL.to( DEVICE )


if LOAD_MODEL == True : 
    print( "Load Model from :%s"%(LOAD_MODEL_PATH))
    MODEL.load_state_dict(torch.load( LOAD_MODEL_PATH ))

EPOCH           = 20
BATCH_SIZE      = 128

LR              = 0.1
Momentum        = 0.9

#OPTIMIZER       = optim.SGD(MODEL.parameters(),lr=LR,momentum=Momentum)
OPTIMIZER       = optim.Adam(MODEL.parameters())
DATA_TRANSFORM  = transforms.Compose([ transforms.Resize((224,224) ) , transforms.ToTensor()])

TRAIN_DATASET_PATH = "./image/train"
TRAIN_DATASET   = datasets.ImageFolder(root=TRAIN_DATASET_PATH,transform=DATA_TRANSFORM )
TRAIN_LOADER    = torch.utils.data.DataLoader(TRAIN_DATASET,BATCH_SIZE,True)    

TEST_DATASET_PATH = "./image/test"
TEST_DATASET    = datasets.ImageFolder(root=TEST_DATASET_PATH,transform=DATA_TRANSFORM)
TEST_LOADER     = torch.utils.data.DataLoader(TEST_DATASET,1,True)

#print(TEST_DATASET.class_to_idx)
#print(TEST_DATASET.imgs)

def main( ) : 
    
    if ONLY_TEST == False : 
        
        for epoch in range( EPOCH ) :
            
            train( epoch  )
            #test( epoch , True )
    
    test( 1 , True )
    
    if SAVE_MODEL == True : 
        print( "Save Model to:%s"%(SAVE_MODEL_PATH))
        torch.save(MODEL.state_dict(), SAVE_MODEL_PATH )
        
    return None

main()

# for temporary test

  
    
