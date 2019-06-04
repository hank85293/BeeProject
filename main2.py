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
from    torch.utils import data 

import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import numpy as np
        
import pandas 

from ResNetModel import ResNetModel 

def train(cross_validation_index , epoch , MODEL , DEVICE , TRAIN_SET , show_plot = False ):
    #switch MODEL to train mode
    MODEL.train()

    TRAIN_LOADER = torch.utils.data.DataLoader( TRAIN_SET , BATCH_SIZE , True )

    OPTIMIZER       = optim.Adam(MODEL.parameters())

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
        
        print( "%15s:%2d"%( "Cross_validation" ,  cross_validation_index + 1 ) )
        print( "%15s:%2d/%2d:"%("Epoch",epoch+1,EPOCH))
        print( "%15s:%2d/%2d"%("Batch_index",batch_index , len( TRAIN_SET ) / BATCH_SIZE + 1 ))
        print( "%15s:%5.2f"%("Loss",loss))
        print( "" )

    

def val( cross_validation_index , MODEL , DEVICE , VALIDATION_SET , show_validation_output = False ):

    MODEL.eval()

    VALIDATION_LOADER = torch.utils.data.DataLoader( VALIDATION_SET , 1 , True )

    VAL_LOSS    = 0.0
    val_correct = 0
    val_total   = 0
    for batch_index , data in enumerate( VALIDATION_LOADER ) :

        picture_input , picture_class = data

        if torch.cuda.is_available() : 

            picture_input , picture_class = picture_input.to( DEVICE ) , picture_class.to( DEVICE)

        picture_input , picture_class = Variable( picture_input ) , Variable( picture_class )
        output = MODEL( picture_input )
        garbage , predict = torch.max( output.data , 1 )

        if predict == picture_class : 
            val_correct += 1 
        
        val_total += 1 

        if show_validation_output == True : 

            print( "%15s:%d"%("Batch_index",batch_index + 1 ) )
            print( "%15s:%15s"%("Label",VALIDATION_SET.classes[picture_class]))
            print( "%15s:%15s"%("Predict",VALIDATION_SET.classes[predict[0]]))
            print( "%15s:%2d"%("Correct",val_correct))
            print( "%15s:%2d"%("Total",val_total))
            print( "%15s:%5.4f%s"%("Accuracy",val_correct/val_total * 100,"%" ))
            print( "" )

    print( "%15s:%2d"%( "Cross_validation", cross_validation_index + 1 ) )
    print( "%15s:%2d"%("Correct",val_correct))
    print( "%15s:%2d"%("Total",val_total))
    print( "%15s:%5.4f%s"%("Accuracy",val_correct/val_total * 100 ,"%"))
    print( "" )
    print( "%20s"%("Save model accuracy" ) )
    all_cross_validation_accuracy.append( val_correct/val_total )
    print( "%20s"%( "Done" ) ) 
    print( "" )

def test( MODEL , DEVICE , TEST_SET , show_batch_output = False ):
    
    MODEL.eval()

    TEST_LOADER = torch.utils.data.DataLoader( TEST_SET , 1 , True )

    test_loss   = 0
    test_correct     = 0
    test_total       = 0 

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
            print( "%15s:%2d"%("Correct",test_correct))
            print( "%15s:%2d"%("Total",test_total))
            print( "%15s:%5.4f%s"%("Accuracy",test_correct/test_total * 100,"%" ))
            print( "" )

    
    print( "%15s:%2d"%("Correct",test_correct))
    print( "%15s:%2d"%("Total",test_total))
    print( "%15s:%5.4f%s"%("Accuracy",test_correct/test_total * 100 ,"%"))
    print( "" )
    
    return None 

SHOW_PLOT   = False 
SHOW_BATCH_OUTPUT = True 

LOAD_MODEL  = False 
LOAD_MODEL_PATH = "./Epoch20.pkl"

SAVE_MODEL  = True
SAVE_MODEL_PATH = "./Cross1.pkl"

USE_GPU     = True

ONLY_TEST   = False 


CROSS_VALIDATION = 5 

EPOCH           = 1
BATCH_SIZE      = 128

LR              = 0.1
Momentum        = 0.9

#OPTIMIZER       = optim.SGD(MODEL.parameters(),lr=LR,momentum=Momentum)

DATA_TRANSFORM  = transforms.Compose([ transforms.Resize((224,224) ) , transforms.ToTensor()])

TRAIN_DATASET_PATH = "./image/train"
TRAIN_DATASET   = datasets.ImageFolder(root=TRAIN_DATASET_PATH,transform=DATA_TRANSFORM )

TEST_DATASET_PATH = "./image/test"
TEST_DATASET    = datasets.ImageFolder(root=TEST_DATASET_PATH,transform=DATA_TRANSFORM)

#print(TEST_DATASET.class_to_idx)
#print(TEST_DATASET.imgs)

all_cross_validation_parameters = []
all_cross_validation_accuracy = []

def chooseBestModel() : 

    print( "%20s:"%( "Show all accuracy" ) + all_cross_validation_accuracy )
    best_accuracy = 0 
    best_index = 0 

    for index in range( all_cross_validation_accuracy ) : 
        
        if all_cross_validation_accuracy[ index ] > best_accuracy : 

            best_accuracy = all_cross_validation_accuracy[ index ]
            best_index = index
    
    print( "%20s:%5.4f"%("Best model accuracy" , best_accuracy ) ) 
    print( "" ) 
    return best_index 

def main( ) : 
    
    if ONLY_TEST == False : 

        for cross_validation_index in range( CROSS_VALIDATION ) : 
            
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
            
            
            print( "%20s:%2d"%( "Cross_validation" , cross_validation_index + 1 ) )
            print( "%20s:%d"%( "Dataset length" , len( TRAIN_DATASET ) ) )
            TRAIN_LENGTH = int( len( TRAIN_DATASET ) * 0.8 )
            VALIDATION_LENGTH = len( TRAIN_DATASET ) - TRAIN_LENGTH 
            TRAIN_SET , VALIDATION_SET = data.random_split( TRAIN_DATASET , ( TRAIN_LENGTH , VALIDATION_LENGTH ) ) 
            print( "%20s:%d"%( "Train set length" , len( TRAIN_SET ) ) )
            print( "%20s:%d"%( "Validation set length" , len( VALIDATION_SET ) ) )
            print( "" )
            
            for epoch in range( EPOCH ) :
                
                train( cross_validation_index , epoch , MODEL , DEVICE , TRAIN_SET )
            
            print( "%20s"%( "Save model parameters" ) )
            all_cross_validation_parameters.append( MODEL.state_dict() )
            print( "%20s"%( "Done" ) )

            val( cross_validation_index , MODEL , DEVICE , VALIDATION_SET )

    best_index = chooseBestModel()
    BESTMODEL = ResNetModel()
    BESTMODEL.load_state_dict( all_cross_validation_parameters[ best_index ] ) 
    test( BESTMODEL , DEVICE , TEST_DATASET , True )
    
    if SAVE_MODEL == True : 
        print( "Save Model to:%s"%(SAVE_MODEL_PATH))
        torch.save(MODEL.state_dict(), SAVE_MODEL_PATH )
        
    return None

main()

# for temporary test

  
    
