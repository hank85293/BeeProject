import torch
import torch.nn.modules
from   torch.autograd import Variable
import torchvision.models as models
import torchvision.datasets as datasets 
import torchvision.transforms as transforms

import pandas

import numpy as np

from ResNetModel import ResNetModel



def extract_feature() :

    MODEL.eval()
    
    DATA_TRANSFORM = transforms.Compose( [ transforms.Resize( (224,224)) , transforms.ToTensor() ])
    EXTRACT_DATASET = datasets.ImageFolder( root=EXTRACT_DATASET_PATH ,transform=DATA_TRANSFORM)
    EXTRACT_LOADER  = torch.utils.data.DataLoader(EXTRACT_DATASET,batch_size=1)

    print( "Showcase the classes:" , EXTRACT_DATASET.class_to_idx )

    all_feature_list = []

    for  index , ( data , target ) in enumerate( EXTRACT_LOADER ) : 
        if torch.cuda.is_available() :
            
            data , target = data.to(DEVICE),target.to(DEVICE)
        
        feature_list = []
        feature_list.append( EXTRACT_DATASET.classes[ target[ 0 ] ] )  
        
        print( "case%7d:%10s"%( index +1, EXTRACT_DATASET.classes[ target[ 0 ] ] ) )

        data , target = Variable( data ) , Variable( target )

        extract_feature = MODEL( data )

        feature_list.extend( extract_feature.data.cpu().numpy()[0] ) 

        all_feature_list.append( feature_list )

    all_feature_list = np.array( all_feature_list )
    output_csv = pandas.DataFrame( all_feature_list ) 
    output_csv.to_csv( SAVE_CSV_PATH )

EXTRACT_DATASET_PATH = "./image/test"
    
LOAD_MODEL = True
LOAD_MODEL_PATH = "./Epoch40.pkl"

SAVE_CSV = True  
SAVE_CSV_PATH = "./featuresEpoch40.csv"

if __name__ == "__main__" : 
    
    MODEL = ResNetModel()
    
    if torch.cuda.is_available() :
        
        print( "Use GPU for extract features" )
        DEVICE = torch.device( "cuda:0" )
        
        if torch.cuda.device_count() > 1 : 
            print( "Multiple GPU:%d"%(torch.cuda.device_count() ))
            MODEL = torch.nn.DataParallel( MODEL )
        
    else : 
        
        print( "Use CPU for extract features" )
        DEVICE = torch.device( "cpu" )
   
    if LOAD_MODEL == True : 
        
        print( "Load model from :%15s"%( LOAD_MODEL_PATH ))
        MODEL.load_state_dict(torch.load( LOAD_MODEL_PATH ))

    MODEL.fc = torch.nn.LeakyReLU(0.1)
    MODEL.to(DEVICE)
        
    extract_feature()
