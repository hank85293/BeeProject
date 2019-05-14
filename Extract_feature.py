import torch
import torch.nn.modules
from   torch.autograd import Variable
import torchvision.models as models
import torchvision.datasets as datasets 
import torchvision.transforms as transforms

import pandas

import numpy as np

from ResNetModel_yuzhe import ResNetModel

def extract_feature() :

    MODEL.eval()
    
    EXTRACT_DATASET = datasets.ImageFolder( root="./image",transform=DATA_TRANSFORM)
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
    output_csv.to_csv( "./features.csv" )


if __name__ == "__main__" : 

    if torch.cuda.is_available() :
        
        DEVICE = torch.device( "cuda:0" )
    else : 

        DEVICE = torch.device( "cpu" )
    MODEL=models.resnet34(pretrained=False)    
    fc_features = MODEL.fc.in_features
    MODEL.fc = torch.nn.LeakyReLU(0.1)
    print(MODEL)
    MODEL.to(DEVICE)
    DATA_TRANSFORM = transforms.Compose( [ transforms.Resize( (224,224)) , transforms.ToTensor() ])

    extract_feature()
