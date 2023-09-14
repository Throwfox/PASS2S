# for proposed method, _seq2seq:zscore
import os
import numpy as np
import pandas as pd
import torch
from torch.utils import data
from torchvision import datasets

# load data from json file
def getData(road,data_path, mode,selecteddata):

    if mode == 'train':
        print('TrainData from:',road)
        data1 = np.load(os.path.join(data_path, road+'/train_data1_y'+road+'_seq2seq'+selecteddata+'.npy'))
        data2 = np.load(os.path.join(data_path, road+'/train_data2_y'+road+'_seq2seq'+selecteddata+'.npy'))
        data_y =  np.concatenate((data1, data2), axis=0)                  
        datax1 = np.load(os.path.join(data_path,road+ '/train_data1_x'+road+'_seq2seq.npy'))
        datax2 = np.load(os.path.join(data_path, road+'/train_data2_x'+road+'_seq2seq.npy'))
        data_x =  np.concatenate((datax1[:data1.shape[0],:,:], datax2[:data2.shape[0],:,:]), axis=0)             
        return data_x, data_y
    elif mode == 'validation':
        print('Validation from:',road)    
        data_y = np.load(os.path.join(data_path, road+'/test_data_y'+road+'_seq2seq'+selecteddata+'.npy')) 
        data_x = np.load(os.path.join(data_path, road+'/test_data_x'+road+'_seq2seq.npy')) 
        data_x = data_x[:data_y.shape[0],:,:]
        data_x = data_x[:int(data_x.shape[0]*0.25),:,:]
        data_y = data_y[:int(data_y.shape[0]*0.25),:]                 
        return data_x,data_y
    elif mode == 'test':
        print('TestData from:',road)    
        data_y = np.load(os.path.join(data_path, road+'/test_data_y'+road+'_seq2seq'+selecteddata+'.npy')) 
        data_x = np.load(os.path.join(data_path, road+'/test_data_x'+road+'_seq2seq.npy')) 
        data_x = data_x[:data_y.shape[0],:,:]
        data_x = data_x[int(data_x.shape[0]*0.25):,:,:]
        data_y = data_y[int(data_y.shape[0]*0.25):,:]
                    
        return data_x,data_y


# data loader class
class TTPLoader(data.Dataset):
    def __init__(self, road,data_path, mode,selecteddata):
        self.mode = mode
        self.selecteddata=selecteddata
        self.data_x,self.data_y = getData(road,data_path, mode,selecteddata)
        #print(f"> Found {len(self.data)} samples in {self.mode} data...\n")
        
    def get_scaler(self):
        return self.scaler

    def __len__(self):
        if self.selecteddata == '_0d_last1h':
          return len(self.data_x)
        else:
          return len(self.data_y)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        _input = self.data_x[[index]][0]
        _target = self.data_y[[index]][0]
        #print(_input.shape,_target.shape) #(181, 316) (12,)             
        return _input, _target


