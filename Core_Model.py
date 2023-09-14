import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.init as init
from processing import inverse_transform_v3# processing.py
from tqdm import tqdm
import warnings 
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# configure
n_batch = 128
n_workers = 0
n_features = 316
hidden_size = 256

def roads(root_path,road):
    origin_df = pd.read_csv(root_path+'/traffic_data_'+road+'_dataframe.csv')
    trav_mean, trav_std = origin_df['traveltime'].mean(), origin_df['traveltime'].std()
    return origin_df,trav_mean,trav_std     

# model
class stackEncoder_att(nn.Module):
    def __init__(self, input_size=n_features, hidden_size=hidden_size,  batch_size=n_batch):
        super(stackEncoder_att,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.AF = nn.Sequential(
          nn.Tanh(),
          nn.Dropout(0.2)
        )
        self.init_weight()                   
    def init_weight(self):
        init.xavier_normal_(self.lstm.weight_hh_l0)
        init.xavier_normal_(self.lstm.weight_ih_l0)
        self.lstm.bias_ih_l0.data.fill_(0.0)
        self.lstm.bias_hh_l0.data.fill_(0.0)
    def forward(self, x):
         # (seq_len, n_batch, out_feature) ([96, n_batch, 316])
        out, hidden = self.lstm(x)   
        return out, hidden

class stackDecoder_att(nn.Module):    
    def __init__(self, hidden_size, output_size=1, nhead=1, n_layer=1):
        super(stackDecoder_att, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size       
        self.de_lstm = nn.LSTM(1, hidden_size,num_layers=n_layer)
        self.out_fc = nn.Linear(2*hidden_size, output_size)
        #self.out_fc_withoutatt = nn.Linear(hidden_size, output_size)              
        self.self_attention = nn.MultiheadAttention(hidden_size, nhead)
        self.init_weight()
        
    def init_weight(self):
        # orthogonal inittor
        init.xavier_normal_(self.de_lstm.weight_hh_l0)
        init.xavier_normal_(self.de_lstm.weight_ih_l0)
        self.de_lstm.bias_ih_l0.data.fill_(0.0)
        self.de_lstm.bias_hh_l0.data.fill_(0.0)
        
    def forward(self, inpt, last_hidden, encoder_outputs):
        # inpt: [batch]
        # last_hidden: ((hidden, cell)) which has shape of ([1,batch, hidden_size])
        # encoder_outputs = [8,batch,hidden_size] # No. of segments = 8
        inpt = inpt.unsqueeze(0).unsqueeze(2)    # [1, batch, n_feature]
        dec_output, hidden = self.de_lstm(inpt, last_hidden)
        ##### attention##########
        context, attn_weight = self.self_attention(dec_output, encoder_outputs, encoder_outputs)   
        out_input = torch.cat([dec_output, context], 2).squeeze(0)
        output = self.out_fc(out_input)
        return  output, hidden
        
class PASS2S(nn.Module):
    def __init__(self, input_size, hidden_size, 
                 batch_size,nhead=1,output_size=1,teach_force=0.5):
        super(PASS2S, self).__init__()
        self.multi_step = nn.ModuleList()
        for i in range(8):
            encoder = stackEncoder_att(input_size=n_features, hidden_size=hidden_size,  batch_size=n_batch)
            decoder = stackDecoder_att(hidden_size=hidden_size, output_size=1, nhead=1, n_layer=1)
            self.multi_step.append(encoder)
            self.multi_step.append(decoder)
        self.teach_force = teach_force
        self.output_size = output_size
        self.fc=nn.Linear(8,1)

    def forward(self, x, tgt):
        # x: [batch,lenth:96,feature:316] target: [batch,12]            
        batch_size, max_len = x.shape[0], tgt.shape[1]
        x = x.permute(1, 0, 2)
        outputs1 = torch.zeros(max_len, batch_size, self.output_size).cuda() #(12, n_batch, 1)
        outputs2 = torch.zeros(max_len, batch_size, self.output_size).cuda()
        outputs3 = torch.zeros(max_len, batch_size, self.output_size).cuda()
        outputs4 = torch.zeros(max_len, batch_size, self.output_size).cuda()
        outputs5 = torch.zeros(max_len, batch_size, self.output_size).cuda()
        outputs6 = torch.zeros(max_len, batch_size, self.output_size).cuda()
        outputs7 = torch.zeros(max_len, batch_size, self.output_size).cuda()
        outputs8 = torch.zeros(max_len, batch_size, self.output_size).cuda()

        # encoder_output: [seq_len, batch, hidden_size] [8,n_batch,hidden_size]
        # hidden: ([1, batch, hidden_size],cell)
        encoder_output1, hidden1= self.multi_step[0](x[0:12])
        encoder_output2, hidden2= self.multi_step[2](x[12:24])
        encoder_output3, hidden3= self.multi_step[4](x[24:36])
        encoder_output4, hidden4= self.multi_step[6](x[36:48])
        encoder_output5, hidden5= self.multi_step[8](x[48:60])
        encoder_output6, hidden6= self.multi_step[10](x[60:72])
        encoder_output7, hidden7= self.multi_step[12](x[72:84])
        encoder_output8, hidden8= self.multi_step[14](x[84:96])

        # first-setp input for decode---the current travel time r#
        output1=x[-1,:,0] #(n_batch)
        output2=x[-1,:,0]
        output3=x[-1,:,0]
        output4=x[-1,:,0]
        output5=x[-1,:,0]
        output6=x[-1,:,0]
        output7=x[-1,:,0]
        output8=x[-1,:,0]

        use_teacher = random.random() < self.teach_force
        if use_teacher:
            for t in range(max_len):
                output1, hidden1 = self.multi_step[1](output1, hidden1, encoder_output1)
                output2, hidden2 = self.multi_step[3](output2, hidden2, encoder_output2) 
                output3, hidden3 = self.multi_step[5](output3, hidden3, encoder_output3)
                output4, hidden4 = self.multi_step[7](output4, hidden4, encoder_output4)
                output5, hidden5 = self.multi_step[9](output5, hidden5, encoder_output5)
                output6, hidden6 = self.multi_step[11](output6, hidden6, encoder_output6)
                output7, hidden7 = self.multi_step[13](output7, hidden7, encoder_output7)
                output8, hidden8 = self.multi_step[15](output8, hidden8, encoder_output8) #output n_batch,1
                
                outputs1[t] = output1
                outputs2[t] = output2
                outputs3[t] = output3
                outputs4[t] = output4
                outputs5[t] = output5
                outputs6[t] = output6
                outputs7[t] = output7
                outputs8[t] = output8

                output1 = tgt[:,t]
                output2 = tgt[:,t]
                output3 = tgt[:,t]
                output4 = tgt[:,t]
                output5 = tgt[:,t]
                output6 = tgt[:,t]
                output7 = tgt[:,t]
                output8 = tgt[:,t]
        else:
            for t in range(max_len):
                output1, hidden1 = self.multi_step[1](output1, hidden1, encoder_output1)
                output2, hidden2 = self.multi_step[3](output2, hidden2, encoder_output2) 
                output3, hidden3 = self.multi_step[5](output3, hidden3, encoder_output3)
                output4, hidden4 = self.multi_step[7](output4, hidden4, encoder_output4)
                output5, hidden5= self.multi_step[9](output5, hidden5, encoder_output5)
                output6, hidden6 = self.multi_step[11](output6, hidden6, encoder_output6)
                output7, hidden7 = self.multi_step[13](output7, hidden7, encoder_output7)
                output8, hidden8 = self.multi_step[15](output8, hidden8, encoder_output8) #output 128,1
                
                outputs1[t] = output1
                outputs2[t] = output2
                outputs3[t] = output3
                outputs4[t] = output4
                outputs5[t] = output5
                outputs6[t] = output6
                outputs7[t] = output7
                outputs8[t] = output8

                output1 = output1.squeeze(1).detach() # output from [128,1] to [128]
                output2 = output2.squeeze(1).detach()
                output3 = output3.squeeze(1).detach()
                output4 = output4.squeeze(1).detach() 
                output5 = output5.squeeze(1).detach() 
                output6 = output6.squeeze(1).detach() 
                output7 = output7.squeeze(1).detach() 
                output8 = output8.squeeze(1).detach()

        # [max_len, batch, output_size] #(12, 128, 1)  for outputs1
        final_outputs=torch.cat((outputs1,outputs2,outputs3,outputs4,outputs5,outputs6,outputs7,outputs8),2) #(12,128,8)
        outputs=self.fc(final_outputs) ##(12,128,1)
        outputs=outputs.squeeze(2).permute(1, 0)

        return outputs
    
    def predict(self, x,max_len):
        with torch.no_grad():
            batch_size = x.shape[0]
            x = x.permute(1, 0, 2)
            outputs1 = torch.zeros(max_len, batch_size, self.output_size).cuda() #(12, 128, 1)
            outputs2 = torch.zeros(max_len, batch_size, self.output_size).cuda()
            outputs3 = torch.zeros(max_len, batch_size, self.output_size).cuda()
            outputs4 = torch.zeros(max_len, batch_size, self.output_size).cuda()
            outputs5 = torch.zeros(max_len, batch_size, self.output_size).cuda()
            outputs6 = torch.zeros(max_len, batch_size, self.output_size).cuda()
            outputs7 = torch.zeros(max_len, batch_size, self.output_size).cuda()
            outputs8 = torch.zeros(max_len, batch_size, self.output_size).cuda()

            # encoder_output: [seq_len, batch, hidden_size] [8,128,256]
            # hidden: ([1, batch, hidden_size],cell)
            encoder_output1, hidden1= self.multi_step[0](x[0:12])
            encoder_output2, hidden2= self.multi_step[2](x[12:24])
            encoder_output3, hidden3= self.multi_step[4](x[24:36])
            encoder_output4, hidden4= self.multi_step[6](x[36:48])
            encoder_output5, hidden5= self.multi_step[8](x[48:60])
            encoder_output6, hidden6= self.multi_step[10](x[60:72])
            encoder_output7, hidden7= self.multi_step[12](x[72:84])
            encoder_output8, hidden8= self.multi_step[14](x[84:96])

            output1=x[-1,:,0] #(128)
            output2=x[-1,:,0]
            output3=x[-1,:,0]
            output4=x[-1,:,0]
            output5=x[-1,:,0]
            output6=x[-1,:,0]
            output7=x[-1,:,0]
            output8=x[-1,:,0]
            for t in range(max_len):
                output1, hidden1 = self.multi_step[1](output1, hidden1, encoder_output1)
                output2, hidden2 = self.multi_step[3](output2, hidden2, encoder_output2) 
                output3, hidden3 = self.multi_step[5](output3, hidden3, encoder_output3)
                output4, hidden4 = self.multi_step[7](output4, hidden4, encoder_output4)
                output5, hidden5= self.multi_step[9](output5, hidden5, encoder_output5)
                output6, hidden6 = self.multi_step[11](output6, hidden6, encoder_output6)
                output7, hidden7 = self.multi_step[13](output7, hidden7, encoder_output7)
                output8, hidden8 = self.multi_step[15](output8, hidden8, encoder_output8) #output 128,1
                
                outputs1[t] = output1
                outputs2[t] = output2
                outputs3[t] = output3
                outputs4[t] = output4
                outputs5[t] = output5
                outputs6[t] = output6
                outputs7[t] = output7
                outputs8[t] = output8

                output1 = output1.squeeze(1).detach() # output from [128,1] to [128]
                output2 = output2.squeeze(1).detach()
                output3 = output3.squeeze(1).detach()
                output4 = output4.squeeze(1).detach() 
                output5 = output5.squeeze(1).detach() 
                output6 = output6.squeeze(1).detach() 
                output7 = output7.squeeze(1).detach() 
                output8 = output8.squeeze(1).detach() 
        final_outputs=torch.cat((outputs1,outputs2,outputs3,outputs4,outputs5,outputs6,outputs7,outputs8),2) #(12,128,8)
        outputs=self.fc(final_outputs) ##(12,128,1)
        outputs=outputs.squeeze(2).permute(1, 0)
        return outputs
    
# forecasting
def make_forecast(model, loader,maxlen,trav_mean,trav_std): 
    # maxlen=12 (1 hour)
    # trave_mean and std to use for recovering the z-score to original data.
    #loader['train'] data: x(normalized) [batch seq:96 feature:316]; y [batch seq:12]
    model.eval()
    actual, predicted = list(), list()
    
    for batch_x, batch_y in tqdm(loader['test']): 
        inputs, y_true = Variable(batch_x.to(device).float()), Variable(batch_y.to(device).float())
        with torch.no_grad():
            y_pred = model.predict(inputs,maxlen)
            y_true = inverse_transform_v3(trav_mean, trav_std, y_true)                              
            actual.extend(y_true.cpu().detach().numpy())
            predicted.extend(y_pred.cpu().detach().numpy())
    actual, predicted = np.array(actual), np.array(predicted)
    return actual, predicted
