import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import copy
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import dataloader as dataloader # dataloader.py
from processing import inverse_transform_v3# processing.py
import utils
from tqdm import tqdm
from Core_Model import *
import warnings 
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('\nINFO:')
print("PyTorch Version: ", torch.__version__)
print('GPU State:', device, '\n')

# configure

n_batch = 128
n_workers = 0
n_features = 316
n_epochs = 10
hidden_size = 256
lr = 1e-4 
weight_decay = 0
alpha=0.1 #weight for KNR

def roads(root_path,road):
    origin_df = pd.read_csv(root_path+'/traffic_data_'+road+'_dataframe.csv')
    trav_mean, trav_std = origin_df['traveltime'].mean(), origin_df['traveltime'].std()
    return origin_df,trav_mean,trav_std     

# data loader
def create_dataloader(road,root_path,selecteddata):
    train_dataset = dataloader.TTPLoader(road,data_path=root_path, mode="train",selecteddata=selecteddata) 
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, 
        batch_size=n_batch, 
        shuffle=True, 
        num_workers=n_workers
    )
    validation_dataset = dataloader.TTPLoader(road,data_path=root_path, mode="validation",selecteddata=selecteddata) 
    validation_loader = torch.utils.data.DataLoader(
        dataset=validation_dataset, 
        batch_size=n_batch, 
        shuffle=True, 
        num_workers=n_workers
    )    
    test_dataset = dataloader.TTPLoader(road,data_path=root_path, mode="test",selecteddata=selecteddata) 
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, 
        batch_size=n_batch, 
        shuffle=True, 
        num_workers=n_workers
    )
    loaders = {'train':train_loader,'validation':validation_loader, 'test':test_loader}
    return loaders

# train and validation process
def fit_lstm(model, loader, n_epochs, n_neurons, learning_rate, optimizer, opt_scheduler,writer,selecteddata,trav_mean,trav_std,road):
    best_mae  = 99999
    criteria = nn.MSELoss()
    for epoch in range(n_epochs):  
        for phase in ['train','validation','test']:
            print(f'Epoch {epoch+1}/{n_epochs} ({phase} phase)')       
            if phase == 'train':
                model.train()  # Set model to training mode
            elif phase == 'validation':
                model.eval()                  
            elif phase == 'test':
                model.eval()            
            running_loss = 0.0
            actual = []
            predicted = []
                    
            for batch_x, batch_y in tqdm(loader[phase]):           
                inputs, y_true = Variable(batch_x.to(device).float()), Variable(batch_y.to(device).float())               
                optimizer.zero_grad()                
                with torch.set_grad_enabled(phase == 'train'):
                    y_true = inverse_transform_v3(trav_mean, trav_std, y_true)
                    if phase == 'train':
                      y_pred = model(inputs, y_true)
                    else:
                      y_pred = model.predict(inputs,y_true.shape[1])                 
                    loss = criteria(y_pred, y_true)                   
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        writer.add_scalar('train/loss', loss, epoch)
                actual.extend(y_true.cpu().detach().numpy())
                predicted.extend(y_pred.cpu().detach().numpy())
                running_loss += loss.item() # * inputs.size(0)
                                              
            # loss and metrics
            actual, predicted = np.array(actual), np.array(predicted)
            if phase == 'train':
                lstm_train_mse = mean_squared_error(actual, predicted,squared=True)
                lstm_train_rmse = mean_squared_error(actual, predicted,squared=False)
                lstm_train_mape = mean_absolute_percentage_error(actual, predicted)
                lstm_train_mae = mean_absolute_error(actual, predicted)                
                lstm_train_smape = utils.symmetric_mean_absolute_error(actual, predicted)
                print(f'| {phase} mse: {lstm_train_mse:.5f} | rmse: {lstm_train_rmse:.5f} | mape: {lstm_train_mape:.5f} | smape: {lstm_train_smape:.5f} | mae: {lstm_train_mae:.5f}')
                writer.add_scalar('train/rmse', lstm_train_rmse, epoch)
                writer.add_scalar('train/mae', lstm_train_mae, epoch)
                writer.add_scalar('train/mape', lstm_train_mape, epoch)
                #wandb.log({"Epoch mse train loss (lstm)": lstm_train_mse, "Epoch rmse train loss (lstm)": lstm_train_rmse, "Epoch mape train loss (lstm)": lstm_train_mape, "Epoch smape train loss (lstm)": lstm_train_smape,"Epoch mae train loss (lstm)": lstm_train_mae}, step=epoch, commit=False)
            elif phase == 'validation':
                lstm_val_mse = mean_squared_error(actual, predicted,squared=True)
                lstm_val_rmse = mean_squared_error(actual, predicted,squared=False)
                lstm_val_mape = mean_absolute_percentage_error(actual, predicted)
                lstm_val_mae = mean_absolute_error(actual, predicted) 
                lstm_val_smape = utils.symmetric_mean_absolute_error(actual, predicted)
                print(f'| {phase} mse: {lstm_val_mse:.5f} | rmse: {lstm_val_rmse:.5f} | mape: {lstm_val_mape:.5f} | smape: {lstm_val_smape:.5f} | mae: {lstm_val_mae:.5f}')
                writer.add_scalar('val/rmse', lstm_val_rmse, epoch)
                writer.add_scalar('val/mae', lstm_val_mae, epoch)
                writer.add_scalar('val/mape', lstm_val_mape, epoch)
                writer.add_scalar('val/smape', lstm_val_smape, epoch) 
            elif phase == 'test':
                lstm_test_mse = mean_squared_error(actual, predicted,squared=True)
                lstm_test_rmse = mean_squared_error(actual, predicted,squared=False)
                lstm_test_mape = mean_absolute_percentage_error(actual, predicted)
                lstm_test_mae = mean_absolute_error(actual, predicted) 
                lstm_test_smape = utils.symmetric_mean_absolute_error(actual, predicted)
                print(f'| {phase} mse: {lstm_test_mse:.5f} | rmse: {lstm_test_rmse:.5f} | mape: {lstm_test_mape:.5f} | smape: {lstm_test_smape:.5f} | mae: {lstm_test_mae:.5f}')
                writer.add_scalar('test/rmse', lstm_test_rmse, epoch)
                writer.add_scalar('test/mae', lstm_test_mae, epoch)
                writer.add_scalar('test/mape', lstm_test_mape, epoch)
                writer.add_scalar('test/smape', lstm_test_smape, epoch)                 
                if lstm_test_mae < best_mae:
                    best_mae = lstm_test_mae
                    best_model_wts = copy.deepcopy(model.state_dict())
                    print('Save model')
                    torch.save(best_model_wts, 'modelsaved/'+road+'/PASS2S_'+selecteddata+'.pt')                                           
        opt_scheduler.step()

    model.load_state_dict(best_model_wts)
    return model

def main():
    road_segments=['nfb0370']
    root_path="Data/"
    last_time = 1
    for road in road_segments:
        origin_df,trav_mean,trav_std=roads(root_path,road)
        for predict_time in range(5,7):    
            selecteddata = '_'+str(predict_time)+'d_last'+str(last_time)+'h'
            maxlen=12*last_time 
            writer = SummaryWriter('modelsaved/stackseq2seq/'+road+str(n_epochs)+selecteddata+str(lr))
            dataloaders = create_dataloader(road,root_path,selecteddata)        
            model = PASS2S(input_size=n_features, hidden_size=hidden_size, output_size=1, batch_size=n_batch)
            model.to(device)
         
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            opt_scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
        
            # fit model
            model = fit_lstm(model, dataloaders, n_epochs, hidden_size, lr, optimizer, opt_scheduler,writer,selecteddata,trav_mean,trav_std,road)
            
            # make forecasts
            actual, forecasts = make_forecast(model, dataloaders,maxlen,trav_mean,trav_std)
        
            #print(forecasts)
            rmse = mean_squared_error(actual, forecasts,squared=False)
            mape = mean_absolute_percentage_error(actual, forecasts)
            mae = mean_absolute_error(actual, forecasts)
            smape = utils.symmetric_mean_absolute_error(actual, forecasts)
            print('Final result',f'|  mae: {mae:.5f} | rmse: {rmse:.5f} | mape: {mape:.5f} | smape: {smape:.5f}')
            print("\ndone\n")


if __name__ == "__main__":
    main()
