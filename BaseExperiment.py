import torch
from tqdm import tqdm
from model import RegressionModel , RegressionModel2
# from segmentation_models_pytorch.losses import FocalLoss
import torch.nn as nn
import torch.optim as optm
import os
import json
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from torchsummary import summary

from sklearn.metrics import f1_score as skf1_score
from CustomLosses import WMSELoss, WMAELoss

class BaseExperiment():
    def __init__(self, trainloader, valloader, training_config, seed=42):
        # TODO: wrap into a function to create work dir
        # Create work dir
        self.work_dir_path = os.path.join('work_dirs', training_config['ex_name'])
        if not os.path.exists(self.work_dir_path):
            os.makedirs(self.work_dir_path)
            
        # self._save_json(custom_model_config, 'model_config.json')
        self._save_json(training_config, 'model_training.json')
            
        self.epochs = training_config['epoch']
        self.patience = training_config['patience']
        self.delta = training_config['delta']
        # self.device = "cuda:0"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Using device: {self.device}')
        in_shape = training_config['in_shape']
        # img_full_shape = training_config['img_shape']
        torch.manual_seed(seed)
        
        self.aux_clssf = training_config['classification_head']
        print('Input shape:', in_shape)
        self.model = RegressionModel(in_shape=in_shape, model_name=training_config['backbone'],\
            aux_clssf=self.aux_clssf, input_channels=training_config['in_shape'][0]).to(self.device)
        # self.model = self._build_model(in_shape, 'resnet18', self.aux_clssf)
        
        print(summary(self.model, tuple(in_shape)))
        self.model= nn.DataParallel(self.model)
        
        self.optm = optm.Adam(self.model.parameters(), lr=training_config['lr'])
        
        self.loss = WMSELoss(weight=1)
        self.mae = WMAELoss(weight=1)
        
        self.ce = nn.CrossEntropyLoss()
        if training_config['clssf_weights'] is not None:
            self.ce = nn.CrossEntropyLoss(weights=torch.Tensor(training_config['clssf_weights']))

        # self.loss = nn.MSELoss()
        # self.mae = nn.L1Loss()
        
        self.trainloader = trainloader
        self.valloader = valloader
        
    # def _build_model(self, in_shape, model_name, aux_clssf=False):
    #     return RegressionModel(in_shape=in_shape, model_name=model_name, aux_clssf=aux_clssf).to(self.device)
        # return RegressionModel2(in_shape=in_shape).to(self.device)
    
    def _save_json(self, data, filename):
        with open(os.path.join(self.work_dir_path, filename), 'w') as f:
            json.dump(data, f)
            
    def train_one_epoch(self):
        train_loss = 0
        
        sum_clssf_loss = 0
        sum_reg_loss = 0
        self.model.train(True)
        for inputs, labels, _, group in tqdm(self.trainloader):
        # for inputs, labels, _ in tqdm(self.trainloader):
            # Zero your gradients for every batch!
            self.optm.zero_grad()
            
            y_pred, y_clssf = self.model(inputs.to(self.device))
            
            loss = self.loss(y_pred, labels.to(self.device))
            
            clssf_loss = self.ce(y_clssf, group.to(self.device))
            
            total_loss = loss
            if self.aux_clssf:
                total_loss = loss + clssf_loss
                
            total_loss.backward()
            
            # Adjust learning weights
            self.optm.step()
            
            train_loss += total_loss.detach()
            sum_clssf_loss += clssf_loss.detach()
            sum_reg_loss += loss.detach()
            
        train_loss = train_loss / len(self.trainloader)
        sum_clssf_loss = sum_clssf_loss / len(self.trainloader)
        sum_reg_loss = sum_reg_loss / len(self.trainloader)
        
        return train_loss, sum_clssf_loss, sum_reg_loss

    def validate_one_epoch(self):
        val_loss = 0
        val_mae = 0
        val_mse = 0
        val_ce = 0
        self.model.eval()
        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for inputs, labels, _, group in tqdm(self.valloader):
            # for inputs, labels, _ in tqdm(self.valloader):
                y_pred, y_clssf = self.model(inputs.to(self.device))
                
                loss = self.loss(y_pred, labels.to(self.device))
                mae = self.mae(y_pred, labels.to(self.device))
                clssf_loss = self.ce(y_clssf, group.to(self.device))
                
                total_loss = loss
                if self.aux_clssf:
                    total_loss = loss + clssf_loss
                
                val_loss += total_loss.detach()
                val_mae += mae.detach()
                val_mse += loss.detach()
                val_ce += clssf_loss.detach()
            
        val_loss = val_loss / len(self.valloader)
        val_mae = val_mae / len(self.valloader)
        val_mse = val_mse / len(self.valloader)
        val_ce = val_ce / len(self.valloader)
        
        return val_loss, val_mae, val_mse, val_ce
    
    def train(self):
        min_val_loss = float('inf')
        early_stop_counter = 0
        for epoch in range(self.epochs):
            
            train_loss, sum_clssf_loss, sum_reg_loss = self.train_one_epoch()
            # train_loss = self.train_one_epoch()
            
            val_loss, val_mae, val_mse, val_ce = self.validate_one_epoch()
            # val_loss, val_mae = self.validate_one_epoch()
            
            
            if val_loss + self.delta < min_val_loss:
                min_val_loss = val_loss
                early_stop_counter = 0
                torch.save(self.model.state_dict(), os.path.join(self.work_dir_path, 'checkpoint.pth'))
            else:
                early_stop_counter += 1
                print(f"Val loss didn't improve! Early Stopping counter: {early_stop_counter}")                
            
            if early_stop_counter >= self.patience:
                print(f'Early Stopping! Early Stopping counter: {early_stop_counter}')
                print(f"Epoch {epoch}: Train Loss = {train_loss:.6f} | Validation Loss = {val_loss:.6f} | Min Validation MAE = {min_val_loss:.6f}")
                break
            
            # print(f"Epoch {epoch}: Train Loss = {train_loss:.6f} | Validation Loss = {val_loss:.6f} | Min Validation MAE = {min_val_loss:.6f}")
            print(f"Epoch {epoch}: Train Loss = {train_loss:.6f} | MSE Loss = {sum_reg_loss:.6f} | CE Loss = {sum_clssf_loss:.6f} | Validation Loss = {val_loss:.6f} | Validation MSE = {val_mse:.6f} | Validation MAE = {val_mae:.6f} | Validation CE = {val_ce:.6f}")
            
            
# def _build_model(self, in_shape, model_name, aux_clssf=False):
#     return RegressionModel(in_shape=in_shape, model_name=model_name, aux_clssf=aux_clssf).to(device)
#     # return RegressionModel2(in_shape=in_shape).to(self.device)
    
def test_model(testloader, training_config):
    work_dir_path = os.path.join('work_dirs', training_config['ex_name'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    in_shape = training_config['in_shape']
        
    model = RegressionModel(in_shape=in_shape, model_name='resnet18',\
        aux_clssf=training_config['classification_head'], input_channels=training_config['in_shape'][0]).to(device)
    # model = _build_model(in_shape, 'resnet18', training_config['classification_head']).to(device)
    
    model= nn.DataParallel(model)
    model.load_state_dict(torch.load(os.path.join(work_dir_path, 'checkpoint.pth')))
    model.eval()
    # model= nn.DataParallel(model)
    
    # mse = nn.MSELoss()
    mse = WMSELoss(weight=1)
    mae = WMAELoss(weight=1)
    # mae = nn.L1Loss()
    
    test_loss = 0.0
    test_mae = 0.0
    # cm = np.zeros((2, 2), dtype=int)
    # Disable gradient computation and reduce memory consumption.
    preds = []
    ids = []
    with torch.no_grad():
        for inputs, study_id in tqdm(testloader):
            y_pred, _ = model(inputs.to(device))
            
            # loss = mse(y_pred, labels.to(device))
            # _mae = mae(y_pred, labels.to(device))
                
            # test_loss += loss.detach()
            # test_mae += _mae.detach()
            
            
            y_pred = y_pred.cpu()[:, 0]
            # Unnormalize the age
            if training_config['classification_head']:
                y_pred = y_pred * (89 - 18) + 18
            # print(study_id.shape, y_pred.shape)
            
            # preds.append([study_id, y_pred])
            preds.append(y_pred)
            ids.append(study_id)

        test_loss = test_loss / (len(testloader))
        test_mae = test_mae / (len(testloader))
    
    # print("======== Metrics ========")
    # print(f'MSE: {test_loss:.6f} | MAE: {test_mae:.6f}')
    
    preds = np.concatenate(preds, axis=0)
    study_ids = np.concatenate(ids, axis=0)
    preds_ids = np.stack([study_ids, preds], axis=0)
    
    print(preds.shape, study_ids.shape)
    # np.save(os.path.join(work_dir_path, 'predictions.npy'), preds)
    
    df = pd.DataFrame(preds_ids.T, columns=['StudyID', 'Age'])
    # df = df['StudyID', 'Age']
    df['StudyID'] = df['StudyID'].astype(int).astype(str)
    df['StudyID'] = df['StudyID'].str.zfill(6)
    # df['StudyID'] = df['StudyID'].apply(lambda x: str(x).str.zfill(5))
    
    df.to_csv(os.path.join(work_dir_path, 'submission_preds.csv'), index=False)
    df['Age'] = df['Age'].round()
    df.to_csv(os.path.join(work_dir_path, 'submission.csv'), index=False)