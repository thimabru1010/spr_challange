import torch
from tqdm import tqdm
from model import RegressionModel , RegressionModel2
# from segmentation_models_pytorch.losses import FocalLoss
import torch.nn as nn
import torch.optim as optm
from torch.optim.lr_scheduler import StepLR
import os
import json
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
# from torchsummary import summary
from torchinfo import summary

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
        
        if training_config['backbone'] != 'swin':
            print(summary(self.model, (training_config['batch_size'], in_shape[0], in_shape[1], in_shape[2])))
        else:
            print(self.model)
        self.model= nn.DataParallel(self.model)
        
        if training_config['optimizer'] == 'adam':
            self.optm = optm.Adam(self.model.parameters(), lr=training_config['lr'])
        elif training_config['optimizer'] == 'sgd':
            self.optm = optm.SGD(self.model.parameters(), lr=training_config['lr'], momentum=training_config['momentum'])
        
        self.scheduler = StepLR(self.optm, step_size=training_config['sched_step_size'],\
            gamma=training_config['sched_decay_factor'])
        
        self.loss = nn.MSELoss()        
        self.mae = nn.L1Loss()
        
        self.ce = nn.CrossEntropyLoss()
        if training_config['clssf_weights'] is not None:
            self.ce = nn.CrossEntropyLoss(weight=torch.Tensor(training_config['clssf_weights']))

        # self.loss = nn.MSELoss()
        # self.mae = nn.L1Loss()
        
        self.trainloader = trainloader
        self.valloader = valloader
        
        # self.use_volume = False
        # if (training_config['n_slices'] == training_config['in_shape'][0]):
        #     self.use_volume = True
        self.use_volume = training_config['n_slices'] != training_config['in_shape'][0]
    
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
            
            # print(inputs.shape)
            y_pred, y_clssf = self.model(inputs.to(self.device))

            # print(y_pred)
            # print(y_pred.shape)
            # print(labels)
            loss = self.loss(y_pred[:, 0], labels.to(self.device))
            
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
                # mae = self.mae(y_pred*(89 - 18) + 18, labels.to(self.device)*(89 - 18) + 18)                
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
        
        
    def validate_one_epoch_vol(self):
        val_loss = 0
        val_mae = 0
        val_mse = 0
        val_acc = 0
        val_ce = 0
        self.model.eval()
        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for inputs, labels, _, group in tqdm(self.valloader):            
                
                # print(inputs.shape, labels.shape)
                inputs = inputs.permute(1, 0, 2, 3)
                y_pred_s, y_clssf_s = self.model(inputs.to(self.device))
                # print(y_pred_s.shape, y_clssf_s.shape)
                
                # print(y_pred_s.shape, labels.shape)
                y_pred = y_pred_s.mean(dim=0)
                
                # print(y_pred.shape, labels.shape)
                loss = self.loss(y_pred, labels.to(self.device))
                mae = self.mae(y_pred, labels.to(self.device))
                # mae = self.mae(y_pred*(89 - 18) + 18, labels.to(self.device)*(89 - 18) + 18)                 
                
                total_loss = loss
                if self.aux_clssf:
                    # y_class = np.bincount(y_clssf_s).argmax()
                    y_class = y_clssf_s.mean(dim=0)
                    clssf_loss = self.ce(y_class, group.to(self.device))
                    val_ce += clssf_loss.detach()
                    total_loss = loss + clssf_loss
                    
                    torch.softmax(y_clssf_s, dim=1)
                    y_class = y_clssf_s.mean(dim=0).argmax()
                    acc = (y_class == group).sum()
                    val_acc += acc.detach()
                    
                    
                
                val_loss += total_loss.detach()
                val_mae += mae.detach()
                val_mse += loss.detach()
                # val_ce += clssf_loss.detach()
            
            
        val_loss = val_loss / len(self.valloader)
        val_mae = val_mae / len(self.valloader)
        val_mse = val_mse / len(self.valloader)
        val_acc = val_acc / len(self.valloader)
        val_ce = val_ce / len(self.valloader)
        
        return val_loss, val_mae, val_mse, val_acc    
    
    
    def train(self):
        min_val_loss = float('inf')
        early_stop_counter = 0
        for epoch in range(self.epochs):
            
            train_loss, sum_clssf_loss, sum_reg_loss = self.train_one_epoch()
            
            if self.use_volume:
                val_loss, val_mae, val_mse, val_ce = self.validate_one_epoch_vol()
            else:
                val_loss, val_mae, val_mse, val_ce = self.validate_one_epoch()
            
            last_lr = self.scheduler.get_last_lr()[0]
            self.scheduler.step()
            
            if val_loss + self.delta < min_val_loss:
                min_val_loss = val_loss
                early_stop_counter = 0
                torch.save(self.model.state_dict(), os.path.join(self.work_dir_path, 'checkpoint.pth'))
            else:
                early_stop_counter += 1
                print(f"Val loss didn't improve! Early Stopping counter: {early_stop_counter}")                
            
            if early_stop_counter >= self.patience:
                print(f'Early Stopping! Early Stopping counter: {early_stop_counter}')              
                break
            
            if self.aux_clssf:
                print(f"Epoch {epoch}: Train Loss = {train_loss:.6f} | MSE Loss = {sum_reg_loss:.6f} | CE Loss = {sum_clssf_loss:.6f} | Val Loss = {val_loss:.6f} | \
                    Val MSE = {val_mse:.6f} | Val MAE = {val_mae:.6f} | Val ACC = {val_ce:.6f}")
            
            print(f"Epoch {epoch}: LR={last_lr:.8f} | Train Loss = {train_loss:.6f} | Val Loss = {val_loss:.6f} | Val MAE = {val_mae:.6f}")
            
            
# def _build_model(self, in_shape, model_name, aux_clssf=False):
#     return RegressionModel(in_shape=in_shape, model_name=model_name, aux_clssf=aux_clssf).to(device)
#     # return RegressionModel2(in_shape=in_shape).to(self.device)
    
def test_model(testloader, training_config):
    work_dir_path = os.path.join('work_dirs', training_config['ex_name'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    in_shape = training_config['in_shape']
        
    model = RegressionModel(in_shape=in_shape, model_name=training_config['backbone'],\
        aux_clssf=training_config['classification_head'], input_channels=training_config['in_shape'][0]).to(device)
    # model = _build_model(in_shape, 'resnet18', training_config['classification_head']).to(device)
    
    model= nn.DataParallel(model)
    model.load_state_dict(torch.load(os.path.join(work_dir_path, 'checkpoint.pth')))
    model.eval()
    
    # Disable gradient computation and reduce memory consumption.
    preds = []
    ids = []
    with torch.no_grad():
        for inputs, study_id in tqdm(testloader):
            if training_config['in_shape'][0] == 1:
                inputs = inputs.permute(1, 0, 2, 3)
            
            print(inputs.shape)
            y_pred_s, _ = model(inputs.to(device))
            
            if training_config['in_shape'][0] == 1:
                y_pred = y_pred_s.mean(dim=0)
            
            print(y_pred.shape)
            y_pred = y_pred.cpu().numpy()
            # Unnormalize the age
            # if training_config['classification_head']:
            #     y_pred = y_pred * (89 - 18) + 18
            # print(study_id.shape, y_pred.shape)
            
            # preds.append([study_id, y_pred])
            preds.append(y_pred)
            ids.append(study_id)

    
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