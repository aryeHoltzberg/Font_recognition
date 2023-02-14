import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
import torch
from torchvision import datasets, models, transforms
from torchvision.models import resnet50, ResNet50_Weights 
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from custom_dataset import CustomDataset
import os
# torch.cuda.empty_cache()
import gc
import argparse

from new_pad import NewPad
import glob

last_model_path = ''

class Train():
    def __init__(self,dataset_path,predict_mode = False) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT).to(self.device)
        self.model.fc = nn.Sequential(
                nn.Linear(2048, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 5)).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters())
        self.best_model = None
        self.best_acc = 0
        self.num_of_trans = 4
        self.data_transforms = self.get_transforms() 
        self.val_path = os.path.join(dataset_path,'val')          
        self.train_path = os.path.join(dataset_path,'train')          
        self.test_path = os.path.join(dataset_path,'test')
        self.val_batch_size = 16          
        self.train_batch_size = 64
        if not predict_mode:          
            val_dataset = CustomDataset(self.val_path,self.data_transforms['validation'])
            train_dataset = CustomDataset(self.train_path,self.data_transforms['train'])
        else:
            val_dataset = None
            train_dataset = None

        if os.path.isdir(self.test_path):
            self.have_test = True
            if predict_mode:
                test_dataset = CustomDataset(self.test_path,self.data_transforms['validation'],predict=True)
            else:
                test_dataset = CustomDataset(self.test_path,self.data_transforms['validation'])
            self.test_batch_size = self.val_batch_size
        else:
            test_dataset = None

        self.image_datasets = {
            'train': train_dataset,
            'validation': val_dataset,
            'test': test_dataset
        }
        self.dataloaders = {}
        if not predict_mode:
            self.dataloaders['train'] = DataLoader(self.image_datasets['train'], batch_size=self.train_batch_size, shuffle=True)
            self.dataloaders['validation'] = DataLoader(self.image_datasets['validation'], batch_size=self.val_batch_size, shuffle=True)
        if self.have_test:
            self.dataloaders['test'] = DataLoader(self.image_datasets['test'], batch_size=self.test_batch_size, shuffle=False)
        self.loss_history = {
            'train': {
                'epoch_losses' : [],
                'epoch_correts' : []
            },
            'validation': {
                'epoch_losses' : [],
                'epoch_correts' : []
            },
        }
    def update_dataset(self,dataset_path):
        self.val_path = os.path.join(dataset_path,'val')          
        self.train_path = os.path.join(dataset_path,'train')          
        val_dataset = CustomDataset(self.val_path,self.data_transforms['validation'])
        train_dataset = CustomDataset(self.train_path,self.data_transforms['vali'])
        self.image_datasets = {
            'train': train_dataset,
            'validation': val_dataset
        }
        self.dataloaders = {
            'train':DataLoader(self.image_datasets['train'], batch_size=64, shuffle=True),
            'validation':DataLoader(self.image_datasets['validation'], batch_size=16, shuffle=True)
        }

    def save_best_model(self, i = 0,path = None):
        EPOCH = i
        if path is None:
            PATH = F"models/supre_model_train_4_{i}.pt"
        else:
            PATH = path
        LOSS = self.loss_history['train']['epoch_losses'][-1]
        
        torch.save({
            'epoch': EPOCH,
            'model_state_dict': self.best_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': LOSS,
            }, PATH)

    def save_model(self, i = 0,path = None):
        EPOCH = i
        if path is None:
            PATH = F"models/supre_model_train_5_{i}.pt"
        LOSS = self.loss_history['train']['epoch_losses'][-1]

        torch.save({
            'epoch': EPOCH,
            'model_state_dict': self.best_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': LOSS,
            }, PATH)

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model.train()

    def super_train(self,e = 10,batch_per_epoch = None):
        level = 0
        strike = 0
        for i in range(e):
            self.data_transforms = self.get_transforms(level)
            val_dataset = CustomDataset(self.val_path,self.data_transforms['validation'])
            train_dataset = CustomDataset(self.train_path,self.data_transforms['train'])
            self.image_datasets = {
                'train': train_dataset,
                'validation': val_dataset
            }
            self.dataloaders = {
                'train':DataLoader(self.image_datasets['train'], batch_size=32, shuffle=True),
                'validation':DataLoader(self.image_datasets['validation'], batch_size=16, shuffle=True)
            }
            self.train_model(1,batch_per_epoch)
            train_acc = self.loss_history['train']['epoch_correts'][-1]
            val_acc = self.loss_history['validation']['epoch_correts'][-1]
            if train_acc > 0.9 and train_acc - val_acc > 0.2:
                if level < self.num_of_trans:
                    level += 1
                strike = 0
            else:
                if strike > 3:
                    strike = 0 
                    if level > 0:
                        level -=1
                strike += 1
            print(f'epoch num {i} train_acc = {train_acc},val_acc = {val_acc} level = {level}')
            # self.save_model(i)


        return
    def train_model(self, num_epochs=1,batch_per_epoch = None):
        for epoch in range(num_epochs):
            # print('Epoch {}/{}'.format(epoch+1, num_epochs))
            # print('-' * 10)
            for phase in ['train', 'validation']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                

                running_loss = 0.0
                running_corrects = 0

                for i, (inputs, labels) in enumerate(tqdm(self.dataloaders[phase])):
                    if batch_per_epoch is not None and batch_per_epoch < i+1 and phase == 'train':
                        break
                    inputs = inputs.to(self.device).float()
                    labels = labels.squeeze(1).to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                    if phase == 'train':
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        # scheduler.step()

                    _, preds = torch.max(outputs, 1)
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                epoch_loss = running_loss / ((i+1)*self.dataloaders[phase].batch_size)
                epoch_acc = running_corrects.double() / ((i+1)*self.dataloaders[phase].batch_size)
                # epoch_loss = running_loss / len(self.image_datasets[phase])
                # epoch_acc = running_corrects.double() / len(self.image_datasets[phase])
                if epoch_acc > self.best_acc and phase == 'validation':
                    self.best_model = self.model
                    self.best_acc = epoch_acc
                self.loss_history[phase]['epoch_losses'].append(epoch_loss)
                self.loss_history[phase]['epoch_correts'].append(epoch_acc.cpu())
                # print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,
                #                                             epoch_loss,epoch_acc))

    def predict(self,models_repo):
        return self.eval_model(models_repo,predict_mode=True)
        
    def eval_model(self,models_repo,predict_mode = False):
        all_perds = []
        all_labels = []
        models = glob.glob(models_repo + "/*")
        for j, model in enumerate(models):
            if model.endswith('.pt'):
                self.load_model(model)
                self.model.eval()
                running_corrects = 0
                predicts = []
                for i, (inputs, labels) in enumerate(tqdm(self.dataloaders['test'],desc=f'model_number {j+1}/{len(models)}')):
                    inputs = inputs.to(self.device).float()
                    if not predict_mode:
                        labels = labels.squeeze(1).to(self.device)
                    if j == 0:
                        if predict_mode:
                            all_labels.extend(labels)
                        else:
                            all_labels.append(labels)
                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)
                    predicts.append(preds)
                    if not predict_mode:
                        running_corrects += torch.sum(preds == labels.data)
                predicts  = torch.cat(predicts)
                predicts = predicts.unsqueeze(0)
                all_perds.append(predicts)
                # acc = running_corrects.double() / ((i+1)*self.dataloaders['test'].batch_size)
                if not predict_mode:
                    acc = running_corrects.double() / len(self.image_datasets['test'])
                    print(acc,model)
                    self.loss_history[model] = {
                        'acc': acc.cpu(),
                        'preds': predicts
                        }
        all_perds = torch.cat(all_perds,dim=0)
        votes ,_ = torch.mode(all_perds,dim=0)
        if not predict_mode:
            all_labels = torch.cat(all_labels)
            print(torch.sum(votes == all_labels.data)/len(all_labels))
        if predict_mode:

            return votes , all_labels

    def get_transforms(self,level = 1):
        mean = np.load('final_mean.npy')
        std = np.load('final_std.npy')
        normalize = transforms.Normalize(mean=mean,
                                    std=std)
        self.trans = [
                transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.Pad(4),
                # transforms.RandomCrop(32, 4),
                # transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
                transforms.RandomErasing(p=0.5),
                transforms.RandomRotation(degrees=10),
                # transforms.RandomRotation(degrees=60),
                # transforms.RandomRotation(degrees=90),
                # transforms.RandomRotation(degrees=120),
                # transforms.RandomRotation(degrees=150),
                # transforms.RandomRotation(degrees=180),
                transforms.GaussianBlur(kernel_size=(3,3), sigma=(9, 11))
                ]
        # self.num_of_trans = len(self.trans)
        if level > self.num_of_trans:
            level = self.num_of_trans
        if level < 0 :
            level = 0
        
        t = np.random.choice(self.trans,self.num_of_trans,replace=False)
        self.t = t
        data_transforms = {
            'train':
                transforms.Compose([
                NewPad(),
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.RandomApply(t),
                transforms.Resize((224,224)),
                transforms.Normalize(mean=mean,
                                    std=std),
            ]),
            'validation':
            transforms.Compose([
                NewPad(),
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean,
                                    std=std),
            ]),
        }
        return data_transforms

def main(args = None,num =0):
    T = Train(args.dataset_path)
    T.load_model('results/final/models/model_6_acc_0.775462962962963.pt')
    T.num_of_trans = 5

    T.update_dataset('datasets/datasets_03')
    T.num_of_trans = 3
    T.super_train(10)

    path = os.path.join('results/final/models',f'model_{num}_acc_{T.best_acc}.pt')
    T.save_best_model(num,path)

    plt.plot(T.loss_history['train']['epoch_losses'], label = "train loss", linestyle="-.")
    plt.plot(T.loss_history['validation']['epoch_losses'], label = "validation loss", linestyle=":")
    plt.legend()
    plt.savefig(os.path.join(f'results/final',f'model_{num}_loss.jpeg'))
    plt.clf()

    plt.plot(T.loss_history['train']['epoch_correts'], label = "train acc", linestyle="-.")
    plt.plot(T.loss_history['validation']['epoch_correts'], label = "validation acc", linestyle=":")
    plt.legend()
    plt.savefig(os.path.join('results/final',f'model_{num}_acc.jpeg'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Genereate Synthetic Scene-Text Images')
    parser.add_argument('--dataset-path',action='store',dest='dataset_path',default = 'datasets/simple/dataset-0',type=str)
    for i in range(1):
        main(parser.parse_args(),6)

    