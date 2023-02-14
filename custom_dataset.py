import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import datasets, models, transforms
from PIL import Image
from create_dataset import FONT_INDEXES
import h5py
from create_dataset import cut_bb
from create_dataset import SPLITTER
FONT_INDEXES_B = {
    'Titillium Web':0,
    'Alex Brush':1,
    'Open Sans':2,
    'Sansation':3,
    'Ubuntu Mono':4
 }

class CustomDataset(Dataset):
    def __init__(self,path,trans = None,predict = False):
        self.imgs_path = path
        file_list = glob.glob(self.imgs_path + "/*")
        if predict:
            file_list = [path]
        self.data = []
        for class_path in file_list:
            class_name = class_path.split("/")[-1]
            for img_path in glob.glob(class_path + "/*.jpg"):
                if predict:
                    class_name = img_path.split(SPLITTER)[1] 
                self.data.append([img_path, class_name])
        self.class_map = FONT_INDEXES
        self.transforms = trans
        self.img_dim = (224,224)
        self.predict = predict
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        # img = cv2.imread(img_path)
        if self.transforms is not None:
            img = Image.open(img_path).convert('RGB')
            img = self.transforms(img)
            # img = img.permute(2,0,1)
        else:
            # img = cv2.resize(img, self.img_dim)
            img = torch.from_numpy(img).to(dtype=torch.float32)
            img = img.permute(2, 0, 1)
        if not self.predict:
            class_id = self.class_map[class_name]
            class_id = torch.tensor([class_id])
        else:
            class_id = class_name
        return img, class_id

class wordDataSet(DataLoader):
    def __init__(self):
        db  = h5py.File('Project/SynthText_train.h5','r')
        self.db = db
        im_names = list(db['data'].keys())[600:900]
        self.data = []
        for name in im_names:
            index = 0
            img_attrs = self.db['data'][name].attrs
            words = img_attrs['txt']
            for i, word in enumerate(words):
                self.data.append([name,word,index])
                index += len(word)
        self.class_map = FONT_INDEXES_B
        self.char_dim = (224,224)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        im_name ,word, font_idx = self.data[idx]
        img = self.db['data'][im_name][:]
        img_attrs = self.db['data'][im_name].attrs
        chars_BB = img_attrs['charBB'][:,:,font_idx:font_idx+len(word)]
        font = str(img_attrs['font'][font_idx].decode("utf-8"))
        
        chars = None
        for i in range(len(word)):
            try:
                char = cut_char(img,chars_BB[:,:,i])
                char = cv2.resize(char, self.char_dim)
                char = torch.from_numpy(char).to(dtype=torch.float32)
                char = char.permute(2, 0, 1)
                char = char.unsqueeze(0)
                if chars is None:
                    chars = char
                else:
                    chars = torch.cat((chars,char))
                class_id = self.class_map[font]
                class_id = torch.tensor([class_id])
            except Exception:
                pass
            # chars = chars.unsqueeze(0)
            # if r_words is None:
            #     r_words = chars
            # else:
            #     r_words = torch.cat((r_words,chars))
            # if ids is None:
            #     ids = class_id
            # else:
            #     ids = torch.cat((ids,class_id))
            # print(ids)
        return chars, class_id

if __name__ == "__main__":
    data_transforms = {
    'train':
        transforms.Compose([
        transforms.RandomApply([
        transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=180),
        transforms.GaussianBlur(kernel_size=(7, 13), sigma=(9, 11))
        ]),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ]),
    'validation':
        transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ]),
    }

    val_dataset = CustomDataset('datasets/datasets_02/val',data_transforms['train'])
    train_dataset = CustomDataset('datasets/datasets_02/train',data_transforms['train'])
    image_datasets = {
    'train': train_dataset,
    'validation': val_dataset
    }
    dataloaders = {
    'train':DataLoader(image_datasets['train'], batch_size=32, shuffle=True),
    'validation':DataLoader(image_datasets['validation'], batch_size=16, shuffle=True)
    }

    for ds in dataloaders.values():
        for count ,(imgs ,labels) in enumerate(ds):
            print("Batch of images has shape: ",imgs.shape)
            print("Batch of labels has shape: ", labels.shape)
            if count > 1:
                break
    words = wordDataSet()
    word_loader = DataLoader(words,batch_size=1, shuffle=False)

    for i, (words, ids) in enumerate(word_loader):
        print(words.shape)
        print(ids.shape)
        print(words.shape[1])
        word = words.squeeze(0)
        for char in word:
            print(char.shape)
        if i > 2:
            break