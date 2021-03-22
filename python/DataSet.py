from  PIL import  Image
import  torch
from  torch.utils.data import  Dataset
import numpy as np

class MyDataSet(Dataset):
    def __init__(self,images_path:list,images_class:list , transform):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform
    def __len__(self):
        return len(self.images_path)
    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        img = img.convert("RGB")
        if img.mode !='RGB':
            print("image:{} isn't RGB".format(self.images_path[item]))
        label = self.images_class[item]
        img=self.transform(img)
        return img,label
    def collate_fn(self,batch):
        images,labels = tuple(zip(*batch))
        images=torch.stack(images,dim=0)
        labels=torch.as_tensor(labels)
        return images,labels