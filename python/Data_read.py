import torch
import numpy as np
import random
from torch import  nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import os
import cv2
import json
import matplotlib.pyplot as plt
learning_rate=0.01
epochs=10

def read_data(root:str,plot_image,vale_rate:float=0.2,):
    random.seed(0)
    assert os.path.exists(root),"dataset root:{}does not exist.".format(root)
    #1.读取文件目录
    audio_class = [file for file in os.listdir(root)]
    #防止文件不是文件夹
    #auduio_class = [file for file in os.listdir(root) if os.path.isdir(os.path.join(root,file))]
    audio_class.sort()
    #2.分类索引存入字典
    class_indices = dict((k,v)for v,k in enumerate(audio_class))
    print(class_indices)
    #3.json文件保存备用
    json_str = json.dumps(dict((val,key)for key,val in class_indices.items()),indent=4)
    with open('class_indices.json','w')as json_file:
        json_file.write(json_str)

    train_images_path = []
    train_images_label = []
    val_images_path = []
    val_images_label = []
    every_class_num = []
    supported = [".jpg",".JPG",".png",".PNG"]
    #5.遍历每个文件夹的每个文件
    for file in audio_class:
        file_path = os.path.join(root,file)
        images = [os.path.join(root,file,i)for i in os.listdir(file_path)if os.path.splitext(i)[-1] in supported]
        images_class = class_indices[file]
        every_class_num.append(len(images))
        val_path = random.sample(images,k=int(len(images)*vale_rate))
        for img_path in images:
            if img_path in val_path:
                val_images_path.append(img_path)
                val_images_label.append(images_class)
            else:
                train_images_path.append(img_path)
                train_images_label.append(images_class)
    print("{} images were found in the datasets.".format(sum(every_class_num)))

    a=np.arange(images_class+1)
    if plot_image:
        plt.bar(a,every_class_num)
        plt.xticks(a)
        for i,v in enumerate(every_class_num):
            plt.text(x=i,y=v+5,s=str(v),ha='center')
        plt.xlabel('image class')
        plt.ylabel('number of images')
        plt.title('image class distribution')
        plt.show()

    return train_images_path,train_images_label,val_images_path,val_images_label

def read_data_to_gray_image(root,plot_image):
    train_images_path,train_images_label,val_images_path,val_images_label=read_data(root,plot_image)
    train_images= []
    val_images =[]
    for image_path in train_images_path:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        train_images.append(image)
    for image_path in val_images_path:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        val_images.append(image)
    train_images= torch.from_numpy(np.array(train_images))
    train_images_label = torch.from_numpy(np.array(train_images_label))
    val_images = torch.from_numpy(np.array(val_images))
    val_images_label = torch.from_numpy(np.array(val_images_label))
    print(train_images.shape)
    print(val_images.shape)
    print(train_images_label.shape)
    print(val_images_label.shape)
    return train_images,train_images_label,val_images,val_images_label

def main():
    root ="data"
    read_data_to_gray_image(root,False)
if __name__ == '__main__':
    main()