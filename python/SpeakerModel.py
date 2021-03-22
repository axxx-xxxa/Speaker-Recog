from Data_read import read_data,read_data_to_gray_image
from ModelTest import mynet
import cv2
import numpy as np
import torch
from torch import nn,optim
from torch.utils.data import DataLoader
from torchvision import transforms
from DataSet import MyDataSet
from sklearn.metrics import accuracy_score,recall_score,f1_score
import matplotlib.pyplot as plt
root ="data"
batchsz=8

def main():
    ############################读数据还要处理 CNM
    # train_images, train_images_label, val_images, val_images_label=read_data_to_gray_image(root,False)
    train_images_path, train_images_label, val_images_path, val_images_label=read_data(root,False)
    data_transform={
        "train":transforms.Compose([transforms.RandomResizedCrop(450,scale=(0.5,1.0)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
        "val": transforms.Compose([transforms.RandomResizedCrop(450,scale=(0.5,1.0)),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) }

    train_data_set = MyDataSet(images_path=train_images_path,
                                images_class=train_images_label,
                                transform=data_transform["train"])
    val_data_set = MyDataSet(images_path=val_images_path,
                               images_class=val_images_label,
                               transform=data_transform["val"])

    train_loader =torch.utils.data.DataLoader(train_data_set,
                                              batch_size=batchsz,
                                              shuffle=True,
                                              collate_fn=train_data_set.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_data_set,
                                               batch_size=batchsz,
                                               shuffle=True,
                                               collate_fn=val_data_set.collate_fn)


    model = mynet()
    criteon = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=0.0005)
    x,label = iter(train_loader).next()
    print(x.shape)
    val,val_label=iter(val_loader).next()
    print(val.shape)
    train_loss = []
    epochs =[]
    accs = []
    for epoch in range(2):
        for batchidx,(x,label) in enumerate(train_loader):
            logits = model(x)
            # logits:[b , 10]
            # label :[b ]
            loss = criteon(logits,label)
             #backpropc
            train_loss.append(loss.item())
            pred = logits.argmax(dim=1)
            acc=accuracy_score(pred,label)
            accs.append(acc)
            epochs.append(epoch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("Epoch:{} -> loss={}".format(epoch,loss.item()))

    train_loss=np.array(train_loss)
    epochs=np.array(epochs)
    # loss
    f1 = np.polyfit(epochs, train_loss, 3)
    p1 = np.poly1d(f1)
    yvals1 = p1(epochs)
    plt.subplot(221)
    plt.title('Loss')
    plt.plot(epochs,train_loss)
    plt.plot(epochs,yvals1)
    #  acc
    f2 = np.polyfit(epochs, accs, 3)
    p2 = np.poly1d(f2)
    yvals2 = p2(epochs)
    plt.subplot(222)
    plt.title('ACC')
    plt.plot(epochs,accs)
    plt.plot(epochs, yvals2)
    plt.show()
    for batchidx,(val,val_label) in enumerate(val_loader):
        logits=model(val)
        loss = criteon(logits, val_label)
        print("val loss = ",loss.item())
        pred = logits.argmax(dim=1)
        print("label:",val_label)
        print("labelhat:",pred)
        print("-ACC=",accuracy_score(val_label,pred))
if __name__ == '__main__':
    main()