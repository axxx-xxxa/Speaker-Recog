import torch
import numpy
import torch.nn
from torchvision import transforms
from torch import nn
from torch.nn import functional as F

class mynet(nn.Module):
    def __init__(self):
        super(mynet, self).__init__()
        self.CNN = nn.Sequential(
            # input [ n , 3 , 450 , 450 ]
            nn.Conv2d(3,10,kernel_size=3,stride=1,padding=0),
            nn.AvgPool2d(kernel_size=3,stride=2,padding=0),
            nn.Conv2d(10,20,kernel_size=3,stride=1),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(20, 30, kernel_size=3, stride=1),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(30,40,kernel_size=3,stride=1),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(40, 50, kernel_size=3, stride=1),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(50, 60, kernel_size=3, stride=1),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
        )
        self.linear = nn.Sequential(
            nn.Linear(60*5*5,40*12),
            nn.ReLU(),
            nn.Linear(40*12, 200),
            nn.ReLU(),
            nn.Linear(200, 50),
            nn.ReLU(),
            nn.Linear(50, 3),
            #softmax 在 CrossEntropyLoss 里面 👇
        )
        #self.criteon = nn.CrossEntropyLoss()

    def forward(self,x):
        batchsz=x.size(0)
        # x [ n , 3, 450 , 450]
        x = self.CNN(x)
        # x [ n , 90 , 14 , 14 ] -> [ n , 90*14*14 ]
        x=x.view(batchsz,60*5*5)
        # x[ n , 90*14*14 ] -> [ n , 7 ]
        logits = self.linear(x)
        # x[ n , 3 ] 这一步不需要写了 softmax在CrossEntropyLoss()里面
        # pred = F.softmax(logits,dim=1)
        #loss = self.criteon(logits,y)
        return    logits
        # tmp=torch.randn(17,1,480,640)
        # out = self.CNN(tmp)
        # print(out.shape)
        # out=out.view(17,70*5*15)
        # end=self.linear(out)
        # print(end.shape)
        # print(end)

def main():
    net = mynet()
    tmp = torch.randn(4, 3, 450, 450)
    end = net.CNN(tmp)
    print(end.shape)
    end=end.view(4,40*12*12)
    endend = net.linear(end)
    print(endend.shape)
if __name__ == '__main__':
    main()
