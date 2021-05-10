# Loading data
import torch 
import torchvision 
from torchvision import transforms 
import torch.nn as nn 
import torch.optim as optim 


path = '/mnt/srv/home/dlpc.485/FashionData/trainset6500_2'
#이미지를 tensor형태로 변환해주는 변수 선언.
transform = transforms.Compose([
                                transforms.ToTensor()
                                ])

#다운받은 이미지를 tensor 변환
#ImageFolder는 클래스별로 폴더화된 계층구조의 폴더를 클래스 별로 labeling해줌
train_set = torchvision.datasets.ImageFolder(path, transform=transform)
print(train_set.classes)
#train_set을 batchsize, suffle, num_workers 설정하여 메모리 load.
train_loader = torch.utils.data.DataLoader(train_set,
                                          batch_size=18,
                                          shuffle=True,
                                          num_workers=1)

def conv(in_channels, out_channels):
  return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)

#convolution, batchNormalization, activation, maxpooling을 하나의 block으로 묶는 클래스.
class Block(nn.Sequential):
  def __init__(self, in_channels, out_channels, stack, act=nn.ReLU(True)):
    m = [conv(in_channels, out_channels), nn.BatchNorm2d(out_channels), act]
    for _ in range(stack-1):
      m.append(conv(out_channels, out_channels))
      m.append(nn.BatchNorm2d(out_channels))
      m.append(act)
    m.append(nn.MaxPool2d(2,2))
    
    super(Block, self).__init__(*m)


#Block 클래스를 이용해 신경망을 여러층으로 쌓는 클래스 정의.
class Vgg11(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = Block(3, 16, 1)
    self.conv2 = Block(16, 32, 1)    
    self.conv3 = Block(32, 64, 2)
    self.conv4 = Block(64, 128 , 2)
    self.conv5 = Block(128, 256, 2)
    
    #fully connected layer 정의.
    #320->160->80->40->20->10 
    #256 channel x 10 x 10 (w,h) = 25600
    self.fc = nn.Sequential(nn.Linear(25600, 10000),
                             nn.ReLU(True),
                             nn.Dropout(),
                             nn.Linear(10000, 1024),
                             nn.ReLU(True),
                             nn.Dropout(),
                             nn.Linear(1024, 13))
    #sofmax를 이용해 출력값을 13개 클래스의 확률로 한다.
    self.classifier = nn.Softmax(dim=1)

  #각 layer 연산을 전방전달하는 함수.
  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.conv4(x)
    x = self.conv5(x)
    x = torch.flatten(x,1)
    x = self.fc(x)
    x = self.classifier(x)
    return x
    

model = Vgg11().cuda()
print(model)

# Configure loss function and optimizer

criterion = nn.CrossEntropyLoss() 
optimizer = optim.SGD(model.parameters(), lr=0.01) 



from torch.autograd import Variable
import numpy as np
epochs = 100 
losses = list()
accuracies = list() 
for epoch in range(epochs):
  epoch_loss = 0 # 해당 epoch_loss 저장할 변수
  epoch_accuracy = 0 #해당 epoch_accuracy를 저장할 변수
  batch = 0 #해당 epoch의 batch
  class_pred = np.zeros((13,13))
    
  for idx, (x, y) in enumerate(train_loader):
    x = Variable(x.cuda()) #train_dataset의 input(이미지 정보)
    y = Variable(y.cuda()) #train_dataset의 label

    optimizer.zero_grad() #역전파 단계를 실행하기 전에 변화도를 0으로 초기화.
    output = model(x) #모델의 학습 결과를 저장.
    
    loss = criterion(output,y) #학습 결과와 학습 데이터의 label의 loss 계산.
    loss.backward() #역전파 학습 실행.

    optimizer.step() #가중치 갱신.
    
    _, y_pred = torch.max(output, 1) #결과값 중에 각 행에 가장 큰 값의 인덱스를 반환.
    accuracy = sum(y == y_pred) #라벨값인 y와 비교하며 accuracy를 측정.

    #loss와 accuracy를 누적해서 더함.
    epoch_accuracy += accuracy.item() 
    epoch_loss += loss.item() 

    batch += len(x)  
    for i in range(len(x)):
        class_pred[y[i]][y_pred[i]]+=1
    
  
    #각 epoch 마다 평균 Accuracy, Loss를 출력
  print('Epoch: {} [Accuracy: {:.1f}%,  \t Loss: {:.6f}]'.format(
  epoch+1,(epoch_accuracy/len(train_loader.dataset))*100., epoch_loss/len(train_loader)))
  #각 epoch에서의 평균 loss와 accuracy를 리스트에 저장.
  losses.append(epoch_loss/len(train_loader)) #20163107 박건희 |
  accuracies.append((epoch_accuracy/len(train_loader.dataset))) #20163107 박건희 |
  
  #for idx, val in enumerate(class_pred):
  #  #Aprint(np.sum(class_pred_copy[idx]))
  #  class_pred[idx] /= np.sum(class_pred[idx]) 
  #  class_pred[idx]*=100
    #print(class_pred_copy[idx])
  #  class_pred=np.round_(class_pred.copy(),2)
  #print(class_pred)
  
# Save model's parameter
#print("time: ", (time.time()-start)//60,"'",(time-time()-start)%60)
torch.save(model.state_dict(),'/mnt/srv/home/FashionData/trainset6500_2VGG11-lr0.01.pth') #20163107 박건희 | train결과로 나온 모델을 저장.


import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(20,5))
plt.subplots_adjust(wspace=0.2)

plt.subplot(1,2,1)
plt.title("$Loss$",fontsize = 18)
plt.plot(losses)
plt.grid()
plt.xlabel("$epochs$", fontsize = 16)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)


plt.subplot(1,2,2)
plt.title("$Accuracy$", fontsize = 18)
plt.plot(accuracies)
plt.grid()
plt.xlabel("$epochs$", fontsize = 16)
plt.xticks(fontsize = 14)
plt.yticks(np.arange(0,1.1,0.1), fontsize = 14)

plt.show()



## Test
 #20163107 박건희 |
import torch
import torchvision
from torch import nn
from torchvision import transforms
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
#from torch import model
batch = 0 
test_accuracy = 0 
model_path = '/mnt/srv/home/FashionData/trainset6500_2VGG11-lr0.01.pth'
test_path = '/mnt/srv/home/dlpc.485/FashionData/testset2'
#model = TheModelClass(*args, **kwargs)
#model = torch.load(model_path)
#model = model().cuda()

model = Vgg11().cuda()
model.load_state_dict(torch.load(model_path))

#model.eval() test단계에서는 eval 메소드를 호출해 dropout이 비활성화.



#이미지를 tensor형태로 변환해주는 변수 선언.
transform = transforms.Compose([
                                transforms.ToTensor()
                                ])
#다운받은  이미지를 tensor 변환하여 train_set으로 설정. 
test_set = torchvision.datasets.ImageFolder(test_path, transform=transform)
#print(test_set.classes)
#20163107 박건희 | train_set을 batchsize, suffle, num_workers 설정하여 메모리 load.
test_loader = torch.utils.data.DataLoader(test_set,
                                          batch_size=25,
                                          shuffle=True,
                                          num_workers=1)
test_pred = np.zeros((13,13))


for idx, (x, y) in enumerate(test_loader):
  print(y)
  x = x.cuda() 
  y = y.cuda()
  output = model(x)
  _, y_pred = torch.max(output, 1) #결과값 중에 각 행에 가장 큰 값의 인덱스를 반환.
  accuracy = sum(y == y_pred)      #이를 라벨값인 y와 비교하며 accuracy를 측정.
  test_accuracy += accuracy.item() #test_accuracy의 평균을 구하기 위해 accuracy를 누적하여 더함.
  batch += len(x)

  
  for i in range(len(x)):
    test_pred[y[i]][y_pred[i]]+=1
  #for idx, val in enumerate(test_pred):
  #  #Aprint(np.sum(class_pred_copy[idx]))
  #  test_pred[idx] /= np.sum(test_pred[idx]) 
  #  test_pred[idx]*=100
  #  #print(class_pred_copy[idx])
  #  test_pred=np.round_(test_pred.copy(),2)
  #print(test_pred,end='\n')
print(np.round(test_pred/936/13,2))
print('Epoch: {} [Accuracy: {:.1f}%]'.format(
    epoch+1,(test_accuracy/len(test_loader.dataset))*100.)
'''
for idx, (x, y) in enumerate(test_loader):
  #20163107 박건희 | 데이터와 라벨값이 gpu 연산이 가능하게 함.
  x = x.cuda() 
  y = y.cuda()

  output = model(x) #]모델에 x를 입력으로 넣어 foward 연산을 통해 결과값 도출.
  loss = criterion(output,y) #손실 함수로 Cross Entropy를 사용하여 loss를 계산.
  
  _, y_pred = torch.max(output, 1) #결과값 중에 각 행에 가장 큰 값의 인덱스를 반환.

  accuracy = sum(y == y_pred)      #이를 라벨값인 y와 비교하며 accuracy를 측정.
  test_accuracy += accuracy.item() #test_accuracy의 평균을 구하기 위해 accuracy를 누적하여 더함.
  batch += len(x)                  #진행률과 test_accuracy의 평균을 구하기 위해서 batch를 누적하여 더함.
  for i in range(len(x)):
    test_pred[y[i]][y_pred[i]]+=1
  
  #배치 사이즈마다 진행도와 Accuracy, Loss.
  print('Test:  [{}/{} ({:.0f}%)],\tAccuracy: {:.1f}%,  \t Loss: {:.6f}'.format(
      batch, len(test_loader.dataset), 100.*(idx+1)/len(test_loader), 100.*(test_accuracy/batch), loss.item()))
  for idx, val in enumerate(test_pred):
    #Aprint(np.sum(class_pred_copy[idx]))
    test_pred[idx] /= np.sum(test_pred[idx]) 
    test_pred[idx]*=100
    #print(class_pred_copy[idx])
    test_pred=np.round_(test_pred.copy(),2)
  print(test_pred,end='\n')
'''



