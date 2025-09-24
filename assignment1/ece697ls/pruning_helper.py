import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import numpy as np
from tensorflow import keras

BATCH_SIZE = 64
EPOCHS = 10

class ConvNet(torch.nn.Module):
    def __init__(self,num_classes):
        super(ConvNet, self).__init__()

        self.num_classes = num_classes
        self.conv1 = torch.nn.Conv2d(3,32,kernel_size=5)
        self.pool = torch.nn.MaxPool2d(2, 2)

        self.conv2 = torch.nn.Conv2d(32,32,kernel_size=3)
        self.pool2 = torch.nn.MaxPool2d(2, 2)

        self.dropout = torch.nn.Dropout(0.5)
        self.dense1 = torch.nn.Linear(1152, 128)
        self.dense_out = torch.nn.Linear(128, num_classes)
    
    def forward(self,input):
        
        x = self.pool(F.relu(self.conv1(input)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense_out(x)

        return x

def test_model(model,test_data,mode='torch'):

    if(mode=='torch'):
      test_x,test_y = torch.tensor(test_data[0]), test_data[1] 

      with torch.no_grad():
          ypred = model(test_x.float())
          ypred = torch.argmax(ypred,dim=1).detach().numpy()
          
    elif(mode=='tensorflow'):
      test_x,test_y = test_data
      if(test_x.shape[1]==3):
        test_x = invert_ch(test_x)

      ypred = model.predict(test_x)
      ypred = np.argmax(ypred,axis=1)
    
    else:
      raise NotImplementedError
    
    print(f'Test Acc: {accuracy_score(test_y,ypred):.4f}')



def invert_ch(data):
  data = data.transpose(0, 3, 2, 1).copy()
  
  return data


def load_model(path,num_classes=6, mode='torch'):

  model = None
  if(mode=='torch'):
    model = ConvNet(6)
    model.load_state_dict(torch.load(path))
  elif(mode=='tensorflow'):
    model = keras.models.load_model(path)
  else:
    raise NotImplementedError
  

  return model
