import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset, DataLoader, TensorDataset


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class AttentionHead(nn.Module):
     def __init__(self, channel_size, ratio=16, kernel_size=7):
          super().__init__()
          self.Channel = ChannelAttention(channel_size, ratio)
          self.Spatial = SpatialAttention(kernel_size)

     def forward(self, x):
          x = self.Channel(x)
          x = self.Spatial(x)
          return x


class ChannelAttention(nn.Module):
     def __init__(self, channel_size, ratio=16):
          super().__init__()
          self.out = max(1, channel_size // ratio)
          self.avg = nn.AdaptiveAvgPool2d((1, 1))
          self.flat = nn.Flatten()

          self.lza = nn.LazyLinear(self.out)
          self.rLU = nn.ReLU()
          self.drop = nn.Dropout(0.1)

          self.lzb = nn.LazyLinear(channel_size)
          self.sigm = nn.Sigmoid()
     
     def forward(self, xo):
          x = self.avg(xo)
          x = self.flat(x)

          x = self.lza(x)
          x = self.rLU(x)
          x = self.drop(x)

          x = self.lzb(x)
          x = self.sigm(x)
          x = x.unsqueeze(-1).unsqueeze(-1)

          return xo * x # element-wise


class SpatialAttention(nn.Module):
     def __init__(self, kernel_size=7):
          super().__init__()
          padding = kernel_size // 2
          self.attn = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding)
          self.sig = nn.Sigmoid()

     def forward(self, xo):
          avgx = torch.mean(xo, dim=1, keepdim=True)
          maxx, _ = torch.max(xo, dim=1, keepdim=True)

          x = torch.cat([avgx, maxx], dim=1)

          x = self.attn(x)
          x = self.sig(x)

          return xo * x


class Conv(nn.Module):
     def __init__(self, output_channels):
          super().__init__()
          self.conv = nn.LazyConv2d(output_channels, kernel_size=3, padding='same')
          self.rLU = nn.ReLU()
          self.drop = nn.Dropout(0.1)
          self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

     def forward(self, x):
          x = self.conv(x)
          x = self.rLU(x)
          x = self.drop(x)
          return self.pool(x)


class Model(nn.Module):
     def __init__(self):
          super().__init__()
          self.conv0 = Conv(32)
          self.cbam0 = AttentionHead(channel_size=32) # mb should be in Conv

          self.conv1 = Conv(64)
          self.cbam1 = AttentionHead(channel_size=64)

          self.conv2 = Conv(128)
          self.cbam2 = AttentionHead(channel_size=128)

          self.conv3 = Conv(256)
          self.cbam3 = AttentionHead(channel_size=256)

          self.conv4 = Conv(512)
          self.cbam4 = AttentionHead(channel_size=512)

          self.flat = nn.Flatten()

          self.dense = nn.LazyLinear(out_features=42)
          self.out = nn.Unflatten(1, torch.Size([21, 2]))

     def forward(self, inputs):
          x = self.conv0(inputs)
          x = self.cbam0(x)

          x = self.conv1(x)
          x = self.cbam1(x)

          x = self.conv2(x)
          x = self.cbam2(x)

          x = self.conv3(x)
          x = self.cbam3(x)

          x = self.conv4(x)
          x = self.cbam4(x)

          x = self.flat(x)
          x = self.dense(x)

          return self.out(x)


class Losses(nn.Module):
     def __init__(self):
          super().__init__()
          self.scale = 224.0

     def mae(self, pred, true):
          return torch.mean(torch.abs(pred - true))

     def pixels(self, pred, true):
          return self.mae(pred, true)*self.scale


class Data(Dataset):
     def __init__(self, inputs, targets):
          self.inputs = inputs
          self.targets = targets

     def __len__(self):
          return len(self.inputs)
     
     def __getitem__(self, idx): # indexing
          img = read_image(self.inputs[idx], mode=ImageReadMode.RGB)

          img = transforms.Resize((224, 224))(img)
          img = img.float() / 255.0 # norm [0,1]

          return img, self.targets[idx]

     def len_batch(self):
          return int(len(self.inputs) // 32)

     def load(self):
          return DataLoader(self, batch_size=32, shuffle=True)

     def to_dev(self):
          for X, y in self.load():
               yield X.to(device, dtype=torch.float32), y.to(device, dtype=torch.float32)


with np.load("/Volumes/HomeXx/compuir/hands_ml/data/preprocessed0.npz") as data:
     training = Data(data['xtrain'], data['ytrain'])
     # xtrain, ytrain = data['xtrain'], data['ytrain']
     validation = Data(data['xtest'], data['ytest'])
     # xtest, ytest = data['xtest'], data['ytest']
     testing = Data(data['xval'], data['yval'])
     # xval, yval = data['xval'], data['yval']


model = Model().to(device, dtype=torch.float32)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss = Losses()


# training = Data(xtrain, ytrain)
# validation = Data(xval, yval)


epochs = 20
for epoch in range(epochs):
     print(f"\nEpoch {epoch}/{epochs}:")
     val_mae = []
     maex = []
     prg = 0

     model.train() # activates training mode
     for inputs, targets in training.to_dev():
          # forward pass
          predictions = model(inputs)
          mae = loss.mae(predictions, targets)

          maex += [mae]

          # back propogate 
          optimizer.zero_grad() # clear gradients
          mae.backward() # compute gradients
          optimizer.step() # update params

          total = training.len_batch()
          prg += 1

          print(f"[{prg}/{total}] MAE: {(mae*224.0):.4f}", end="\r")

     print(f"Training Loss [{epoch}/{epochs}]: {((sum(maex)/len(maex))*224.0):.4f}")

     model.eval() # update model mode (from training)
     with torch.no_grad(): # no gradient computations
          for val_inputs, val_targets in validation.to_dev():
               # forward pass
               val_predictions = model(val_inputs)
               # evaluate
               mae = loss.mae(val_predictions, val_targets)
               val_mae += [mae]

               print(f"[val] MAE: {(mae*224.0):.4f}", end="\r")
               # print(f"[val] MAE: {mae:.4f}", end="\r")

          print(f"Validation Loss [{epoch}/{epochs}]: {((sum(val_mae)/len(val_mae))*224.0):.4f}")


# testing = Data(xtest, ytest)
test_mae = []
for test_inputs, test_targets in testing.to_dev():
     # forward pass
     test_predictions = model(test_inputs)
     # evaluate
     mae = loss.mae(test_predictions, test_targets)
     test_mae += [mae]

     print(f"[tst] MAE: {(mae*224.0):.4f}", end="\r")
     # print(f"[test] MAE: {mae:.4f}", end="\r")

print(f"Test Loss: {((sum(test_mae)/len(test_mae))*224.0):.4f}")
