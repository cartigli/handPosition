import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset, DataLoader, TensorDataset


class ResidualConvolutionBlock(nn.Module):
     def __init__(self, output_channels):
          super().__init__()
          self.conv = nn.LazyConv2d(output_channels, kernel_size=3, padding='same')
          self.attn = AttentionHead(channel_size=output_channels)
          self.drop = nn.Dropout(0.1)
          self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

          self.shortcut = nn.LazyConv2d(output_channels, kernel_size=1) # project to match dims

     def forward(self, x):
          res = self.shortcut(x)

          x = self.conv(x)
          x = F.relu(x)
          x = self.attn(x)
          x = self.drop(x)

          x = F.relu(x + res)
          return self.pool(x)


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
          x = F.relu(x)
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


class Model(nn.Module):
     def __init__(self):
          super().__init__()
          self.rcb_1 = ResidualConvolutionBlock(32)
          self.rcb_2 = ResidualConvolutionBlock(64)
          self.rcb_3 = ResidualConvolutionBlock(128)
          self.rcb_4 = ResidualConvolutionBlock(256)
          self.rcb_5 = ResidualConvolutionBlock(512)

          self.flat = nn.Flatten()

          self.dense = nn.LazyLinear(out_features=42)
          self.out = nn.Unflatten(1, torch.Size([21, 2]))

          self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
          self.loss = Losses()

     def forward(self, inputs):
          x = self.rcb_1(inputs)
          x = self.rcb_2(x)
          x = self.rcb_3(x)
          x = self.rcb_4(x)
          x = self.rcb_5(x)

          x = self.flat(x)

          x = self.dense(x)
          return self.out(x)

     # def predict(self, inputs):
     #      for X, y in inputs:
     #           prediction = self(X)
     #           mae = self.loss.mae(prediction, y)
     #           yield mae
     
     def config(self):
          self.to(self.device)
          optimizer = torch.optim.Adam(self.parameters(), lr=0.0002)
          return self, self.loss, optimizer, self.device


     #      # return self.model.to(device), self.optimizer, self.loss
     #      model = self.to(device)
     #      optimizer = torch.optim.Adam(model.parameters(), lr=0.0004)
     #      return model, optimizer, self.loss


# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# model = Model().to(device, dtype=torch.float32)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0004)
# loss = Losses()


class Losses(nn.Module):
     def __init__(self):
          super().__init__()
          self.scale = 224.0

     def mae(self, pred, true):
          return torch.mean(torch.abs(pred - true))

     def pixel(self, pred, true):
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
          return int(len(self.inputs) // 16)

     def load(self):
          return DataLoader(self, batch_size=16, shuffle=True)

     def to_dev(self, device):
          for X, y in self.load():
               yield X.to(device, dtype=torch.float32), y.to(device, dtype=torch.float32)


with np.load("/Volumes/HomeXx/compuir/hands_ml/data/preprocessed0.npz") as data:
     training_Data = Data(data['xtrain'], data['ytrain'])
     validation_Data = Data(data['xtest'], data['ytest'])
     testing_Data = Data(data['xval'], data['yval'])


# model, optimizer, loss = Model().config()


# class MLX():
#      # def __init__(self, model, optimizer, loss, data, epochs, name=None):
#      def __init__(self, model, training, validating):
#           super().__init__()
#           self.model, self.optimizer, self.loss = model
#           # self.model = model
#           self.train = training
#           self.val = validating
#      # def mlx(self):
#      #      for epoch in self.epochs:
#      #           self.heavy_lifting()
#      #      # for epoch in self.epochs:
#      def mlx(self):
#           total = self.training.len_batch()
#           maex = []
#           prg = 0
#           for inputs, targets in self.training.to_dev():
#                pred = self.model(inputs)
#                mae = self.loss.mae(pred, targets)
#                maex += [mae]
#                prg += 1
#                print(f"[{prg}/{total}] {self.name}: {mae:.4f}", end="\r")
#                yield mae
#           print(f"{self.name}: {((sum(maex)/len(maex))*224.0):.4f}")
#      def train(self):
#           for mae in self.predict():
#                self.optimizer.zero_grad()
#                mae.backward()
#                self.optimizer.step()
#      def verify(self):
#           for mae in self.predict():
#                pass

model, loss, optimizer, device = Model().config()


def predict(batches):
     for inputs, targets in batches.to_dev(device):
          predictions = model(inputs)

          mae = loss.mae(predictions, targets)
          pixels = loss.pixel(predictions, targets)

          yield mae, pixels


epochs = 20
for epoch in range(epochs):
     print(f"\nEpoch {epoch+1}/{epochs}:")

     # for inputs, targets in training.to_dev():
     #      predictions = model(inputs)
     #      mae = loss.mae(predictions, targets)

     maex = []
     prg = 0
     model.train() # activates training mode
     total = training_Data.len_batch()

     for mae, pixels in predict(training_Data):
          # back propogate 
          optimizer.zero_grad() # clear gradients
          mae.backward() # compute gradients
          optimizer.step() # update params

          maex += [pixels]
          prg += 1

          print(f"[{prg}/{total}] MAE: {pixels:.4f}", end="\r")

     print(f"Training Loss [{epoch}/{epochs}]: {(sum(maex)/len(maex)):.4f}")

     val_mae = []
     model.eval() # update model mode (from training)
 
     with torch.no_grad(): # no gradient computations

          # for val_inputs, val_targets in validation.to_dev():
          #      val_predictions = model(val_inputs)
          #      mae = loss.mae(val_predictions, val_targets)

          for _, pixels in predict(validation_Data):
               val_mae += [pixels]

               print(f"[val] MAE: {pixels:.4f}", end="\r")

          print(f"Validation Loss [{epoch}/{epochs}]: {(sum(val_mae)/len(val_mae)):.4f}")


# for test_inputs, test_targets in testing.to_dev():
#      test_predictions = model(test_inputs)
#      mae = loss.mae(test_predictions, test_targets)

test_mae = []

for _, pixels in predict(testing_Data):
     test_mae += [pixels]

     print(f"[tst] MAE: {pixels:.4f}", end="\r")

print(f"Test Loss: {(sum(test_mae)/len(test_mae)):.4f}")
