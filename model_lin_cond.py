import torch.nn as nn
import torch.optim
from torch.nn import DataParallel
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), 640, 1, 1)

class Shape(nn.Module):
    def forward(self, input):
        print(input.shape)
        return input

class Encoder(nn.Module):
    def __init__(self, input_dim=256, num_channels=2, latent_dim=32, label_size=2, device=torch.device('cuda')):
        super(Encoder, self).__init__()
        self.device = device
        self.fc1 = nn.Linear((input_dim*num_channels)+label_size,1024)
        self.fc2 = nn.Linear(1024,512)
        self.fc3 = nn.Linear(512,256)
        self.fc41 = nn.Linear(256,latent_dim)
        self.fc42 = nn.Linear(256,latent_dim)
    
    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size())
        esp = esp.to(dtype=torch.float64).to(self.device)
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc41(h), self.fc42(h)
        z = self.reparametrize(mu, logvar).to(self.device)
        return z, mu, logvar
    
    def encode(self,x,c):
        xc = torch.cat([x,c], dim=1)
        h1 = F.relu(self.fc1(xc))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        z, mu, logvar = self.bottleneck(h3)
        return z, mu, logvar
    
    def forward(self, x, c):
        z, mu, logvar = self.encode(x,c)
        return z, mu, logvar
    
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

class Decoder(nn.Module):
    def __init__(self,out_dim,num_channels,latent_dim,label_size,device):
        super(Decoder,self).__init__()
        self.device = device
        self.fc5 = nn.Linear(latent_dim+label_size,256)
        self.fc6 = nn.Linear(256,512)
        self.fc7 = nn.Linear(512,1024)
        self.fc8 = nn.Linear(1024,out_dim*num_channels)
    
    def decode(self,z,c):
        zc = torch.cat([z,c],dim=1)
        h1 = F.relu(self.fc5(zc))
        h2 = F.relu(self.fc6(h1))
        h3 = F.relu(self.fc7(h2))
        h4 = self.fc8(h3)
        return F.sigmoid(h4)
    
    def forward(self,z,c):
        z = self.decode(z,c)
        return z
    
class VAE(nn.Module):
    def __init__(self, input_dim=256, num_channels=2, latent_dim=32, label_size=2, device=torch.device('cuda')):
        super(VAE, self).__init__()
        self.device = device
        
        self.encoder = Encoder(input_dim, num_channels,latent_dim,label_size,device)
        self.decoder = Decoder(input_dim, num_channels,latent_dim,label_size,device)
        
    def forward(self, x, c):
        z, mu, logvar = self.encoder(x,c)
        z = self.decoder(z,c)
        return z, mu, logvar
    
    
def get_cond_model(dev, input_dim, num_channels, latent_dim, label_size, lr=1e-3):
    model = VAE(input_dim, num_channels, latent_dim, label_size, dev)
    model = model.to(dev).double()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    return model, opt

def load_cond_model(path, input_dim, num_channels, latent_dim=32, label_size=2, dev=torch.device('cuda')):
    model = VAE(input_dim, num_channels, latent_dim, label_size, dev).double()
    model.load_state_dict(torch.load(path, map_location=dev))
    return model
