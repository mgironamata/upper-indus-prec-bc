import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.distributions.gamma import Gamma

from utils import init_sequential_weights

__all__ = ['MLP',
           'SimpleCNN',
           'GeoStatCNN',
          ]

class MLP(nn.Module):
    """Multilayer perceptron
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        hidden_channels (list): Number of channels per hidden layer
        likelihood_fn (string): Likelihood function
        dropout_rate (float): Dropout rate
        linear_model (Boolean): If True, nonlinearities are dropped from the model. 
        
    """

    def __init__(self,in_channels,hidden_channels=[10],likelihood_fn='bgmm',dropout_rate=0.25,linear_model=False):

        super(MLP, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.likelihood = likelihood_fn
        self.dropout_rate = dropout_rate
        self.linear_model = linear_model
        
        if self.likelihood == None:
            self.out_channels = 1
        elif self.likelihood == 'gaussian':
            self.out_channels = 2
        elif self.likelihood == 'gamma':
            self.out_channels = 2
        elif self.likelihood == 'gamma_nonzero':
            self.out_channels = 2
        elif self.likelihood == 'ggmm':
            self.out_channels = 5   
        elif self.likelihood == 'bgmm':
            self.out_channels = 3
        elif self.likelihood == 'b2gmm':
            self.out_channels = 6
        elif self.likelihood == 'b2sgmm':
            self.out_channels = 7
        elif self.likelihood == 'btgmm':
            self.out_channels = 4
 
        #self.f = self.build_weight_model()
        self.exp = torch.exp
        self.sigmoid = torch.sigmoid

        # # Linear model
        # if self.linear_model:
        #     self.relu = nn.Identity()
        # else:
        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(self.dropout_rate)

        # Hidden layers
        if self.linear_model:
            self.lin = nn.Linear(self.in_channels, self.out_channels)
        else:
            self.hidden = nn.ModuleList()
            self.hidden.append(nn.Linear(self.in_channels, self.hidden_channels[0]))
            
            for k in range(len(self.hidden_channels)-1):
                self.hidden.append(nn.Linear(self.hidden_channels[k], self.hidden_channels[k+1]))

            # Output layer
            self.out = nn.Linear(self.hidden_channels[-1], self.out_channels)  

        

    def build_weight_model(self):
        """Returns a point-wise function that transforms the in_channels-dimensional
        input features to dimensionality out_channels.

        Returns:
          torch.nn.Module: Linear layer applied point-wise to channels.  
        """

        model = nn.Sequential(
            nn.Linear(self.in_channels, self.hidden_channels),
            nn.ReLU(),
            nn.Linear(self.hidden_channels,self.out_channels),
            )
        init_sequential_weights(model)
        return model

    def forward(self, x):
        
        if self.linear_model:
            x = self.lin(x)
        else:
            for layer in self.hidden[:]:
                x = self.dropout(x)
                x = self.relu(layer(x))
            x = self.out(x)
        
        if self.likelihood==None:
            return x
        elif self.likelihood=='gaussian':
            x[:,1] = self.exp(x[:,1])
            return x
        elif self.likelihood=='gamma_nonzero':
            x[:,:] = self.exp(x[:,:]) # alpha, beta
            return x
        elif self.likelihood=='gamma':
            x[:,:] = self.exp(x[:,:]) # alpha, beta
            return x
        elif self.likelihood=='ggmm':
            x[:,:-1] = self.exp(x[:,:-1]) # alpha1, beta1, alpha2, beta2
            x[:,-1]= self.sigmoid(x[:,-1]) # q: weight paramater for gamma mixture model 
            return x
        elif self.likelihood=='bgmm':
            x[:,0] = self.sigmoid(x[:,0]) # pi
            x[:,1:] = self.exp(x[:,1:]) # alpha, beta
            return x
        elif self.likelihood=='b2gmm':
            x[:,0] = self.sigmoid(x[:,0]) # pi
            x[:,1:-1] = self.exp(x[:,1:-1]) # alpha1, alpha2, beta1, beta2
            x[:,-1] = self.sigmoid(x[:,-1]) # q : weight parameter for gamma mixture model (#TO REVIEW)
            return x
        elif self.likelihood=='b2sgmm':
            x[:,0] = self.sigmoid(x[:,0]) # pi
            x[:,1:5] = self.exp(x[:,1:-2]) # alpha1, alpha2, beta1, beta2
            x[:,5] = self.sigmoid(x[:,-2]) # q : weight parameter for gamma mixture model (TO REVIEW)
            x[:,6] = self.exp(x[:,-1]) # t : threshold 
            return x
        elif self.likelihood=='btgmm':
            x[:,0] = self.sigmoid(x[:,0]) # pi
            x[:,1:-1] = self.exp(x[:,1:]) # alpha, beta
            x[:-1] = self.exp(x[:-1]) # threshold

class SimpleCNN(nn.Module):

    def __init__(self):

        super(SimpleCNN, self).__init__()
        
        self.conv1=nn.Conv2d(in_channels=4, out_channels=6, kernel_size=5)
        self.conv2=nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3)

        self.fc1=nn.Linear(in_features = 12*3*3 + 6, out_features= 100)
        self.fc2=nn.Linear(in_features = 100, out_features= 50)
        self.out=nn.Linear(in_features = 50, out_features= 3)

        self.exp = torch.exp
        self.sigmoid = torch.sigmoid

    def forward(self, t, s):

        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = t.reshape(-1, 12*3*3)
        t = torch.cat((t,s),dim=1)
        
        t = self.fc1(t)
        t = F.relu(t)

        t = self.fc2(t)
        t = F.relu(t)

        t = self.out(t)

        t[:,0] = self.sigmoid(t[:,0]) # pi
        t[:,1:] = self.exp(t[:,1:]) # alpha, beta

        return t

class GeoStatCNN(nn.Module):

    def __init__(self, in_channels, in_features, out_features):

        super(GeoStatCNN, self).__init__()

        self.in_channels = in_channels
        self.in_features = in_features
        self.out_features = out_features
        
        self.conv1=nn.Conv2d(in_channels=self.in_channels, out_channels=128, kernel_size=3, stride=3)
        self.conv2=nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)
        self.conv3=nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)
        self.conv4=nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)
        
        self.fc1=nn.Linear(in_features = self.in_features, out_features= 512)

        self.fc2=nn.Linear(in_features = 1024, out_features= 256)
        self.fc3=nn.Linear(in_features = 256, out_features= 128)
        self.out=nn.Linear(in_features = 128, out_features= self.out_features)

        self.exp = torch.exp
        self.sigmoid = torch.sigmoid

    def forward(self, t, s):

        t = self.conv1(t)
        t = F.relu(t)
        
        t = self.conv2(t)
        t = F.relu(t)
        
        t = self.conv3(t)
        t = F.relu(t)
        
        t = self.conv4(t)
        t = F.relu(t)
        
        t = F.avg_pool2d(t, kernel_size=3, stride=1)

        t = t.reshape(-1, 128*2*2)

        s = self.fc1(s)
        s = F.relu(s)

        t = torch.cat((t,s),dim=1)
        
        t = self.fc2(t)
        t = F.relu(t)

        t = self.fc3(t)
        t = F.relu(t)

        t = self.out(t)

        #t[:,0] = self.sigmoid(t[:,0]) # pi
        #t[:,1:] = self.exp(t[:,1:]) # alpha, beta

        return t


if __name__ == "__main__":

    network = MLP(in_channels=5, 
            hidden_channels=[10], 
            likelihood_fn='bgmm', # 'gaussian', gamma', 'ggmm', bgmm', 'b2gmm', 'b2sgmm'
            dropout_rate=0,
            linear_model=True
           )
    # network = GeoStatCNN(in_channels=1, in_features=3, out_features=2)
    print("Number of parameters is: ", sum(p.numel() for p in network.parameters() if p.requires_grad))