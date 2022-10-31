import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.distributions.gamma import Gamma

import pdb

__all__ = ['init_sequential_weights',
           'MLP',
           'VGLM',
           'SimpleCNN',
           'GeoStatCNN',
           'SimpleRNN'
          ]

def _define_out_channels(self):

    if self.likelihood == None:
        return 1
    elif self.likelihood == 'gaussian':
        return 2
    elif self.likelihood == 'gamma':
        return 2
    elif self.likelihood == 'gamma_nonzero':
        return 2
    elif self.likelihood == 'ggmm':
        return 5   
    elif self.likelihood == 'bgmm':
        return 3
    elif self.likelihood == 'b2gmm':
        return 6
    elif self.likelihood == 'b2sgmm':
        return 7
    elif self.likelihood == 'btgmm':
        return 4
    elif self.likelihood == 'bernoulli_gaussian':
        return 3
    elif self.likelihood == 'bernoulli_loggaussian':
        return 3
    elif self.likelihood == 'bernoulli_gumbel':
        return 3
    elif self.likelihood == 'bernoulli_halfnormal':
        return 2    

def _compute_likelihood(self, x):

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
        return x
    elif self.likelihood=='bernoulli_gaussian':
        x[:,0] = self.sigmoid(x[:,0]) # pi
        x[:,2] = self.exp(x[:,2]) # sigma
        return x
    elif self.likelihood=='bernoulli_loggaussian':
        x[:,0] = self.sigmoid(x[:,0]) # pi
        x[:,2] = self.exp(x[:,2]) # sigma
        return x
    elif self.likelihood=='bernoulli_gumbel':
        x[:,0] = self.sigmoid(x[:,0]) # pi
        x[:,2] = self.exp(x[:,2]) # sigma
        return x
    elif self.likelihood=='bernoulli_halfnormal':
        x[:,0] = self.sigmoid(x[:,0]) # pi
        x[:,1] = self.exp(x[:,1]) # sigma
        return x


def init_sequential_weights(model, bias=0.0):
    """Initialize the weights of a nn.Sequential model with Glorot
    initialization.

    Args:
        model (:class:`nn.Sequential`): Container for model.
        bias (float, optional): Value for initializing bias terms. Defaults
            to `0.0`.

    Returns:
        (nn.Sequential): model with initialized weights
    """
    for layer in model:
        if hasattr(layer, 'weight'):
            nn.init.xavier_normal_(layer.weight, gain=1)
        if hasattr(layer, 'bias'):
            nn.init.constant_(layer.bias, bias)
    return model

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

    def __init__(self, in_channels, hidden_channels=[10], likelihood_fn='bgmm', dropout_rate=0.25):

        super(MLP, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.likelihood = likelihood_fn
        self.dropout_rate = dropout_rate
        
        self.out_channels = _define_out_channels(self)

        self.exp = torch.exp
        self.sigmoid = torch.sigmoid
        self.relu = nn.ReLU()
        
        self.dropout = nn.Dropout(self.dropout_rate)

        self.hidden = nn.ModuleList()
        self.hidden.append(nn.Linear(self.in_channels, self.hidden_channels[0]))
        
        for k in range(len(self.hidden_channels)-1):
            self.hidden.append(nn.Linear(self.hidden_channels[k], self.hidden_channels[k+1]))

        # Output layer
        self.out = nn.Linear(self.hidden_channels[-1], self.out_channels)  

    def forward(self, x):
        
        for layer in self.hidden[:]:
            x = self.dropout(x)
            x = self.relu(layer(x))
        x = self.out(x)

        x = _compute_likelihood(self, x)

        return x

class VGLM(nn.Module):
        
    def __init__(self, in_channels, likelihood_fn='bgmm'):

        super(VGLM, self).__init__()
        
        self.in_channels = in_channels
        self.likelihood = likelihood_fn
        self.out_channels = _define_out_channels(self)

        self.exp = torch.exp
        self.sigmoid = torch.sigmoid

        self.lin = nn.Linear(self.in_channels, self.out_channels)

    def forward(self, x):
        
        x = self.lin(x)
        x = _compute_likelihood(self, x)

        return x

class SimpleRNN(nn.Module):

    def __init__(self, in_channels, likelihood_fn='bgmm'):

        super(SimpleRNN, self).__init__()

        self.in_channels = in_channels
        self.likelihood = likelihood_fn
        
        self.out_channels = _define_out_channels(self)

        self.rnn = nn.RNN(input_size = self.in_channels, hidden_size = self.out_channels, num_layers = 1, batch_first=True)

        self.exp = torch.exp
        self.sigmoid = torch.sigmoid

    def forward(self, x):

        # x = torch.unsqueeze(x,1) # makes shape: [seq_length, batch_size = 1, input_channels] 
        # input (x) shape should be: batch size, seq length, channels
        z = self.rnn(x)[0] # [seq_length, batch_size = 1, output_channels]
        t = z.reshape(z.shape[0]*z.shape[1], z.shape[2]) # [seq_length, output_channels]
        x = _compute_likelihood(self, t)
        # x - torch.squeeze(x, 1)
        # print(x.shape)

        return x

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

    # network = MLP(in_channels=5, 
    #         hidden_channels=[10], 
    #         likelihood_fn='bgmm', # 'gaussian', gamma', 'ggmm', bgmm', 'b2gmm', 'b2sgmm'
    #         dropout_rate=0,
    #         linear_model=True
    #        )

    # network = GeoStatCNN(in_channels=1, in_features=3, out_features=2)

    network = SimpleRNN(in_channels=5, likelihood_fn='bgmm')

    print("Number of parameters is: ", sum(p.numel() for p in network.parameters() if p.requires_grad))