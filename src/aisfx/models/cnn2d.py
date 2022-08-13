import torch
import torch.nn as nn
import torch.nn.functional as F

"""
EmbeddingNet function code is from DCASE Cross-Task Baseline Systems: https://arxiv.org/abs/1808.00773
Last accessed February 1, 2022.
"""

def init_layer(layer, nonlinearity='leaky_relu'):
    """Initialize a Linear or Convolutional layer. """
    nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)   

def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.running_mean.data.fill_(0.)
    bn.weight.data.fill_(1.)
    bn.running_var.data.fill_(1.)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ks=3, strd=1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(ks, ks), stride=(strd, strd),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(ks, ks), stride=(strd, strd),
                              padding=(1, 1), bias=False)
                                 
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.init_weights()
    def init_weights(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)
    def forward(self, input, pool_size=(2, 2), pool_type='max'):
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        else:
            raise Exception('Incorrect argument!')
        return x

class EmbeddingNet(nn.Module):
    def __init__(self, pool_type, ks, strd, hl1):
        super(EmbeddingNet, self).__init__()
        """
        Cnn_9layers_AvgPooling
        """
        hl2, hl3, hl4 = hl1*2, hl1*2*2, hl1*2*2*2
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=hl1, ks=ks, strd=strd)
        self.conv_block2 = ConvBlock(in_channels=hl1, out_channels=hl2, ks=ks, strd=strd)
        self.conv_block3 = ConvBlock(in_channels=hl2, out_channels=hl3, ks=ks, strd=strd)
        self.conv_block4 = ConvBlock(in_channels=hl3, out_channels=hl4, ks=ks, strd=strd)
        self.hl4 = hl4
        self.pool_type = pool_type
    def forward(self, input):
        '''
        Input: (batch_size, times_steps, freq_bins)'''
        batch_size = input.shape[0]
        x = torch.unsqueeze(input, 1)
        '''(batch_size, 1, times_steps, freq_bins)'''
        x = self.conv_block1(x, pool_size=(2, 2), pool_type=self.pool_type)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type=self.pool_type)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type=self.pool_type)
        x = self.conv_block4(x, pool_size=(1, 1), pool_type=self.pool_type)
        '''(batch_size, feature_maps, time_steps, freq_bins)'''
        x = torch.mean(x, dim=3)             # (batch_size, feature_maps, time_stpes)
        (output, _) = torch.max(x, dim=2)    # (batch_size, feature_maps)
        return output

class CrossNet(nn.Module):
    def __init__(self, embedding_net, ds_dict, cuda):
        super(CrossNet, self).__init__()

        # Pass in a dictionary of the mappings for each dataset
        # ds_dict = {ds_name: (idx, num_classes)}
        self.ds_dict = ds_dict
        
        self.embedding_net = embedding_net
        hl4 = self.embedding_net.hl4
        self.hl4 = hl4

        l1 = [nn.Linear(hl4, hl4//2).to(cuda) for x in self.ds_dict.keys()]
        l2 = [nn.Linear(hl4//2, hl4//4).to(cuda) for x in self.ds_dict.keys()]
        l3 = [nn.Linear(hl4//4, self.ds_dict[x][1]).to(cuda) for x in self.ds_dict.keys()]
        self.init_weights(l1)
        self.init_weights(l2)
        self.init_weights(l3)
        
        self.fc = nn.ModuleList([nn.Sequential(x,
                                nn.ReLU(),
                                y,
                                nn.ReLU(),
                                z) for x, y, z in zip(l1, l2, l3)])
        for x in self.fc:
            for child in x.children():
                for param in child.parameters():
                    param.requires_grad = False

    def init_weights(self, list_layers):
        for x in list_layers:
            init_layer(x)

    def getDatasetIdx(self, ds_name):
        return self.ds_dict[ds_name][0]

    def classifier_gradSwitch(self, ds_name, switch):     
        ds_idx = self.getDatasetIdx(ds_name)   
        for child in self.fc[ds_idx].children():
            for param in child.parameters():
                param.requires_grad = switch
        for child in self.fc.children():
            for param in child.parameters():
                print(param.requires_grad)

    def forward(self, x, ds_name):
        ds_idx = self.getDatasetIdx(ds_name)   
        output = self.embedding_net(x)
        output_linear = self.fc[ds_idx](output)
        return output, output_linear

    def get_embedding(self, x):
        return self.embedding_net(x)