import torch
import torch.nn as nn
import torch.nn.functional as F

class AttributeNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self, input_size, hidden_size, output_size):
        super(AttributeNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x

class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self, input_size, hidden_size):
        super(RelationNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,1)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        #x = F.sigmoid(self.fc2(x))
        x = self.fc2(x)
        return x
    
class IntegratedDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels, att_values):
        super(IntegratedDataset, self).__init__()
        self.features = features
        self.labels = labels
        self.att_values = att_values

    def __getitem__(self, index):
        single_feature, single_label = self.features[index], self.labels[index]
        single_att_value = self.att_values[index]
        
        return single_feature, single_label, single_att_value

    def __len__(self):
        return len(self.features)