import torch
import torch.nn as nn
import torch.nn.init as init
import warnings

import torch.nn.functional as F
warnings.filterwarnings('ignore')


class MSFCNN(nn.Module):
    def __init__(self, input_dim=46, num_classes=4):
        super(MSFCNN, self).__init__()

        self.conv_layer11 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv_layer12 = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.conv_layer1 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv_layer3 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.conv_layer_w = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=4, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.flatten = nn.Flatten()
        self.fc_layer = nn.Sequential(
            nn.Linear(64*2, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )

        self.apply(self.init_weights)
        self.Residual = MSRN()

    def init_weights(self, m):
        if type(m) == nn.Conv1d or type(m) == nn.Linear:

            init.xavier_uniform_(m.weight)
            if m.bias is not None:

                init.constant_(m.bias, 0.0)

    def forward(self, x1, x2):
        '''
        x1: PSTAAP
        x2: PhysicoChemical
        '''
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x1 = x1.to(device)
        x1 = x1.unsqueeze(1)  # (batch_size, 1, 46)
        x1 = self.conv_layer11(x1)  # (batch_size, 32, 22)
        x1, w1 = self.Residual(x1)  # (batch_size, 64, 1)

        x2 = x2.to(device)  # (batch_size, 49, 12)
        x2 = x2.transpose(1, 2)  # (batch_size, 12, 49)
        x2 = self.conv_layer12(x2)  # (batch_size, 32, 22)
        x2, w2 = self.Residual(x2)  # (batch_size, 64, 1)
    
        x = torch.cat((x1, x2), dim=2)  # (batch_size, 64, 2)
        
        w = torch.cat((w1, w2), dim=2)  # (batch_size, 64, 8)
        w = self.conv_layer_w(w)  # (batch_size, 4, 2)
        w = F.normalize(w, p=2, dim=1)
        x = self.flatten(x)  # (b,64*2)
        
        x = self.fc_layer(x)  # (b,4)
        x = x.unsqueeze(2)  # (b,4,1)
        x = torch.bmm(w, x)
        x = x.squeeze()

        return x



class MSRN(nn.Module):
    def __init__(self, input_dim=46, num_classes=4):
        super(MSRN, self).__init__()

        self.conv_layer1 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool1d(kernel_size=2)
        )

        self.conv_layer2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool1d(kernel_size=2)
        )

        self.conv_layer3 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool1d(kernel_size=2)
        )

        self.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Conv1d or type(m) == nn.Linear:

            init.xavier_uniform_(m.weight)
            if m.bias is not None:

                init.constant_(m.bias, 0.0)

    def forward(self, x):
        x1 = self.conv_layer1(x)  # (64,10)
        x2 = self.conv_layer2(x1)  # (64,4)
        w1 = x2
        x3 = self.conv_layer3(x2)  # (64,1)
        return x3, w1
