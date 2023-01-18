import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTM(nn.Module):
    def __init__(self, input_feature_dim=2048, fc_dim=1024, hidden_size=512,
        input_drop_rate=0.3, lstm_drop_rate=0.6, fc_drop_rate=0.7, use_bn=True):
        super(BiLSTM, self).__init__()

        input_size = input_feature_dim
        output_size = fc_dim
        self.embed_sizes = input_feature_dim
        self.embed_fc = nn.Linear(input_size, output_size)
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=output_size,
            hidden_size=self.hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=lstm_drop_rate,
            bidirectional=True
        )
        # The probability is set to 0 by default
        self.input_shotmask = ShotMask(p=0)
        self.input_dropout = nn.Dropout(p=input_drop_rate)
        self.fc_dropout = nn.Dropout(p=fc_drop_rate)
        self.fc1 = nn.Linear(self.hidden_size*2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)
        self.softmax = nn.Softmax(2)
        self.use_bn = use_bn
        
        if self.use_bn:
            self.bn1 = nn.BatchNorm1d(output_size)
            self.bn2 = nn.BatchNorm1d(hidden_size)
        
        
    def forward(self, x, y):
        if self.training:
            x = self.input_shotmask(x, y)
        x = self.input_dropout(x)
        x = self.embed_fc(x)
        
        if self.use_bn:
            seq_len, C = x.shape[1:3]
            x = x.view(-1, C)
            x = self.bn1(x)
            x = x.view(-1, seq_len, C)
        
        x = self.fc_dropout(x)
        self.lstm.flatten_parameters()
        out, (_, _) = self.lstm(x, None)
        out = self.fc1(out)
        if self.use_bn:
            seq_len, C = out.shape[1:3]
            out = out.view(-1, C)
            out = self.bn2(out)
            out = out.view(-1, seq_len, C)
        out = self.fc_dropout(out)
        out = F.relu(out)
        out = self.fc2(out)
        if not self.training:
            out = self.softmax(out)
        return out


class ShotMask(nn.Module):
    '''
    Drop the shot from the middle of a scene
    '''
    def __init__(self, p=0.2):
        super(ShotMask, self).__init__()
        self.p = p

    def forward(self, x, y):
        # keep the cue
        B, L , _ = x.size()
        y_shift = torch.cat([torch.zeros(B,1,1).bool().to(y.device), y.bool()],dim=1)[:,:L,:]
        self.mask = torch.rand(*y.size()) >= self.p
        self.mask = self.mask.bool().to(x.device) | y.bool() | y_shift
        out = x.mul(self.mask)
        return out

if __name__ == '__main__':
    B, seq_len, C = 10, 20, 2048
    input = torch.randn(B, seq_len, C)
    model = BiLSTM()
    out = model(input)
    # torch.Size([10, 20, 2])
    print(out.size())