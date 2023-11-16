import torch.nn as nn
import torch



class DeepLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(DeepLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)

        self.lstm_encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.lstm_decoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)

        self.attention = nn.MultiheadAttention(hidden_size, num_heads=1)

        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()

        out, (hn, cn) = self.lstm_encoder(x, (h0, c0))

        # 调整维度以适应注意力机制的输入要求
        out = out.permute(1, 0, 2)  # 将 seq_len 维度放在第一位
        # 注意力机制处理
        out, _ = self.attention(out, out, out)
        out = out.permute(1, 0, 2)  # 将 seq_len 维度放回原位

        out, (_, _) = self.lstm_decoder(out, (hn, cn))

        out = out[:, -1, :]

        out = self.fc1(out)
        out = self.dropout(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.fc2(out)
        out = self.dropout(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.fc_out(out)
        return out


if __name__=='__main__':
    model = DeepLSTM(input_size=78, hidden_size=128, output_size=384, num_layers=1)

    input = torch.randint(0, 1, (128,200,78)).float()
    input = input.cuda()

    model.cuda()

    out = model(input)

    print(out.shape)