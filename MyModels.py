
import torch.nn as nn

class MultiStreamNet(nn.Module):
    def __init__(self, modelA, modelB):
        super(MultiStreamNet, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.classifier = nn.Linear(200, 200)
        
    def forward(self, x1, x2):
        x1 = self.modelA(x1)
        x2 = self.modelB(x2)
        x = torch.cat((x1, x2), dim=1)
        x = self.classifier(F.relu(x))
        return x


class RNN(nn.Module):
    
    def __init__(self, input_size, output_size, hidden_dim, n_layers, dropout=0.25):

        super(RNN, self).__init__()

        self.input_size  = input_size
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.n_layers  =  n_layers
        self.dropout  =  dropout
        
        
        # define model layers
        self.lstm = nn.LSTM( input_size, hidden_dim, n_layers, dropout=dropout, batch_first=True)        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim,hidden_dim )
        self.fcc2=nn.Linear(hidden_dim,output_size)

    
    
    def forward(self, nn_input, hidden):

        batch_size = nn_input.size(0)
        seq_len    = nn_input.size(1)
        lstm_output, hidden = self.lstm(nn_input, (hidden[0], hidden[1]))
        lstm_output = lstm_output.contiguous().view( -1, self.hidden_dim)
        out = self.fc(lstm_output.float())
        out = self.fcc2(out)
        out = out.view(batch_size, -1, self.output_size)
        out = out[:, -1]
        #out = F.log_softmax(out, dim=1)

        return out, hidden
    
    
    def init_hidden(self, batch_size):

        weight = next(self.parameters()).data
        
        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        
        return hidden
        
        
