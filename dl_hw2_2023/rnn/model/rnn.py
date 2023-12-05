import torch
import torch.nn as nn
import math


class RNN(nn.Module):
    # RNN model is composed of three parts: a word embedding layer, a rnn network and a output layer
    # The word embedding layer have input as a sequence of word index (in the vocabulary) and output a sequence of
    # vector where each one is a word embedding
    #
    # The network has input of each word embedding and output a hidden feature corresponding to each word embedding
    # The output layer has input as the hidden feature and output the probability of each word in the vocabulary
    # feel free to change the init arguments if necessary
    def __init__(self, nvoc, ninput, nhid, nlayers, device='cpu', model='gru'):
        super(RNN, self).__init__()
        print(f"device: {device}")
        self.device = device
        self.drop = nn.Dropout(0.5)

        # self.embed change input size BxL into BxLxE
        self.embed = nn.Embedding(nvoc, ninput)

        # WRITE CODE HERE witnin two '#' bar                                              #
        # Construct you RNN model here. You can add additional parameters to the function #
        ###################################################################################
        # Task 1.1 rnn=GRU
        if model == 'gru':
            self.rnn = nn.GRU(ninput, nhid, nlayers)
        elif model == 'lstm':
            self.rnn = LSTM(ninput, nhid, nlayers)

        ###################################################################################
        self.decoder = nn.Linear(nhid, nvoc)
        self.init_weights()

        self.decoder.to(self.device)
        self.embed.to(self.device)
        self.rnn.to(self.device)

        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        init_uniform = 0.1
        self.embed.weight.data.uniform_(-init_uniform, init_uniform)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_uniform, init_uniform)

    # feel free to change the forward arguments if necessary
    def forward(self, input, state):
        embeddings = self.drop(self.embed(input))

        # WRITE CODE HERE within two '#' bar                                             #
        # With embeddings, you can get your output here.                                 #
        # Output has the dimension of sequence_length * batch_size * number of classes   #
        ##################################################################################
        output, state = self.rnn(embeddings, state)
        ##################################################################################
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), state


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers: int = 1):
        super().__init__()
        # 目前只实现了一层...
        # 多层不知道矩阵怎么拆开、且保持梯度
        assert num_layers == 1

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.params = []

        # W*X_t + U*h_{t-1} + B
        self.W = nn.Parameter(torch.Tensor(input_size, hidden_size * 4))
        self.U = nn.Parameter(torch.Tensor(hidden_size, hidden_size * 4))
        self.B = nn.Parameter(torch.Tensor(hidden_size * 4))

        self.init_weights()

    def init_weights(self):
        init_uniform = 1.0 / math.sqrt(self.hidden_size)
        for p in self.parameters():
            p.data.uniform_(-init_uniform, init_uniform)

    def forward(self, x, state):
        # x.size = (seq, batch_size, vocab_size)
        seq_size, batch_size, _ = x.size()

        if state is None:
            state = (torch.zeros((batch_size, self.hidden_size), device=x.device),
                     torch.zeros((batch_size, self.hidden_size), device=x.device))

        h_t, c_t = state
        hs = self.hidden_size

        hidden_res = []

        for seq_x in x:
            gates = seq_x @ self.W + h_t @ self.U + self.B
            i_t, f_t, o_t, g_t = (
                torch.sigmoid(gates[:, :hs]),
                torch.sigmoid(gates[:, hs:2 * hs]),
                torch.sigmoid(gates[:, 2 * hs:3 * hs]),
                torch.tanh(gates[:, 3 * hs:]),
            )
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_res.append(h_t)

        return torch.cat(hidden_res).reshape([seq_size, batch_size, -1]), (h_t, c_t)

###########################################################################################
