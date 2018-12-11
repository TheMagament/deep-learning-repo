import torch
import torch.nn as nn

init_range = 0.1
def init_weights(some_layer):
    if type(some_layer) == nn.Embedding:
        some_layer.weight.data.uniform_(-init_range, init_range)
    elif type(some_layer) == nn.Linear:
        some_layer.bias.data.fill_(0)
        some_layer.weight.data.uniform_(-init_range, init_range)
    elif type(some_layer) == nn.LSTM:
        for name, param in some_layer.named_parameters():
            if 'bias' in name:
                nn.init.uniform_(param,-init_range, init_range)
            elif 'weight' in name:
                nn.init.uniform_(param,-init_range, init_range)
    elif type(some_layer) == nn.GRU:
        for name, param in some_layer.named_parameters():
            if 'bias' in name:
                nn.init.uniform_(param,-init_range, init_range)
            elif 'weight' in name:
                nn.init.uniform_(param,-init_range, init_range)


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, word_num, input_size, hidden_units_num, layers_num, dropouth=0.5, dropouti=0.5, initrange=0.25):
        super(RNNModel, self).__init__()
        self.encoder = nn.Embedding(word_num, input_size)
        self.input_drop = nn.Dropout(dropouti)
        assert rnn_type in ['LSTM', 'GRU'], 'RNN type is not supported'
        if rnn_type == 'LSTM':
            self.rnns = [torch.nn.LSTM(input_size if layer == 0 else hidden_units_num,
                                       hidden_units_num if layer != layers_num - 1 else input_size, 1, dropout=0) for
                         layer in range(layers_num)]
        elif rnn_type == 'GRU':
            self.rnns = [torch.nn.GRU(input_size if layer == 0 else hidden_units_num,
                                      hidden_units_num if layer != layers_num - 1 else input_size, 1, dropout=0) for
                         layer in range(layers_num)]

        self.output_drop = nn.Dropout(dropouth)
        print(self.rnns)
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.decoder = nn.Linear(input_size, word_num) #on output


        global init_range
        init_range = initrange
        self.apply(init_weights)

        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_units_num = hidden_units_num
        self.layers_num = layers_num


    def forward(self, input, initial_hidden_states):
        emb = self.encoder(input)
        first_input = self.input_drop(emb)

        raw_output = first_input
        new_hidden = []
        for layer, rnn in enumerate(self.rnns):
            current_input = raw_output
            raw_output, new_h = rnn(current_input, initial_hidden_states[layer])
            new_hidden.append(new_h)
            if layer != self.layers_num - 1:
                raw_output = self.output_drop(raw_output)
        final_hidden_states = new_hidden

        output = self.decoder(raw_output)
        output = output.view(output.size(0) * output.size(1), output.size(2))
        return output, final_hidden_states

    def get_first_hidden(self, batch_size,args):
        tmp = torch.Tensor()
        if args.cuda:
            tmp = tmp.cuda()
        if self.rnn_type == 'LSTM':
            return [(tmp.new(1, batch_size, self.hidden_units_num if layer != self.layers_num - 1 else self.input_size).zero_(),
                     tmp.new(1, batch_size, self.hidden_units_num if layer != self.layers_num - 1 else self.input_size).zero_())
                    for layer in range(self.layers_num)]
        elif self.rnn_type == 'GRU':
            return [tmp.new(1, batch_size, self.hidden_units_num if layer != self.layers_num - 1 else self.input_size).zero_()
                    for layer in range(self.layers_num)]
