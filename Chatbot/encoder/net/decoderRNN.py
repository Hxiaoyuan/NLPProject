from Chatbot.encoder.net import *
import torch.nn.functional as F


class LuongAttnDecoderRNN(nn.Module):

    def __init__(self, embedding, hidden_size, output_size, attn_model=None, n_layer=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.n_layers = n_layer

        self.gru = nn.GRU(hidden_size, hidden_size, n_layer, dropout=(0 if n_layer == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)

        # 单向GRU转发
        rnn_out, hidden = self.gru(embedded, last_hidden)
        output = self.out(rnn_out)
        output = F.softmax(output, dim=2)
        return output, hidden
