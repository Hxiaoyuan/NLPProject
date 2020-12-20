from Chatbot.encoder.net import *


class EncoderRNN(nn.Module):

    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.n_layers = n_layers
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):

        embedded = self.embedding(input_seq)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)

        # 正向通过GRU
        outputs, hidden = self.gru(packed, hidden)
        # 打开填充
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        # 总和双向GRU输出
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        # 返回输出和最终隐藏状态
        return outputs, hidden
