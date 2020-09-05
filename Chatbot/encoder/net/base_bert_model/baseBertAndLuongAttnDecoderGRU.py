from Chatbot.encoder.net import *
from Chatbot.encoder.net.attn import Attn
import torch.nn.functional as F


class BaseBertAndLuongAttnDecoderGRU(nn.Module):

    def __init__(self, attn_model, hidden_size, output_size, n_layers=1, dropout=0):
        super(BaseBertAndLuongAttnDecoderGRU, self).__init__()
        self.hidden_size = hidden_size
        self.attn = Attn(attn_model, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.n_layers = n_layers
        self.embedding_dropout = nn.Dropout(dropout)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout))

    def forward(self, input_voc, last_hidden, encoder_output):
        embedded = self.embedding_dropout(input_voc)
        # GRU 单向转发
        rnn_output, hidden = self.gru(embedded, last_hidden)

        # 计算注意力
        attn_weights = self.attn(rnn_output, encoder_output)
        # 获取上下文
        context = attn_weights.bmm(encoder_output.transpose(0, 1))

        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))

        # 预测下一个单词
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)

        return output, hidden