from Chatbot.encoder.net import *
import torch.nn.functional as F


# 根据luong et al.发表的global attention 机制
class Attn(nn.Module):

    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden__size = hidden_size

        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "：该方法并不存在")
        if self.method == 'general':
            self.attn = torch.nn.Linear(self.hidden__size, self.hidden__size)
        elif self.method == 'concat':
            self.attn = torch.nn.Linear(self.hidden__size * 2, hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden*encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    # def concat_score(self, hidden, encoder_output):

    def forward(self, hidden, encoder_outputs):
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        attn_energies = attn_energies.t()

        return F.softmax(attn_energies, dim=1).unsqueeze(1)
