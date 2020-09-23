from Chatbot.encoder.net.base_bert_model import *
from Chatbot.encoder.config import *
from Chatbot.encoder.voc.baseBertVoc import BaseBertVoc


def init_model():
    bert_model = BaseBertVoc()
    encoder = EncoderGRU(hidden_size=bert_hidden_size, n_layers=bert_n_layers, dropout=bert_dropout)
    decoder = BaseBertAndLuongAttnDecoderGRU(attn_model=bert_att_model_name, hidden_size=bert_hidden_size,
                                             output_size=bert_model.word_nums, n_layers=bert_n_layers, dropout=bert_dropout)
    print("加载seq2seq模型...")
    _path = os.path.join('../', 'data', 'save', LOAD_BASE_BERT_MODEL_FILE)
    assert os.path.isfile(_path)
    checkpoint = torch.load(_path)
    encoder.load_state_dict(checkpoint['encoderGRU'])
    decoder.load_state_dict(checkpoint['decoderGRU'])

    return encoder, decoder



if __name__ == '__main__':

    print("机器人正在初始化模型")