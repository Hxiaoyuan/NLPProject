from Chatbot.encoder.net.base_bert_model import *
from Chatbot.encoder.config import *
from Chatbot.encoder.voc.baseBertVoc import BaseBertVoc
import re


def normalizeString(s):
    s = re.sub('\'', ' ', s)
    return re.sub('\.|\?|,|"', "", s).lower()


def evaluate_input(encoder, decoder, bert_voc, max_length):
    while True:
        input_sentence = ''
        input_sentence = input(">>")
        if input_sentence.lower() == 'q' or input_sentence.lower() == 'exit':
            print("Bot: I will miss you, bye bye...")
            return
        input_sentence = normalizeString(input_sentence)
        try:
            input_sentence = BaseBertVoc.sentenceToBertVec([word for word in input_sentence.split(' ')], bert_voc)
            encoder_output, encoder_hidden = encoder(input_sentence, torch.LongTensor([len(input_sentence)]))
            decoder_hidden = encoder_hidden[:decoder.n_layers]
            all_tokens = torch.zeros([0], dtype=torch.long)
            all_score = torch.zeros([0])
            decoder_input = BaseBertVoc.getWordBertVec(bert_voc.cls, bert_voc)
            for i in range(max_length):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_output)
                decoder_scores, decoder_index = torch.max(decoder_output, dim=1)
                all_tokens = torch.cat((all_tokens, decoder_index), dim=0)
                decoder_input = BaseBertVoc.idsToBertVec(decoder_index.numpy(), bert_voc)
                all_score = torch.cat((all_score, decoder_scores), dim=0)
            decoder_words = [bert_voc.bertTokenizer.convert_ids_to_tokens([token])[0] for token in all_tokens.numpy()]
            print('Bot：', ' '.join([word for word in decoder_words if word != bert_voc.sep and word != bert_voc.cls]))
        except BaseException as e:
            print(e)



def init_model(bert_voc):
    encoder = EncoderGRU(hidden_size=bert_hidden_size, n_layers=bert_n_layers, dropout=bert_dropout)
    decoder = BaseBertAndLuongAttnDecoderGRU(attn_model=bert_att_model_name, hidden_size=bert_hidden_size,
                                             output_size=bert_voc.word_nums, n_layers=bert_n_layers,
                                             dropout=bert_dropout)
    print("加载seq2seq模型...")
    _path = os.path.join('../', 'data', 'save', LOAD_BASE_BERT_MODEL_FILE)
    assert os.path.isfile(_path)
    checkpoint = torch.load(_path)
    encoder.load_state_dict(checkpoint['encoderGRU'])
    decoder.load_state_dict(checkpoint['decoderGRU'])

    return encoder, decoder


if __name__ == '__main__':
    bert_voc = BaseBertVoc()
    print("机器人正在初始化数据...")
    encoder, decoder = init_model(bert_voc)
    evaluate_input(encoder, decoder, bert_voc, MAX_LENGTH)
