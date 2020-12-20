from Chatbot.encoder.config import *
import Chatbot.encoder.voc.voc as VOC
import torch.nn as nn
from Chatbot.encoder.net import *


def evaluate(encoder, decoder, input_seq, input_length, max_length):
    encoder_outputs, encoder_hidden = encoder(input_seq, input_length)
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    decoder_input = torch.ones(1, 1, dtype=torch.long) * SOS_token
    all_tokens = torch.zeros([0], dtype=torch.long)
    all_scores = torch.zeros([0])

    for _ in range(max_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)

        decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
        decoder_input = decoder_input.view(1, 1)
        all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
        all_scores = torch.cat((all_scores, decoder_scores), dim=0)
        # decoder_input = torch.unsqueeze(decoder_input, 1)
        # decoder_input = decoder_input.view((decoder_input.shape[1], decoder_input[2]))
    return all_tokens, all_scores


def evaluateInput(encoder, decoder, voc):
    input_sentence = ''
    while True:
        try:
            input_sentence = input('> ')

            if input_sentence.lower() == 'q' or input_sentence.lower() == 'exit':
                print("I will miss you, bye bye...")
                break
            input_sentence = VOC.normalizeString(input_sentence)
            indexs_batch = [VOC.indexesFromSentence(voc, input_sentence)]
            lengths = torch.tensor([len(indexes) for indexes in indexs_batch])
            # lengths = torch.tensor(l)
            input_batch = torch.LongTensor(indexs_batch).transpose(0, 1)
            tokens, socres = evaluate(encoder, decoder, input_batch, lengths, MAX_LENGTH)
            decoder_words = [voc.index2word[token.item()] for token in tokens]
            outwords = [x for x in decoder_words if not (x == 'EOS' or x == 'PAD')]
            print('Bot：', ' '.join(outwords))
            # return decoder_words
        except KeyError:
            print("Error: Encountered unknown word.")


def initModel(voc):
    embedding = nn.Embedding(voc.num_words, hidden_size)
    _path = os.path.join(LOAD_MODEL_FILE_PATH, LOAD_MODEL_FILE_NAME)
    checkpoint = torch.load(_path)
    encoder_sd = checkpoint['encoder']
    decoder_sd = checkpoint['decoder']
    embedding_sd = checkpoint['embedding']
    voc_sd = checkpoint['voc_dict']
    loss = checkpoint['loss']
    embedding.load_state_dict(embedding_sd)
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = LuongDecoderRnn('dot', embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)

    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
    return encoder, decoder


if __name__ == '__main__':
    checkpoint = torch.load(voc_save_path)
    voc = VOC.Voc(checkpoint['name'])
    voc.word2count = checkpoint['word2count']
    voc.word2index = checkpoint['word2index']
    voc.index2word = checkpoint['index2word']
    voc.trimmed = checkpoint['trimmed']
    voc.num_words = checkpoint['num_words']

    encoder, decoder = initModel(voc)

    evaluateInput(encoder, decoder, voc)
