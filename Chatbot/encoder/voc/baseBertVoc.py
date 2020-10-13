from Chatbot.encoder.config import *
from pytorch_pretrained_bert import BertModel, BertTokenizer
import re
import torch


def normalizeString(s):
    s = re.sub('\'', ' ', s)
    return re.sub('\.|\?|,|"', "", s).lower()


'''
    可以做实验一： 使用bert 在decoder种生成每个词的向量
    实验二： 使用bert 在decoder 中根据上文生成最后一个词的向量  
    可以综合以上几个实验做下对比
'''


class BaseBertVoc():

    def __init__(self):
        super(BaseBertVoc, self).__init__()
        self.bertTokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.cls = '[CLS]'
        self.sep = '[SEP]'
        self.bertModel = BertModel.from_pretrained('bert-base-uncased')
        self.word_nums = len(self.bertTokenizer.vocab)
        self.bertModel.eval()

    @staticmethod
    def zeroPadding(input_sen):
        import itertools

        tmp_l = list(itertools.zip_longest(*input_sen, fillvalue=BERT_PAD))
        result = []
        for r in range(len(input_sen)):
            sen = []
            for l in range(len(tmp_l)):
                sen.append(tmp_l[l][r])
            result.append(sen)
        return result

    @staticmethod
    def inputInit(input_sen):
        lengths = torch.tensor([len(sen) for sen in input_sen])
        input_sen = BaseBertVoc.zeroPadding(input_sen)
        return input_sen, lengths

    @staticmethod
    def outputInit(output_sen):
        max_length = max([len(output) for output in output_sen])
        output_sen = BaseBertVoc.zeroPadding(output_sen)
        mask = BaseBertVoc.binaryMatrix(output_sen)
        mask = torch.LongTensor(mask).t()
        return output_sen, mask, max_length

    @staticmethod
    def filterPair(p):
        return len(p[0]) < bert_max_length and len(p[1]) < bert_max_length

    @staticmethod
    def filterPairs(pairs):
        return [pair for pair in pairs if BaseBertVoc.filterPair(pair)]

    @staticmethod
    def loadcorpus(filepath, filename):
        lines = open(os.path.join(filepath, filename), encoding='utf-8').read().strip().split('\n')
        pairs = [[pair for pair in l.split('\t')] for l in lines]
        return pairs

    @staticmethod
    def loadCorpusPairs(bertVoc):
        sen_pairs = BaseBertVoc.loadcorpus(LOAD_CORPUS_PATH, LOAD_CORPUS_NAME)
        pairs = []
        tokenizer = bertVoc.bertTokenizer
        for sen_pair in sen_pairs:
            pairs.append([tokenizer.tokenize(normalizeString(sen)) for sen in sen_pair])

        return BaseBertVoc.filterPairs(pairs)

    @staticmethod
    def batch2TrainData(batch_pair, bert_voc):
        # for pair in batch_pair
        batch_pair.sort(key=lambda x: len(x[0]), reverse=True)
        input_batch, output_batch = [], []
        for pair in batch_pair:
            input_batch.append(pair[0])
            output_batch.append(pair[1])

        # 获得句子真实长度及填充
        inps, inp_length = BaseBertVoc.inputInit(input_batch)
        inps = BaseBertVoc.sentencesToBertVec(inps, bert_voc)
        # for inp in inps:
        outs, mask, out_length = BaseBertVoc.outputInit(output_batch)
        outs = BaseBertVoc.handlerTargetSentens(outs, bert_voc)
        return inps, inp_length, outs, mask, out_length

    @staticmethod
    def binaryMatrix(l):
        m = []
        for i, seq in enumerate(l):
            m.append([])
            for word in seq:
                if word == BERT_PAD:
                    m[i].append(0)
                else:
                    m[i].append(1)
        return m

    @staticmethod
    def sentenceToBertVec(sentence, bert_voc):
        marked_text = [bert_voc.cls] + sentence + [bert_voc.sep]
        indexed_token = bert_voc.bertTokenizer.convert_tokens_to_ids(marked_text)
        segment_id = [1] * len(indexed_token)
        token_tensor = torch.tensor([indexed_token])
        segment_tensor = torch.tensor([segment_id])
        with torch.no_grad():
            encoded_layers, _ = bert_voc.bertModel(token_tensor, segment_tensor)
        tokens_embeddings = []
        for token_i in range(len(marked_text)):
            hidden_layers = []
            for layer_i in range(len(encoded_layers)):
                hidden_layers.append(encoded_layers[layer_i][0][token_i])
            tokens_embeddings.append(hidden_layers)
        # 使用后四行作为向量
        vec_last4 = [torch.sum(torch.stack(embedding)[-4:], 0) for embedding in tokens_embeddings]
        del vec_last4[0]
        del vec_last4[len(vec_last4) - 1]
        vec_last4 = [vec.tolist() for vec in vec_last4]
        vec_last4 = torch.tensor(vec_last4)
        return vec_last4.view(vec_last4.shape[0], 1, -1)

    @staticmethod
    def sentencesToBertVec(sentences, bert_voc):
        result = None
        for sentence in sentences:
            if result is None:
                result = BaseBertVoc.sentenceToBertVec(sentence, bert_voc)
            else:
                result = torch.cat((result, BaseBertVoc.sentenceToBertVec(sentence, bert_voc)), 1)
            # result.append(BaseBertVoc.sentenceToBertVec(sentence, bert_voc))
        return result

    @staticmethod
    def handlerTargetSentens(sentences, bert_voc):
        token_ids = [bert_voc.bertTokenizer.convert_tokens_to_ids(sentence) for sentence in sentences]
        token_tensor = torch.tensor(token_ids).t()
        return token_tensor

    # 得到单词的向量
    @staticmethod
    def getWordBertVec(word, bert_voc):
        token_index = bert_voc.bertTokenizer.convert_tokens_to_ids(
            bert_voc.bertTokenizer.tokenize(word))
        token_tensor = torch.tensor([token_index])
        segment_tensor = torch.tensor([[1] * len(token_index)])
        encoder_layers, _ = bert_voc.bertModel(token_tensor, segment_tensor)
        tokens_embeddings = []
        for token_i in range(len(token_index)):
            hidden_layers = []
            for layer_i in range(len(encoder_layers)):
                hidden_layers.append(encoder_layers[layer_i][0][token_i])
            tokens_embeddings.append(hidden_layers)
        # 使用后四行作为向量
        vec_last4 = [torch.sum(torch.stack(embedding)[-4:], 0) for embedding in tokens_embeddings]
        return vec_last4[0].view(1, 1, -1)

    @staticmethod
    def idsToBertVec(ids, bert_voc):
        segment_tensor = torch.tensor([[1] * len(ids)])
        encoder_layers, _ = bert_voc.bertModel(torch.tensor([ids]), segment_tensor)
        token_embedding = []
        for token_i in range(len(ids)):
            hidden_layers = []
            for layer_i in range(len(encoder_layers)):
                hidden_layers.append(encoder_layers[layer_i][0][token_i])
            token_embedding.append(hidden_layers)
        vec_last4 = [torch.sum(torch.stack(embedding)[-4:], 0) for embedding in token_embedding]
        vec_last4 = [vec.tolist() for vec in vec_last4]
        vec_last4 = torch.tensor(vec_last4)

        return vec_last4.view(1, vec_last4.shape[0], -1)


if __name__ == '__main__':
    bertVoc = BaseBertVoc()
    pairs = BaseBertVoc.loadCorpusPairs(bertVoc)

    # a = BaseBertVoc.getWordBertVec(bertVoc.cls, bertVoc)
    # b = BaseBertVoc.getWordBertVec(bertVoc.sep, bertVoc)
    n_iteration = 2
    batch_size = 6
    import random

    training_batches = [BaseBertVoc.batch2TrainData(
        [random.choice(pairs) for _ in range(batch_size)], bertVoc)
        for _ in range(n_iteration)]
