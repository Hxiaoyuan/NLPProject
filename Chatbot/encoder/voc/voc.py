from Chatbot.encoder.config import *
import unicodedata
import re
import random
import torch

PAD_token = 0
SOS_token = 1
EOS_token = 2


class Voc:

    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.word2count = {}
        self.num_words = 3
        self.trimmed = False

    def addSentence(self, sentence):
        words = sentence.split(" ")
        for word in words:
            self.addword(word)

    def addword(self, word):
        if word == ' ':
            return
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, w in self.word2count.items():
            if w >= min_count:
                keep_words.append(k)

        print("keep_words {} / {} = {:.4f}".format(
            len(keep_words), len(self.word2count), len(keep_words) / len(self.word2count)
        ))

        # 字典重新初始化
        self.word2count = {}
        self.word2index = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = {}

        for word in keep_words:
            self.addword(word)


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    s = re.sub('\'', ' ', s)
    return re.sub('\.|\?|,|"', "", s).lower()


def readVocs(datafile, corpus_name):
    print("Reading lines Start...")
    lines = open(datafile, encoding='utf-8').read().strip().split('\n')
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    voc = Voc(corpus_name)
    return voc, pairs


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def loadPrepareData(corpus, corpus_name, datafile, save_dir):
    print("Start preparing training data ...")

    voc, pairs = readVocs(os.path.join(corpus, FORMATTED_MOVIE_LINES), corpus_name)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words ...")
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print("Counted words:", voc.num_words)
    return voc, pairs


def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]


def zeroPadding(l, fillvalue=PAD_token):
    import itertools
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))


def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m


def inputVal(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths


def outputVal(l, voc):
    indexs_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_length = max([len(indexs) for indexs in indexs_batch])
    padList = zeroPadding(indexs_batch)
    mask = binaryMatrix(padList)
    mask = torch.LongTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_length


def batch2TrainData(voc, pair_batch):
    for children in pair_batch:
        children[0] = re.sub(' +', ' ', children[0])

    pair_batch.sort(key=lambda x: len(x[0].split(' ')), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVal(input_batch, voc)
    output, mask, max_length = outputVal(output_batch, voc)
    return inp, lengths, output, mask, max_length


def getVoc():
    save_dir = os.path.join(corpus, 'save')
    datafile = os.path.join(corpus, FORMATTED_MOVIE_LINES)
    voc, pairs = loadPrepareData(corpus, corpus_name, datafile, save_dir)
    # print("\n checking pairs")
    # for pair in pairs[:30]:
    #     print(pair)
    # for pair in pairs[:30]:
    #     inputVal(pair, voc)
    #
    small_batch_size = 5
    batches = batch2TrainData(voc, [random.choice(pairs) for _ in range(small_batch_size)])
    input_variable, lengths, target_variable, mask, max_target_len = batches
    # print("input_variable:", input_variable)
    print("lengths:", lengths)
    # print("target_variable:", target_variable)
    # print("mask:", mask)
    print("max_target_len:", max_target_len)
    return voc, pairs


if __name__ == '__main__':
    save_dir = os.path.join('../', corpus, 'save')
    datafile = os.path.join('../', corpus, FORMATTED_MOVIE_LINES)
    voc, pairs = loadPrepareData(os.path.join('../', corpus), corpus_name, datafile, save_dir)

    save_voc = os.path.join('../', voc_save_path)
    torch.save({
        'name': voc.name,
        'word2index': voc.word2index,
        'index2word': voc.index2word,
        'word2count': voc.word2count,
        'num_words': voc.num_words,
        'trimmed': voc.trimmed,
        'pairs': pairs
    }, save_voc)
