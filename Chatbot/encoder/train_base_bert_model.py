from Chatbot.encoder.net.base_bert_model import *
from Chatbot.encoder.config import *
import torch.optim as optim
from Chatbot.encoder.voc.baseBertVoc import BaseBertVoc
import random


def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    try:
        crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    except BaseException as e:
        print(e)
        return None, None
    # d = crossEntropy.masked_select(mask.byte())
    loss = crossEntropy.masked_select(mask.byte()).mean()
    loss = loss.to(device)
    return loss, nTotal.item()


def trainIters(bertVoc, encoder, decoder, encoder_optimizer, decoder_optimizer, start_interation=0):

    pairs = BaseBertVoc.loadCorpusPairs(bertVoc)
    training_batches = [BaseBertVoc.batch2TrainData(
        [random.choice(pairs) for _ in range(bert_batch_size)], bertVoc)
        for _ in range(bert_n_iteration)]
    start_interation = 0
    print_loss = 0
    for iteration in range(start_interation, bert_n_iteration):
        train_batche = training_batches[iteration]
        _input, input_lengths, _output, mask, out_max_length = train_batche
        loss = train(_input, input_lengths, _output, mask, out_max_length,
                     encoder, decoder, encoder_optimizer, decoder_optimizer, bertVoc)

        if loss is None:
            print_loss = 0
            break
        print_loss += loss
        # 打印进度
        if iteration % BERT_PRINT_EVERY == 0:
            print_loss_log = print_loss / BERT_PRINT_EVERY
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}"
                  .format(iteration, iteration / (bert_n_iteration - 1) * 100, print_loss_log))
            print_loss = 0

        if (iteration + 1) % BERT_SAVE_MODEL_EVERY == 0:
            save_path = os.path.join(LOAD_BASE_BERT_MODEL_FILE_PATH, SAVE_BASE_BERT_MODEL_FILE + str(iteration) + '.pt')
            torch.save({
                'interation': iteration,
                'loss': loss,
                'encoderGRU': encoder.state_dict(),
                'decoderGRU': decoder.state_dict(),
                'encoder_optimizer': encoder_optimizer.state_dict(),
                'decoder_optimizer': decoder_optimizer.state_dict()
            }, save_path)


def train(_input, input_lengths, _output, mask, out_max_length,
          encoder, decoder, encoder_optimizer, decoder_optimizer, bert_voc):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # 初始化变量
    loss = 0
    print_losses = []
    n_totals = 0
    # _input
    encoder_output, encoder_hidden = encoder(_input, input_lengths)

    # decoder_input = torch.LongTensor([[SOS_token for _ in range(bert_batch_size)]])
    decoder_input = None
    for _ in range(bert_batch_size):
        if decoder_input is None:
            decoder_input = BaseBertVoc.getWordBertVec(bert_voc.cls, bert_voc)
        else:
            decoder_input = torch.cat((decoder_input, BaseBertVoc.getWordBertVec(bert_voc.cls, bert_voc)), 1)
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    for t in range(out_max_length):
        decoder_out, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_output)
        _, topi = decoder_out.topk(1)
        # decoder_input = torch.LongTensor([[topi[i][0] for i in range(bert_batch_size)]])
        decoder_input = [topi[i][0] for i in range(bert_batch_size)]
        decoder_input = BaseBertVoc.idsToBertVec(decoder_input, bert_voc)
        decoder_input = decoder_input.to(device)
        # 计算损失
        mask_loss, nTotal = maskNLLLoss(decoder_out, _output[t], mask[t])
        if mask_loss is None:
            return
        loss += mask_loss
        print_losses.append(mask_loss.item() * nTotal)
        n_totals += nTotal
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    return sum(print_losses) / n_totals


if __name__ == '__main__':
    print("Building VOC ...")
    # voc, pairs = VOC.getVoc()
    # checkpoint = torch.load(voc_save_path)
    # voc = VOC.Voc(checkpoint['name'])
    bertVoc = BaseBertVoc()
    print("Building moding...")

    # 初始化编码器及解码器
    start_interation = 0
    encoderGRU = EncoderGRU(bert_hidden_size, encoder_n_layers, bert_dropout)
    bertDecoderRGU = BaseBertAndLuongAttnDecoderGRU(bert_att_model_name, bert_hidden_size, bertVoc.word_nums, bert_n_layers,
                                                    bert_dropout)
    encoderGRU.train()
    bertDecoderRGU.train()

    if IS_LOAD_MODEL:
        assert os.path.isfile(os.path.join(LOAD_BASE_BERT_MODEL_FILE_PATH, LOAD_BASE_BERT_MODEL_FILE))
        load_file_checkpoint = torch.load(os.path.join(LOAD_BASE_BERT_MODEL_FILE_PATH, LOAD_BASE_BERT_MODEL_FILE))
        encoderGRU_sd = load_file_checkpoint['encoderGRU']
        bertDecoderGRU_sd = load_file_checkpoint['decoderGRU']
        start_interation = load_file_checkpoint['interation'] + 1
        loss = load_file_checkpoint['loss']
        encoderGRU.load_state_dict(encoderGRU_sd)
        bertDecoderRGU.load_state_dict(bertDecoderGRU_sd)

    print("Building Optimizer ...")
    encoder_optimizer = optim.Adam(encoderGRU.parameters(), lr=bert_lr)
    decoder_optimizer = optim.Adam(bertDecoderRGU.parameters(), lr=bert_lr)

    if IS_LOAD_MODEL:
        assert os.path.isfile(os.path.join('../', 'data', 'save', LOAD_BASE_BERT_MODEL_FILE))
        encoder_optimizer.load_state_dict(load_file_checkpoint['encoder_optimizer'])
        decoder_optimizer.load_state_dict(load_file_checkpoint['decoder_optimizer'])
    # 加载模型书写
    print("Loading Train corpus...")

    print("Start Training")
    trainIters(bertVoc, encoderGRU, bertDecoderRGU, encoder_optimizer, decoder_optimizer,
               start_interation=start_interation)
