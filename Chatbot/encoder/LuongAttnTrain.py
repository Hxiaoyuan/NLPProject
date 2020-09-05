from Chatbot.encoder.config import *
from Chatbot.encoder.net import *
import Chatbot.encoder.voc.voc as VOC
import torch.optim as optim
import random


def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    # d = crossEntropy.masked_select(mask.byte())
    loss = crossEntropy.masked_select(mask.byte()).mean()
    loss = loss.to(device)
    return loss, nTotal.item()


def train(input_variable, lengths, target_variable, mask, max_target_len,
          encoder, decoder, embedding,
          encoder_optimizer, decoder_optimizer, batch_size, clip, max_length=MAX_LENGTH
          ):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # input_variable = input_variable.to(device)
    # lengths = lengths.to(device)

    # 初始化变量
    loss = 0
    print_losses = []
    n_totals = 0

    encoder_output, encoder_hidden = encoder(input_variable, lengths)

    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    # decoder_input = decoder_input.to(device)
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    for t in range(max_target_len):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_output)
        _, topi = decoder_output.topk(1)
        decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
        decoder_input = decoder_input.to(device)
        # 计算并累计损失
        mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
        loss += mask_loss
        print_losses.append(mask_loss.item() * nTotal)
        n_totals += nTotal

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    return sum(print_losses) / n_totals


def trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer,
               decoder_optimizer, embedding, encoder_n_layers, decoder_n_layers,
               save_dir, n_iteration, batch_size, print_every, save_every, clip,
               corpus_name, start_interation=1, loss=0):
    training_batches = [VOC.batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)])
                        for _ in range(n_iteration)]

    # Initializing
    print("Initializing ...")
    print_loss = 0
    for iteration in range(start_interation, n_iteration + 1):
        training_batche = training_batches[iteration - 1]

        # 从batch 提取字段
        input_variable, lengths, target_variable, mask, max_target_len = training_batche

        # 使用batch运行训练迭代
        loss = train(input_variable, lengths, target_variable, mask, max_target_len,
                     encoder, decoder, embedding, encoder_optimizer, decoder_optimizer,
                     batch_size, clip)
        print_loss += loss

        # 打印进度
        if iteration % PRINT_EVERY == 0:
            print_loss_avg = print_loss / PRINT_EVERY
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}"
                  .format(iteration, iteration / n_iteration * 100, print_loss_avg))
            print_loss = 0

        if iteration % SAVE_EVERY == 0:
            print("model saving ...")
            save_path = os.path.join(MODEL_INFO_SAVE_PATH, LAST_NEW_MODEL_FILE_NAME + '_' + str(iteration) + '.pt')
            torch.save({
                'iteration': iteration,
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'encoder_optimizer': encoder_optimizer.state_dict(),
                'decoder_optimizer': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc_dict': voc.__dict__,
                'embedding': embedding.state_dict()
            }, save_path)
            print("model saving sucess")



batch_size = 64
learning_rate = 0.0005
n_iteration = 8000
print_every = 10
decoder_learning_ratio = 5.0
isLoadModel = True
# isLoadModel = False
modelFileIsExist = os.path.isfile(os.path.join(MODEL_INFO_SAVE_PATH, MODEL_INFO_FILE_NAME))
print("model info file is exist: {}".format(modelFileIsExist))

if __name__ == '__main__':
    print("Building VOC ...")
    # voc, pairs = VOC.getVoc()
    checkpoint = torch.load(voc_save_path)
    voc = VOC.Voc(checkpoint['name'])
    voc.word2count = checkpoint['word2count']
    voc.word2index = checkpoint['word2index']
    voc.index2word = checkpoint['index2word']
    voc.trimmed = checkpoint['trimmed']
    voc.num_words = checkpoint['num_words']
    pairs = checkpoint['pairs']
    print("Building encoder and decoder ...")
    embedding = nn.Embedding(voc.num_words, hidden_size)

    # 初始化编码器及解码器
    start_interation = 1
    loss = 0
    if isLoadModel and modelFileIsExist:
        _path = os.path.join(MODEL_INFO_SAVE_PATH, MODEL_INFO_FILE_NAME)
        checkpoint = torch.load(_path)
        encoder_sd = checkpoint['encoder']
        decoder_sd = checkpoint['decoder']
        embedding_sd = checkpoint['embedding']
        voc_sd = checkpoint['voc_dict']
        encoder_optimizer_sd = checkpoint['encoder_optimizer']
        decoder_optimizer_sd = checkpoint['decoder_optimizer']
        start_interation = checkpoint['iteration'] + 1
        loss = checkpoint['loss']

    if isLoadModel and modelFileIsExist:
        embedding.load_state_dict(embedding_sd)
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder =LuongDecoderRnn('dot', embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
    if isLoadModel and modelFileIsExist:
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)
    print("model build finished !")
    clip = 50.0
    encoder.train()
    decoder.train()

    print("Building Optimizer ...")
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
    if isLoadModel and modelFileIsExist:
        encoder_optimizer.load_state_dict(encoder_optimizer_sd)
        decoder_optimizer.load_state_dict(decoder_optimizer_sd)

    print("Starting Training!")
    trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
               embedding, encoder_n_layers, decoder_n_layers, "", n_iteration, batch_size, print_every,
               12, clip, corpus_name, start_interation=start_interation, loss=loss)