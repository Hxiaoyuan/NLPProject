import torch
import os
import csv
from Chatbot.encoder.voc import *

MAX_LENGTH = 10
hidden_size = 500
model_name = 'cb_model'
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1

# 数据路径相关
corpus_name = "cornell movie-dialogs corpus"
corpus = os.path.join('../', 'data', corpus_name)
corpus_movie_lines = "movie_lines.txt"
corpus_movie_conversations_lines = "movie_conversations.txt"

USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')
print("当前电脑数据训练使用：{}".format(device))

# voc数据保存路径
voc_save_path = os.path.join('../', 'data', 'save', 'voc_save_path.pt')

# 预处理文件相关
MOVIE_LINES_FIELDS = ['lineID', 'characterID', 'movieID', 'character', 'text']
MOVIE_CONVERSATIONS_FIELDS = ['characterID', 'character2ID', 'movieID', "utteranceIDs"]

FORMATTED_MOVIE_LINES = "formatted_movie_lines.txt"

PAD_token = 0
SOS_token = 1
EOS_token = 2

# 模型数据保存读取配置
SAVE_EVERY = 50
PRINT_EVERY = 10
MODEL_INFO_FILE_NAME = 'model_parameter_Luong_6450.pt'
# MODEL_INFO_FILE_NAME = 'model_parameter_4000.pt'
MODEL_INFO_SAVE_PATH = os.path.join('../', 'data', 'save')
LAST_NEW_MODEL_FILE_NAME = "model_parameter_Luong"

# 模型加载配置
# LOAD_MODEL_FILE_NAME = 'model_parameter_1600.pt'
LOAD_MODEL_FILE_NAME = 'model_parameter_Luong_7150.pt'
LOAD_MODEL_FILE_PATH = os.path.join('../', 'data', 'save')

# ============================== 基于bert的改良机器人 config ======================

'''

'''

bert_batch_size = 2
bert_n_iteration = 6

'''
    bert_max_length: 训练序列的最大长度
'''
bert_max_length = 12
bert_hidden_size = 768
bert_n_layers = 2
bert_att_model_name = 'dot'
bert_dropout = 0.1
bert_lr = 0.01

BERT_PAD = '[PAD]'

LOAD_CORPUS_PATH = os.path.join('../', 'data', corpus_name)
LOAD_CORPUS_NAME = "formatted_movie_lines_test.txt"
LOAD_BASE_BERT_MODEL_FILE = 'model_base_bert_luong_'
LOAD_BASE_BERT_MODEL_FILE_PATH = os.path.join('../', 'data', 'save')
