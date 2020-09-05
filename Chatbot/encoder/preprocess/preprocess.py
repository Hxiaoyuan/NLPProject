from Chatbot.encoder.config import *


def printLines(corpus_path, n=10):
    with open(corpus_path, 'rb') as dataFiles:
        lines = dataFiles.readlines()
        for line in lines[:n]:
            print(line)


#
def loadLines(filename, fileds):
    lines = {}
    with open(filename, 'r', encoding="iso-8859-1") as f:
        for line in f:
            values = line.split(' +++$+++ ')
            lineObj = {}
            for i, field in enumerate(fileds):
                lineObj[field] = values[i]
            lines[lineObj['lineID']] = lineObj
    return lines


def loadConversations(filename, lines, fields):
    conversation = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split(' +++$+++ ')
            convObj = {}
            for i, field in enumerate(fields):
                convObj[field] = values[i]
            lineIds = eval(convObj['utteranceIDs'])
            convObj["lines"] = []
            for lineId in lineIds:
                convObj["lines"].append(lines[lineId])
            conversation.append(convObj)
    return conversation


def extractSentencePairs(conversations):
    pairs = []
    for conversation in conversations:
        for i in range(len(conversation["lines"]) - 1):
            inputLine = conversation['lines'][i]["text"].strip()
            targetLine = conversation['lines'][i + 1]["text"].strip()
            if inputLine and targetLine:
                pairs.append((inputLine, targetLine))
    return pairs


if __name__ == '__main__':
    delimiter = '\t'
    lines = {}
    conversations = []
    print("\n预处理语料库(movie_lines)数据Start...")
    lines = loadLines(os.path.join(corpus, corpus_movie_lines), MOVIE_LINES_FIELDS)
    # print(lines)
    print("\n预处理语料库(movie_lines)数据Finish...")
    print("\n预处理语料库(movie_conversations)数据Start...")
    conversations = loadConversations(os.path.join(corpus, corpus_movie_conversations_lines), lines,
                                      MOVIE_CONVERSATIONS_FIELDS)
    print("\n预处理语料库(movie_conversations)数据Finish...")
    print("\n预处理格式化文件写入...")
    with open(os.path.join(corpus, FORMATTED_MOVIE_LINES), 'w', encoding='utf-8') as outFile:
        writer = csv.writer(outFile, delimiter=delimiter, lineterminator='\n')
        for pair in extractSentencePairs(conversations):
            writer.writerow(pair)
    print("\n Sample lines from file:")
    printLines(os.path.join(corpus, FORMATTED_MOVIE_LINES))
