import gensim
import numpy as np
from gensim import utils
import gensim.models.doc2vec
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from sklearn.model_selection import train_test_split

with utils.open('./data/pos.txt', 'r', encoding='utf-8') as infile:
    pos_reviews = []
    line = infile.readline()
    while line:
        pos_reviews.append(line)
        line = infile.readline()
with utils.open('./data/neg.txt', 'r', encoding='utf-8') as infile:
    neg_reviews = []
    line = infile.readline()
    while line:
        neg_reviews.append(line)
        line = infile.readline()
with utils.open('./data/unsup.txt', 'r', encoding='utf-8') as infile:
    unsup_reviews = []
    line = infile.readline()
    while line:
        unsup_reviews.append(line)
        line = infile.readline()

y = np.concatenate((np.ones(len(pos_reviews)), np.zeros(len(neg_reviews))))

x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos_reviews, neg_reviews)), y, test_size=0.2)


def labelizeReviews(reviews, label_type):
    for i, v in enumerate(reviews):
        label = '%s_%s' % (label_type, i)
        yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(v, max_len=100), [label])


x_train_tag = list(labelizeReviews(x_train, 'train'))
x_test_tag = list(labelizeReviews(x_test, 'test'))
unsup_reviews_tag = list(labelizeReviews(unsup_reviews, 'unsup'))

size = 128
model_dm = gensim.models.Doc2Vec(min_count=1, window=8, vector_size=size, sample=1e-3, negative=5, workers=3, epochs=10)
model_dbow = gensim.models.Doc2Vec(min_count=1, window=8, vector_size=size, sample=1e-3, negative=5, dm=0, workers=3,
                                   epochs=10)

alldata = x_train_tag
alldata.extend(x_test_tag)
alldata.extend(unsup_reviews_tag)

model_dm.build_vocab(alldata)
model_dbow.build_vocab(alldata)


def sentences_perm(sentences):
    import random
    shuffled = list(sentences)
    random.shuffle(shuffled)
    return shuffled


for epoch in range(10):
    print('epoch: {}'.format(epoch))
    model_dm.train(sentences_perm(alldata), total_examples=model_dm.corpus_count, epochs=1)
    model_dbow.train(sentences_perm(alldata), total_examples=model_dm.corpus_count, epochs=1)

# 第一种方法
train_arrays_dm = np.zeros((len(x_train), 128))
train_arrays_dbow = np.zeros((len(x_train), 128))
for i in range(len(x_train)):

    tag = 'train_' + str(i)
    train_arrays_dm[i] = model_dm.docvecs[tag]
    train_arrays_dbow[i] = model_dbow.docvecs[tag]
train_arrays = np.hstack((train_arrays_dm, train_arrays_dbow))
test_arrays_dm = np.zeros((len(x_test), 128))
test_arrays_dbow = np.zeros((len(x_test), 128))
for i in range(len(x_test)):
    tag = 'test_' + str(i)
    test_arrays_dm[i] = model_dm.docvecs[tag]
    test_arrays_dbow[i] = model_dbow.docvecs[tag]
test_arrays = np.hstack((test_arrays_dm, test_arrays_dbow))
# 第二种def getVecs(model, corpus):
# vecs = []
# for i in corpus:
#     vec = model.infer_vector(gensim.utils.simple_preprocess(i,max_len=300))
#     vecs.append(vec)
#     return vecs
# train_vecs_dm = getVecs(model_dm, x_train)
# train_vecs_dbow = getVecs(model_dbow, x_train)
# train_vecs = np.hstack((train_vecs_dm, train_vecs_dbow))


classifier = LogisticRegression()
classifier.fit(train_arrays, y_train)

LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1, penalty='l2',
                   random_state=None, tol=0.0001)

log.info(classifier.score(test_arrays, y_test))
y_prob = classifier.predict_proba(test_arrays)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='area = %.2f' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc='lower right')
