"""
    1.在自然语言处理中常常使用预训练的word2vec，这个预训练的词向量可以使用google的GoogleNews-vectors-negative300.bin
    2.GoogleNews-vectors-negative300.bin是训练好的300维的新闻语料词向量
    3.本函数的作用就是把一个词转换成词向量，以供我们后期使用。没有在该word2vec中的词采用其他的方式构建，如采用均匀分布或者高斯分布等随机初始化
"""
import numpy as np


# loads 300x1 word vectors from file.
def load_bin_vec(fname, vocab):
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split()) # 3000000 300
        binary_len = np.dtype('float32').itemsize * layer1_size # 1200
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return word_vecs


# add random vectors of unknown words which are not in pre-trained vector file.
# if pre-trained vectors are not used, then initialize all words in vocab with random value.
def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, k)


vectors_file = 'E:\GoogleNews-vectors-negative300.bin\GoogleNews-vectors-negative300.bin'
vocab = ['I', 'can', 'do']

vectors = load_bin_vec(vectors_file, vocab)  # pre-trained vectors
add_unknown_words(vectors, vocab)
print(vectors['I'])
print('*'*40)
print(vectors['can'])
print('*'*40)
print(vectors['do'])