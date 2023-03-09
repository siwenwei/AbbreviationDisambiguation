import os.path

from gensim.models import Word2Vec
from gensim.models import KeyedVectors


model = KeyedVectors.load_word2vec_format(os.path.join('E:/phrase2word/phrase2word'
                                          '/pubmed_phrase_word_combo_skip_model_vectors_win_5.bin'), binary=True)
# vec = model['diabetes']
paris = [
    ('minor', 'below-knee'),
    ('minor', 'below-the-knee'),
    ('minor', 'large-conductance')
]
for w1, w2 in paris:
    print('%r\t%r\t%.2f' %(w1, w2, model.similarity(w1, w2)),)

# word = model.most_similar('diabetes', topn=1)
# print(word)



