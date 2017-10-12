#    Copyright (C) 2017 Tiancheng Zhao, Carnegie Mellon University
'''
# The top directory is a python dictionary
type(data) = dict
data.keys() = ['train', 'valid', 'test']

# Train/valid/test is a list, each element is one corpus
train = data['train']
type(train) = list

# Each corpus is a tuple
corpus = train[0]
corpus = (context, response)

# the context may be the title, the top-focus sentences, or the keywords of one article
# 1. title: [[<s>, a, b, c, ..., </s>]]
# 2. top-focus sentences: [[<s>, a1, b1, c1, ..., </s>], [<s>, a2, b2, c2, ..., </s>], ...]
# 3. keywords: [[c, a, b]]
# the response is one comment: [<s>, a, b, c, ..., </s>]
'''

import cPickle as pkl
from collections import Counter
import numpy as np
import random

class Corpus(object):
    def __init__(self, corpus_path, max_train_size=500000, max_valid_size=2000, max_test_size=500,
                 max_vocab_cnt=30000, word2vec=None, word2vec_dim=None):
        """
        :param corpus_path: the folder that contains the corpus pickle file
        """
        self._path = corpus_path
        self.word_vec_path = word2vec
        self.word2vec_dim = word2vec_dim
        self.word2vec = None
        data = pkl.load(open(self._path, "rb"))
        self.train_corpus = random.sample(data["train"], min(max_train_size, len(data["train"]))) # [(context, response), ...]
        self.valid_corpus = random.sample(data["valid"], min(max_valid_size, len(data["valid"])))
        self.test_corpus = random.sample(data["test"], min(max_test_size, len(data["test"])))
        self.build_vocab(max_vocab_cnt)
        self.load_word2vec()
        print("Done loading corpus")

    def preprocess_for_keywords2comment(self, dataset):
        """
        :param dataset: [([[keyword1, keyword2, ...]], [<s>, w1, ..., </s>])]
        :return: [([[c1, c2, ...], [c1, ...], ...], [<s>, c1, ..., </s>])]
        """
        new_dataset = []
        for context, response in dataset:
            if len(context[0]) == 0:
                continue
            new_context = [list(w) for w in context[0]]
            new_response = [list(w) for w in response[1:-1]]
            new_response = [response[0]] + reduce(lambda x, y: x + y, new_response) + [response[-1]]
            new_dataset.append((new_context, new_response))
        return new_dataset

    def build_vocab(self, max_vocab_cnt):
        all_context_words = []
        all_response_words = []
        for context, response in self.train_corpus:
            new_context = reduce(lambda x, y: x + y, context)
            all_context_words.extend(new_context)
            all_response_words.extend(response)

        def _cutoff_vocab(all_word):
            vocab_count = Counter(all_word).most_common() # word frequence
            raw_vocab_size = len(vocab_count)
            discard_wc = np.sum([c for t, c, in vocab_count[max_vocab_cnt:]])
            vocab_count = vocab_count[0:max_vocab_cnt]
            vocab = ["<pad>", "<unk>"] + [t for t, cnt in vocab_count]
            rev_vocab = {t:idx for idx, t in enumerate(vocab)}

            print("raw vocab size %d vocab size %d at cut_off %d OOV rate %f"
                  % (raw_vocab_size, len(vocab_count), vocab_count[-1][1], float(discard_wc) / len(all_word)))

            return vocab, rev_vocab

        print("Loading context vocabulary")
        self.context_vocab, self.rev_context_vocab = _cutoff_vocab(all_context_words)

        print("Loading response vocabulary")
        self.response_vocab, self.rev_response_vocab = _cutoff_vocab(all_response_words)

    def load_word2vec(self):
        pass
        # if self.word_vec_path is none:
        #     return
        # with open(self.word_vec_path, "rb") as f:
        #     lines = f.readlines()
        # raw_word2vec = {}
        # for l in lines:
        #     w, vec = l.split(" ", 1)
        #     raw_word2vec[w] = vec
        # # clean up lines for memory efficiency
        # self.word2vec = []
        # oov_cnt = 0
        # for v in self.vocab:
        #     str_vec = raw_word2vec.get(v, none)
        #     if str_vec is none:
        #         oov_cnt += 1
        #         vec = np.random.randn(self.word2vec_dim) * 0.1
        #     else:
        #         vec = np.fromstring(str_vec, sep=" ")
        #     self.word2vec.append(vec)
        # print("word2vec cannot cover %f vocab" % (float(oov_cnt)/len(self.vocab)))

    def get_corpus(self):
        def _to_id_sentence(sen, rev_vocab):
            unk_id = rev_vocab["<unk>"]
            if type(sen) is list:
                return [_to_id_sentence(sub_sen, rev_vocab) for sub_sen in sen]
            else:
                return rev_vocab.get(sen, unk_id)

        def _to_id_corpus(data):
            results = []
            for context, response in data:
                cxt_ids = _to_id_sentence(context, self.rev_context_vocab)
                res_ids = _to_id_sentence(response, self.rev_response_vocab)
                results.append((cxt_ids, res_ids))
            return results

        id_train = _to_id_corpus(self.train_corpus)
        id_valid = _to_id_corpus(self.valid_corpus)
        id_test = _to_id_corpus(self.test_corpus)
        return {'train': id_train, 'valid': id_valid, 'test': id_test}
