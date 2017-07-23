#    Copyright (C) 2017 Tiancheng Zhao, Carnegie Mellon University
import pickle as pkl
from collections import Counter
import numpy as np
import nltk


class SWDADialogCorpus(object):
    dialog_act_id = 0
    sentiment_id = 1
    liwc_id = 2

    def __init__(self, corpus_path, max_vocab_cnt=10000, word2vec=None, word2vec_dim=None):
        """
        :param corpus_path: the folder that contains the SWDA dialog corpus
        """
        self._path = corpus_path
        self.word_vec_path = word2vec
        self.word2vec_dim = word2vec_dim
        self.word2vec = None
        self.dialog_id = 0
        self.meta_id = 1
        self.utt_id = 2
        self.sil_utt = ["<s>", "<sil>", "</s>"]
        data = pkl.load(open(self._path, "rb"))
        self.train_corpus = self.process(data["train"])
        self.valid_corpus = self.process(data["valid"])
        self.test_corpus = self.process(data["test"])
        self.build_vocab(max_vocab_cnt)
        self.load_word2vec()
        print("Done loading corpus")

    def process(self, data):
        """new_dialog: [(a, 1/0), (a,1/0)], new_meta: (a, b, topic), new_utt: [[a,b,c)"""
        """ 1 is own utt and 0 is other's utt"""
        new_dialog = []
        new_meta = []
        new_utts = []
        bod_utt = ["<s>", "<d>", "</s>"]
        all_lenes = []

        for l in data:
            lower_utts = [(caller, ["<s>"] + nltk.WordPunctTokenizer().tokenize(utt.lower()) + ["</s>"], feat)
                          for caller, utt, feat in l["utts"]]
            all_lenes.extend([len(u) for c, u, f in lower_utts])

            a_age = float(l["A"]["age"])/100.0
            b_age = float(l["B"]["age"])/100.0
            a_edu = float(l["A"]["education"])/3.0
            b_edu = float(l["B"]["education"])/3.0
            vec_a_meta = [a_age, a_edu] + ([0, 1] if l["A"]["sex"] == "FEMALE" else [1, 0])
            vec_b_meta = [b_age, b_edu] + ([0, 1] if l["B"]["sex"] == "FEMALE" else [1, 0])

            # for joint model we mode two side of speakers together. if A then its 0 other wise 1
            meta = (vec_a_meta, vec_b_meta, l["topic"])
            dialog = [(bod_utt, 0, None)] + [(utt, int(caller=="B"), feat) for caller, utt, feat in lower_utts]

            new_utts.extend([bod_utt] + [utt for caller, utt, feat in lower_utts])
            new_dialog.append(dialog)
            new_meta.append(meta)

        print("Max utt len %d, mean utt len %.2f" % (np.max(all_lenes), float(np.mean(all_lenes))))
        return new_dialog, new_meta, new_utts

    def build_vocab(self, max_vocab_cnt):
        all_words = []
        for tokens in self.train_corpus[self.utt_id]:
            all_words.extend(tokens)
        vocab_count = Counter(all_words).most_common()
        raw_vocab_size = len(vocab_count)
        discard_wc = np.sum([c for t, c, in vocab_count[max_vocab_cnt:]])
        vocab_count = vocab_count[0:max_vocab_cnt]

        # create vocabulary list sorted by count
        print("Load corpus with train size %d, valid size %d, "
              "test size %d raw vocab size %d vocab size %d at cut_off %d OOV rate %f"
              % (len(self.train_corpus), len(self.valid_corpus), len(self.test_corpus),
                 raw_vocab_size, len(vocab_count), vocab_count[-1][1], float(discard_wc) / len(all_words)))

        self.vocab = ["<pad>", "<unk>"] + [t for t, cnt in vocab_count]
        self.rev_vocab = {t: idx for idx, t in enumerate(self.vocab)}
        self.unk_id = self.rev_vocab["<unk>"]
        print("<d> index %d" % self.rev_vocab["<d>"])
        print("<sil> index %d" % self.rev_vocab.get("<sil>", -1))

        # create topic vocab
        all_topics = []
        for a, b, topic in self.train_corpus[self.meta_id]:
            all_topics.append(topic)
        self.topic_vocab = [t for t, cnt in Counter(all_topics).most_common()]
        self.rev_topic_vocab = {t: idx for idx, t in enumerate(self.topic_vocab)}
        print("%d topics in train data" % len(self.topic_vocab))

        # get dialog act labels
        all_dialog_acts = []
        for dialog in self.train_corpus[self.dialog_id]:
            all_dialog_acts.extend([feat[self.dialog_act_id] for caller, utt, feat in dialog if feat is not None])
        self.dialog_act_vocab = [t for t, cnt in Counter(all_dialog_acts).most_common()]
        self.rev_dialog_act_vocab = {t: idx for idx, t in enumerate(self.dialog_act_vocab)}
        print(self.dialog_act_vocab)
        print("%d dialog acts in train data" % len(self.dialog_act_vocab))

    def load_word2vec(self):
        if self.word_vec_path is None:
            return
        with open(self.word_vec_path, "rb") as f:
            lines = f.readlines()
        raw_word2vec = {}
        for l in lines:
            w, vec = l.split(" ", 1)
            raw_word2vec[w] = vec
        # clean up lines for memory efficiency
        self.word2vec = []
        oov_cnt = 0
        for v in self.vocab:
            str_vec = raw_word2vec.get(v, None)
            if str_vec is None:
                oov_cnt += 1
                vec = np.random.randn(self.word2vec_dim) * 0.1
            else:
                vec = np.fromstring(str_vec, sep=" ")
            self.word2vec.append(vec)
        print("word2vec cannot cover %f vocab" % (float(oov_cnt)/len(self.vocab)))

    def get_utt_corpus(self):
        def _to_id_corpus(data):
            results = []
            for line in data:
                results.append([self.rev_vocab.get(t, self.unk_id) for t in line])
            return results
        # convert the corpus into ID
        id_train = _to_id_corpus(self.train_corpus[self.utt_id])
        id_valid = _to_id_corpus(self.valid_corpus[self.utt_id])
        id_test = _to_id_corpus(self.test_corpus[self.utt_id])
        return {'train': id_train, 'valid': id_valid, 'test': id_test}

    def get_dialog_corpus(self):
        def _to_id_corpus(data):
            results = []
            for dialog in data:
                temp = []
                # convert utterance and feature into numeric numbers
                for utt, floor, feat in dialog:
                    if feat is not None:
                        id_feat = list(feat)
                        id_feat[self.dialog_act_id] = self.rev_dialog_act_vocab[feat[self.dialog_act_id]]
                    else:
                        id_feat = None
                    temp.append(([self.rev_vocab.get(t, self.unk_id) for t in utt], floor, id_feat))
                results.append(temp)
            return results
        id_train = _to_id_corpus(self.train_corpus[self.dialog_id])
        id_valid = _to_id_corpus(self.valid_corpus[self.dialog_id])
        id_test = _to_id_corpus(self.test_corpus[self.dialog_id])
        return {'train': id_train, 'valid': id_valid, 'test': id_test}

    def get_meta_corpus(self):
        def _to_id_corpus(data):
            results = []
            for m_meta, o_meta, topic in data:
                results.append((m_meta, o_meta, self.rev_topic_vocab[topic]))
            return results

        id_train = _to_id_corpus(self.train_corpus[self.meta_id])
        id_valid = _to_id_corpus(self.valid_corpus[self.meta_id])
        id_test = _to_id_corpus(self.test_corpus[self.meta_id])
        return {'train': id_train, 'valid': id_valid, 'test': id_test}

