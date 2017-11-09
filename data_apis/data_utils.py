#    Copyright (C) 2017 Tiancheng Zhao, Carnegie Mellon University

import numpy as np


# Data feed
class LongDataLoader(object):
    """A special efficient data loader for TBPTT"""
    batch_size = 0
    ptr = 0
    num_batch = None
    batch_indexes = None
    indexes = None
    data_size = None
    name = None

    def _shuffle_indexes(self):
        np.random.shuffle(self.indexes)

    def _prepare_batch(self, batch_idx):
        raise NotImplementedError("Have to override prepare batch")

    def epoch_init(self, batch_size, shuffle=True):
        self.ptr = 0

        if shuffle:
            self._shuffle_indexes()

        # create batch indexes
        self.batch_size = batch_size
        temp_num_batch = self.data_size // batch_size
        self.batch_indexes = []
        for i in range(temp_num_batch):
            self.batch_indexes.append(self.indexes[i * self.batch_size:(i + 1) * self.batch_size])

        left_over = self.data_size-temp_num_batch*batch_size

        self.num_batch = len(self.batch_indexes)

        print("%s begins with %d batches with %d left over samples" % (self.name, self.num_batch, left_over))

    def next_batch(self):
        if self.ptr < self.num_batch:
            cur_batch = self._prepare_batch(self.ptr)
            self.ptr += 1
            return cur_batch
        else:
            return None


class DataLoader(LongDataLoader):
    def __init__(self, name, data, config):
        self.name = name
        self.data = data
        self.data_size = len(data)
        self.max_utt_size = config.max_utt_len
        self.indexes = np.arange(0, self.data_size)

    def pad_to(self, tokens, do_pad=True):
        if len(tokens) >= self.max_utt_size:
            return tokens[0:self.max_utt_size-1] + [tokens[-1]]
        elif do_pad:
            return tokens + [0] * (self.max_utt_size-len(tokens))
        else:
            return tokens

    def _prepare_batch(self, batch_idx):
        batch_ids = self.batch_indexes[batch_idx]
        rows = [self.data[idx] for idx in batch_ids] #[(context, response), ...]

        # input_context, context_lens, outputs, output_lens
        titles, title_lens, context_utts, context_lens, out_utts, out_lens, out_topics = [], [], [], [], [], [], []
        for context, response, topic in rows:
            titles.append(self.pad_to(context[0]))
            title_lens.append(len(context[0]))

            context_utts.append([self.pad_to(sen) for sen in context[1:]])
            context_lens.append(len(context) - 1)

            out_utt = self.pad_to(response, do_pad=False)
            out_utts.append(out_utt)
            out_lens.append(len(out_utt))

            out_topics.append(topic)

        vec_title_lens = np.array(title_lens)
        vec_titles = np.zeros((self.batch_size, self.max_utt_size), dtype=np.int32)
        vec_context_lens = np.array(context_lens)
        vec_contexts = np.zeros((self.batch_size, np.max(vec_context_lens), self.max_utt_size), dtype=np.int32)
        vec_out_lens = np.array(out_lens)
        vec_outs = np.zeros((self.batch_size, np.max(out_lens)), dtype=np.int32)
        vec_out_topics = np.array(out_topics)

        for b_id in range(self.batch_size):
            vec_outs[b_id, 0:vec_out_lens[b_id]] = out_utts[b_id]
            vec_contexts[b_id, 0:vec_context_lens[b_id], :] = np.array(context_utts[b_id])
            vec_titles[b_id, :] = titles[b_id]

        return vec_titles, vec_title_lens, vec_contexts, vec_context_lens, vec_outs, vec_out_lens, vec_out_topics
