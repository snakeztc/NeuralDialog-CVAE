#    Copyright (C) 2017 Tiancheng Zhao, Carnegie Mellon University

import numpy as np


# Data feed
class LongDataLoader(object):
    """A special efficient data loader for TBPTT"""
    batch_size = 0
    backward_size = 0
    step_size = 0
    ptr = 0
    num_batch = None
    batch_indexes = None
    grid_indexes = None
    indexes = None
    data_lens = None
    data_size = None
    prev_alive_size = 0
    name = None

    def _shuffle_batch_indexes(self):
        np.random.shuffle(self.batch_indexes)

    def _prepare_batch(self, cur_grid, prev_grid):
        raise NotImplementedError("Have to override prepare batch")

    def epoch_init(self, batch_size, backward_size, step_size, shuffle=True, intra_shuffle=True):
        assert len(self.indexes) == self.data_size and len(self.data_lens) == self.data_size

        self.ptr = 0
        self.batch_size = batch_size
        self.backward_size = backward_size
        self.step_size = step_size
        self.prev_alive_size = batch_size

        # create batch indexes
        temp_num_batch = self.data_size // batch_size
        self.batch_indexes = []
        for i in range(temp_num_batch):
            self.batch_indexes.append(self.indexes[i * self.batch_size:(i + 1) * self.batch_size])

        left_over = self.data_size-temp_num_batch*batch_size

        # shuffle batch indexes
        if shuffle:
            self._shuffle_batch_indexes()

        # create grid indexes
        self.grid_indexes = []
        for idx, b_ids in enumerate(self.batch_indexes):
            # assume the b_ids are sorted
            all_lens = [self.data_lens[i] for i in b_ids]
            max_len = self.data_lens[b_ids[-1]]
            min_len = self.data_lens[b_ids[0]]
            assert np.max(all_lens) == max_len
            assert np.min(all_lens) == min_len
            num_seg = (max_len-self.backward_size) // self.step_size
            if num_seg > 0:
                cut_start = range(0, num_seg*self.step_size, step_size)
                cut_end = range(self.backward_size, num_seg*self.step_size+self.backward_size, step_size)
                assert cut_end[-1] < max_len
                cut_start = [0] * (self.backward_size-2) +cut_start # since we give up on the seq training idea
                cut_end = range(2, self.backward_size) + cut_end
            else:
                cut_start = [0] * (max_len-2)
                cut_end = range(2, max_len)

            new_grids = [(idx, s_id, e_id) for s_id, e_id in zip(cut_start, cut_end) if s_id < min_len-1]
            if intra_shuffle and shuffle:
               np.random.shuffle(new_grids)
            self.grid_indexes.extend(new_grids)

        self.num_batch = len(self.grid_indexes)
        print("%s begins with %d batches with %d left over samples" % (self.name, self.num_batch, left_over))

    def next_batch(self):
        if self.ptr < self.num_batch:
            current_grid = self.grid_indexes[self.ptr]
            if self.ptr > 0:
                prev_grid = self.grid_indexes[self.ptr-1]
            else:
                prev_grid = None
            self.ptr += 1
            return self._prepare_batch(cur_grid=current_grid, prev_grid=prev_grid)
        else:
            return None


class SWDADataLoader(LongDataLoader):
    def __init__(self, name, data, meta_data, config):
        assert len(data) == len(meta_data)
        self.name = name
        self.data = data
        self.meta_data = meta_data
        self.data_size = len(data)
        self.data_lens = all_lens = [len(line) for line in self.data]
        self.max_utt_size = config.max_utt_len
        print("Max len %d and min len %d and avg len %f" % (np.max(all_lens),
                                                            np.min(all_lens),
                                                            float(np.mean(all_lens))))
        self.indexes = list(np.argsort(all_lens))

    def pad_to(self, tokens, do_pad=True):
        if len(tokens) >= self.max_utt_size:
            return tokens[0:self.max_utt_size-1] + [tokens[-1]]
        elif do_pad:
            return tokens + [0] * (self.max_utt_size-len(tokens))
        else:
            return tokens

    def _prepare_batch(self, cur_grid, prev_grid):
        # the batch index, the starting point and end point for segment
        b_id, s_id, e_id = cur_grid

        batch_ids = self.batch_indexes[b_id]
        rows = [self.data[idx] for idx in batch_ids]
        meta_rows = [self.meta_data[idx] for idx in batch_ids]
        dialog_lens = [self.data_lens[idx] for idx in batch_ids]

        topics = np.array([meta[2] for meta in meta_rows])
        cur_pos = [np.minimum(1.0, e_id/float(l)) for l in dialog_lens]

        # input_context, context_lens, floors, topics, a_profiles, b_Profiles, outputs, output_lens
        context_lens, context_utts, floors, out_utts, out_lens, out_floors, out_das = [], [], [], [], [], [], []
        for row in rows:
            if s_id < len(row)-1:
                cut_row = row[s_id:e_id]
                in_row = cut_row[0:-1]
                out_row = cut_row[-1]
                out_utt, out_floor, out_feat = out_row

                context_utts.append([self.pad_to(utt) for utt, floor, feat in in_row])
                floors.append([int(floor==out_floor) for utt, floor, feat in in_row])
                context_lens.append(len(cut_row) - 1)

                out_utt = self.pad_to(out_utt, do_pad=False)
                out_utts.append(out_utt)
                out_lens.append(len(out_utt))
                out_floors.append(out_floor)
                out_das.append(out_feat[0])
            else:
                print(row)
                raise ValueError("S_ID %d larger than row" % s_id)

        # my_profiles = np.array([meta[out_floors[idx]] + [cur_pos[idx]] for idx, meta in enumerate(meta_rows)])
        my_profiles = np.array([meta[out_floors[idx]] for idx, meta in enumerate(meta_rows)])
        ot_profiles = np.array([meta[1-out_floors[idx]] for idx, meta in enumerate(meta_rows)])
        vec_context_lens = np.array(context_lens)
        vec_context = np.zeros((self.batch_size, np.max(vec_context_lens), self.max_utt_size), dtype=np.int32)
        vec_floors = np.zeros((self.batch_size, np.max(vec_context_lens)), dtype=np.int32)
        vec_outs = np.zeros((self.batch_size, np.max(out_lens)), dtype=np.int32)
        vec_out_lens = np.array(out_lens)
        vec_out_das = np.array(out_das)

        for b_id in range(self.batch_size):
            vec_outs[b_id, 0:vec_out_lens[b_id]] = out_utts[b_id]
            vec_floors[b_id, 0:vec_context_lens[b_id]] = floors[b_id]
            vec_context[b_id, 0:vec_context_lens[b_id], :] = np.array(context_utts[b_id])

        return vec_context, vec_context_lens, vec_floors, topics, my_profiles, ot_profiles, vec_outs, vec_out_lens, vec_out_das








