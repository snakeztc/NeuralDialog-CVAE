#    Copyright (C) 2017 Tiancheng Zhao, Carnegie Mellon University

import os
import re
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.rnn import OutputProjectionWrapper
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import rnn_cell_impl as rnn_cell
from tensorflow.python.ops import variable_scope
from tensorflow.python.layers import core as layers_core

import decoder_fn_lib
import utils
from models.seq2seq import dynamic_rnn_decoder
from utils import gaussian_kld
from utils import get_bi_rnn_encode
from utils import get_bow
from utils import get_rnn_encode
from utils import norm_log_liklihood
from utils import sample_gaussian


class BaseTFModel(object):
    global_t = tf.placeholder(dtype=tf.int32, name="global_t")
    learning_rate = None
    scope = None

    @staticmethod
    def print_model_stats(tvars):
        total_parameters = 0
        for variable in tvars:
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parametes = 1
            for dim in shape:
                variable_parametes *= dim.value
            print("Trainable %s with %d parameters" % (variable.name, variable_parametes))
            total_parameters += variable_parametes
        print("Total number of trainable parameters is %d" % total_parameters)

    @staticmethod
    def get_rnncell(cell_type, cell_size, keep_prob, num_layer):
        if cell_type == "gru":
            cell = rnn_cell.GRUCell(cell_size)
        else:
            cell = rnn_cell.LSTMCell(cell_size, use_peepholes=False, forget_bias=1.0)

        if keep_prob < 1.0:
            cell = rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)

        if num_layer > 1:
            cell = rnn_cell.MultiRNNCell([cell] * num_layer, state_is_tuple=True)

        return cell

    @staticmethod
    def print_loss(prefix, loss_names, losses, postfix):
        template = "%s "
        for name in loss_names:
            template += "%s " % name
            template += " %f "
        template += "%s"
        template = re.sub(' +', ' ', template)
        avg_losses = []
        values = [prefix]

        for loss in losses:
            values.append(np.mean(loss))
            avg_losses.append(np.mean(loss))
        values.append(postfix)

        print(template % tuple(values))
        return avg_losses

    def train(self, global_t, sess, train_feed):
        raise NotImplementedError("Train function needs to be implemented")

    def valid(self, *args, **kwargs):
        raise NotImplementedError("Valid function needs to be implemented")

    def batch_2_feed(self, *args, **kwargs):
        raise NotImplementedError("Implement how to unpack the back")

    def optimize(self, sess, config, loss, log_dir):
        if log_dir is None:
            return
        # optimization
        if self.scope is None:
            tvars = tf.trainable_variables()
        else:
            tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)
        grads = tf.gradients(loss, tvars)
        if config.grad_clip is not None:
            grads, _ = tf.clip_by_global_norm(grads, tf.constant(config.grad_clip))
        # add gradient noise
        if config.grad_noise > 0:
            grad_std = tf.sqrt(config.grad_noise / tf.pow(1.0 + tf.to_float(self.global_t), 0.55))
            grads = [g + tf.truncated_normal(tf.shape(g), mean=0.0, stddev=grad_std) for g in grads]

        if config.op == "adam":
            print("Use Adam")
            optimizer = tf.train.AdamOptimizer(config.init_lr)
        elif config.op == "rmsprop":
            print("Use RMSProp")
            optimizer = tf.train.RMSPropOptimizer(config.init_lr)
        else:
            print("Use SGD")
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

        self.train_ops = optimizer.apply_gradients(zip(grads, tvars))
        self.print_model_stats(tvars)
        train_log_dir = os.path.join(log_dir, "checkpoints")
        print("Save summary to %s" % log_dir)
        self.train_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)


class RnnCVAE(BaseTFModel):

    def __init__(self, sess, config, api, log_dir, forward, scope=None):
        self.vocab = api.vocab
        self.rev_vocab = api.rev_vocab
        self.vocab_size = len(self.vocab)
        self.topic_vocab = api.topic_vocab
        self.topic_vocab_size = len(self.topic_vocab)

        self.sess = sess
        self.scope = scope
        self.max_utt_len = config.max_utt_len
        self.go_id = self.rev_vocab["<s>"]
        self.eos_id = self.rev_vocab["</s>"]
        self.context_cell_size = config.cxt_cell_size
        self.sent_cell_size = config.sent_cell_size
        self.dec_cell_size = config.dec_cell_size
        self.bow_weights = config.bow_weights

        with tf.name_scope("io"):
            # all dialog context and known attributes
            self.input_contexts = tf.placeholder(dtype=tf.int32, shape=(None, None, self.max_utt_len), name="context")
            self.context_lens = tf.placeholder(dtype=tf.int32, shape=(None,), name="context_lens")

            # target response given the dialog context
            self.output_tokens = tf.placeholder(dtype=tf.int32, shape=(None, None), name="output_token")
            self.output_lens = tf.placeholder(dtype=tf.int32, shape=(None,), name="output_lens")
            self.output_topics = tf.placeholder(dtype=tf.int32, shape=(None,), name="output_topic")

            # optimization related variables
            self.learning_rate = tf.Variable(float(config.init_lr), trainable=False, name="learning_rate")
            self.learning_rate_decay_op = self.learning_rate.assign(tf.multiply(self.learning_rate, config.lr_decay))
            self.global_t = tf.placeholder(dtype=tf.int32, name="global_t")
            self.use_prior = tf.placeholder(dtype=tf.bool, name="use_prior")

        max_context_len = array_ops.shape(self.input_contexts)[1]
        max_out_len = array_ops.shape(self.output_tokens)[1]
        batch_size = array_ops.shape(self.input_contexts)[0]

        if config.use_hcf:
            with variable_scope.variable_scope("topicEmbedding"):
                t_embedding = tf.get_variable("embedding", [self.topic_vocab_size, config.topic_embed_size], dtype=tf.float32)
                topic_embedding = embedding_ops.embedding_lookup(t_embedding, self.output_topics)

        with variable_scope.variable_scope("wordEmbedding"):
            self.embedding = tf.get_variable("embedding", [self.vocab_size, config.embed_size], dtype=tf.float32)
            embedding_mask = tf.constant([0 if i == 0 else 1 for i in range(self.vocab_size)], dtype=tf.float32,
                                         shape=[self.vocab_size, 1])
            embedding = self.embedding * embedding_mask

            input_embedding = embedding_ops.embedding_lookup(embedding, tf.reshape(self.input_contexts, [-1]))
            input_embedding = tf.reshape(input_embedding, [-1, self.max_utt_len, config.embed_size])
            output_embedding = embedding_ops.embedding_lookup(embedding, self.output_tokens)

            # context nn
            if config.sent_type == "bow":
                input_embedding, sent_size = get_bow(input_embedding)
                output_embedding, _ = get_bow(output_embedding)

            elif config.sent_type == "rnn":
                sent_cell = self.get_rnncell("gru", self.sent_cell_size, config.keep_prob, 1)
                input_embedding, sent_size = get_rnn_encode(input_embedding, sent_cell, scope="sent_rnn")
                output_embedding, _ = get_rnn_encode(output_embedding, sent_cell, self.output_lens,
                                                     scope="sent_rnn", reuse=True)
            elif config.sent_type == "bi_rnn":
                fwd_sent_cell = self.get_rnncell("gru", self.sent_cell_size, keep_prob=1.0, num_layer=1)
                bwd_sent_cell = self.get_rnncell("gru", self.sent_cell_size, keep_prob=1.0, num_layer=1)
                input_embedding, sent_size = get_bi_rnn_encode(input_embedding, fwd_sent_cell, bwd_sent_cell, scope="sent_bi_rnn")
                output_embedding, _ = get_bi_rnn_encode(output_embedding, fwd_sent_cell, bwd_sent_cell, self.output_lens, scope="sent_bi_rnn", reuse=True)
            else:
                raise ValueError("Unknown sent_type. Must be one of [bow, rnn, bi_rnn]")

            # reshape input into dialogs
            input_embedding = tf.reshape(input_embedding, [-1, max_context_len, sent_size])
            if config.keep_prob < 1.0:
                input_embedding = tf.nn.dropout(input_embedding, config.keep_prob)

        with variable_scope.variable_scope("contextRNN"):
            enc_cell = self.get_rnncell(config.cell_type, self.context_cell_size, keep_prob=1.0, num_layer=config.num_layer)
            # and enc_last_state will be same as the true last state
            encoder_outputs, enc_last_state = tf.nn.dynamic_rnn(
                enc_cell,
                input_embedding,
                dtype=tf.float32,
                sequence_length=self.context_lens)

            if config.num_layer > 1:
                enc_last_state = tf.concat(enc_last_state, 1)

        # combine with other attributes
        if config.use_hcf:
            attribute_embedding = topic_embedding
            attribute_fc1 = layers.fully_connected(attribute_embedding, 30, activation_fn=tf.tanh, scope="attribute_fc1")

        cond_embedding = enc_last_state

        with variable_scope.variable_scope("recognitionNetwork"):
            if config.use_hcf:
                recog_input = tf.concat([cond_embedding, output_embedding, attribute_fc1], 1)
            else:
                recog_input = tf.concat([cond_embedding, output_embedding], 1)
            self.recog_mulogvar = recog_mulogvar = layers.fully_connected(recog_input, config.latent_size * 2, activation_fn=None, scope="muvar")
            recog_mu, recog_logvar = tf.split(recog_mulogvar, 2, axis=1)

        with variable_scope.variable_scope("priorNetwork"):
            prior_fc1 = layers.fully_connected(cond_embedding, np.maximum(config.latent_size * 2, 100),
                                               activation_fn=tf.tanh, scope="fc1")
            prior_mulogvar = layers.fully_connected(prior_fc1, config.latent_size * 2, activation_fn=None,
                                                    scope="muvar")
            prior_mu, prior_logvar = tf.split(prior_mulogvar, 2, axis=1)

            # use sampled Z or posterior Z
            latent_sample = tf.cond(self.use_prior,
                                    lambda: sample_gaussian(prior_mu, prior_logvar),
                                    lambda: sample_gaussian(recog_mu, recog_logvar))

        with variable_scope.variable_scope("generationNetwork"):
            gen_inputs = tf.concat([cond_embedding, latent_sample], 1)

            # BOW loss
            bow_fc1 = layers.fully_connected(gen_inputs, 400, activation_fn=tf.tanh, scope="bow_fc1")
            if config.keep_prob < 1.0:
                bow_fc1 = tf.nn.dropout(bow_fc1, config.keep_prob)
            self.bow_logits = layers.fully_connected(bow_fc1, self.vocab_size, activation_fn=None, scope="bow_project")

            # Y loss
            if config.use_hcf:
                meta_fc1 = layers.fully_connected(latent_sample, 400, activation_fn=tf.tanh, scope="meta_fc1")
                if config.keep_prob <1.0:
                    meta_fc1 = tf.nn.dropout(meta_fc1, config.keep_prob)
                self.topic_logits = layers.fully_connected(meta_fc1, self.topic_vocab_size, scope="topic_project")
                topic_prob = tf.nn.softmax(self.topic_logits)
                pred_attribute_embedding = tf.matmul(topic_prob, t_embedding)
                if forward:
                    selected_attribute_embedding = pred_attribute_embedding
                else:
                    selected_attribute_embedding = attribute_embedding
                dec_inputs = tf.concat([gen_inputs, selected_attribute_embedding], 1)
            else:
                self.topic_logits = tf.zeros((batch_size, self.topic_vocab_size))
                selected_attribute_embedding = None
                dec_inputs = gen_inputs

            # Decoder
            if config.num_layer > 1:
                dec_init_state = [layers.fully_connected(dec_inputs, self.dec_cell_size, activation_fn=None,
                                                        scope="init_state-%d" % i) for i in range(config.num_layer)]
                dec_init_state = tuple(dec_init_state)
            else:
                dec_init_state = layers.fully_connected(dec_inputs, self.dec_cell_size, activation_fn=None, scope="init_state")

        with variable_scope.variable_scope("decoder"):
            # Create an attention mechanism
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                self.dec_cell_size, encoder_outputs,
                memory_sequence_length=self.context_lens)
            dec_cell = self.get_rnncell(config.cell_type, self.dec_cell_size, config.keep_prob, config.num_layer)
            dec_cell = tf.contrib.seq2seq.AttentionWrapper(
                dec_cell, attention_mechanism)

            projection_layer = layers_core.Dense(self.vocab_size, use_bias=False)

            if forward:
                # Helper
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    embedding,
                    tf.fill([batch_size], self.go_id), self.eos_id)

                # Decoder
                decoder = tf.contrib.seq2seq.BasicDecoder(
                    dec_cell, helper, dec_init_state,
                    output_layer=projection_layer)

                # Dynamic decoding
                outputs, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder, maximum_iterations=self.max_utt_len)
            else:
                # loop_func = decoder_fn_lib.context_decoder_fn_train(dec_init_state, selected_attribute_embedding)
                dec_input_embedding = embedding_ops.embedding_lookup(embedding, self.output_tokens)
                dec_input_embedding = dec_input_embedding[:, 0:-1, :]
                dec_seq_lens = self.output_lens - 1

                if config.keep_prob < 1.0:
                    dec_input_embedding = tf.nn.dropout(dec_input_embedding, config.keep_prob)

                # apply word dropping. Set dropped word to 0
                if config.dec_keep_prob < 1.0:
                    keep_mask = tf.less_equal(tf.random_uniform((batch_size, max_out_len-1), minval=0.0, maxval=1.0),
                                              config.dec_keep_prob)
                    keep_mask = tf.expand_dims(tf.to_float(keep_mask), 2)
                    dec_input_embedding = dec_input_embedding * keep_mask
                    dec_input_embedding = tf.reshape(dec_input_embedding, [-1, max_out_len-1, config.embed_size])

                helper = tf.contrib.seq2seq.TrainingHelper(dec_input_embedding, dec_seq_lens)

                decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, helper, dec_init_state,
                                                          output_layer=projection_layer)

                outputs, _ = tf.contrib.seq2seq.dynamic_decode(decoder)

            dec_outs = outputs.rnn_output
            self.dec_out_words = outputs.sample_id

        if not forward:
            with variable_scope.variable_scope("loss"):
                labels = self.output_tokens[:, 1:]
                label_mask = tf.to_float(tf.sign(labels))

                rc_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=dec_outs, labels=labels)
                rc_loss = tf.reduce_sum(rc_loss * label_mask, reduction_indices=1)
                self.avg_rc_loss = tf.reduce_mean(rc_loss)
                # used only for perpliexty calculation. Not used for optimzation
                self.rc_ppl = tf.exp(tf.reduce_sum(rc_loss) / tf.reduce_sum(label_mask))

                """ as n-trial multimodal distribution. """
                tile_bow_logits = tf.tile(tf.expand_dims(self.bow_logits, 1), [1, max_out_len - 1, 1])
                bow_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tile_bow_logits, labels=labels) * label_mask
                bow_loss = tf.reduce_sum(bow_loss, reduction_indices=1)
                self.avg_bow_loss  = tf.reduce_mean(bow_loss)
                bow_weights = tf.to_float(self.bow_weights)

                # reconstruct the meta info about X
                if config.use_hcf:
                    topic_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.topic_logits, labels=self.output_topics)
                    self.avg_topic_loss = tf.reduce_mean(topic_loss)
                else:
                    self.avg_topic_loss = 0.0

                kld = gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar)
                self.avg_kld = tf.reduce_mean(kld)
                if log_dir is not None:
                    kl_weights = tf.minimum(tf.to_float(self.global_t)/config.full_kl_step, 1.0)
                else:
                    kl_weights = tf.constant(1.0)

                self.kl_w = kl_weights
                self.elbo = self.avg_rc_loss + kl_weights * self.avg_kld
                aug_elbo = bow_weights * self.avg_bow_loss + self.avg_topic_loss + self.elbo

                tf.summary.scalar("topic_loss", self.avg_topic_loss)
                tf.summary.scalar("rc_loss", self.avg_rc_loss)
                tf.summary.scalar("elbo", self.elbo)
                tf.summary.scalar("kld", self.avg_kld)
                tf.summary.scalar("bow_loss", self.avg_bow_loss)

                self.summary_op = tf.summary.merge_all()

                self.log_p_z = norm_log_liklihood(latent_sample, prior_mu, prior_logvar)
                self.log_q_z_xy = norm_log_liklihood(latent_sample, recog_mu, recog_logvar)
                self.est_marginal = tf.reduce_mean(rc_loss + bow_loss - self.log_p_z + self.log_q_z_xy)

            self.optimize(sess, config, aug_elbo, log_dir)

        self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2)

    def batch_2_feed(self, batch, global_t, use_prior, repeat=1):
        context, context_lens, outputs, output_lens, output_topics = batch
        feed_dict = {self.input_contexts: context, self.context_lens: context_lens,
                     self.output_tokens: outputs, self.output_lens: output_lens,
                     self.output_topics: output_topics, self.use_prior: use_prior}
        if repeat > 1:
            tiled_feed_dict = {}
            for key, val in feed_dict.items():
                if key is self.use_prior:
                    tiled_feed_dict[key] = val
                    continue
                multipliers = [1]*len(val.shape)
                multipliers[0] = repeat
                tiled_feed_dict[key] = np.tile(val, multipliers)
            feed_dict = tiled_feed_dict

        if global_t is not None:
            feed_dict[self.global_t] = global_t

        return feed_dict

    def train(self, global_t, sess, train_feed, update_limit=5000):
        elbo_losses = []
        rc_losses = []
        rc_ppls = []
        kl_losses = []
        bow_losses = []
        local_t = 0
        start_time = time.time()
        loss_names =  ["elbo_loss", "bow_loss", "rc_loss", "rc_peplexity", "kl_loss"]
        while True:
            batch = train_feed.next_batch()
            if batch is None:
                break
            if update_limit is not None and local_t >= update_limit:
                break
            feed_dict = self.batch_2_feed(batch, global_t, use_prior=False)
            _, sum_op, elbo_loss, bow_loss, rc_loss, rc_ppl, kl_loss = sess.run([self.train_ops, self.summary_op,
                                                                         self.elbo, self.avg_bow_loss,
                                                                         self.avg_rc_loss, self.rc_ppl, self.avg_kld],
                                                                         feed_dict)
            self.train_summary_writer.add_summary(sum_op, global_t)
            elbo_losses.append(elbo_loss)
            bow_losses.append(bow_loss)
            rc_ppls.append(rc_ppl)
            rc_losses.append(rc_loss)
            kl_losses.append(kl_loss)

            global_t += 1
            local_t += 1
            if local_t % (train_feed.num_batch / 10) == 0:
                kl_w = sess.run(self.kl_w, {self.global_t: global_t})
                self.print_loss("%.2f" % (train_feed.ptr / float(train_feed.num_batch)),
                                loss_names, [elbo_losses, bow_losses, rc_losses, rc_ppls, kl_losses], "kl_w %f" % kl_w)

        # finish epoch!
        epoch_time = time.time() - start_time
        avg_losses = self.print_loss("Epoch Done", loss_names,
                                     [elbo_losses, bow_losses, rc_losses, rc_ppls, kl_losses],
                                     "step time %.4f" % (epoch_time / (local_t + 1)))

        return global_t, avg_losses[0]

    def valid(self, name, sess, valid_feed):
        elbo_losses = []
        rc_losses = []
        rc_ppls = []
        bow_losses = []
        kl_losses = []

        while True:
            batch = valid_feed.next_batch()
            if batch is None:
                break
            feed_dict = self.batch_2_feed(batch, None, use_prior=False, repeat=1)

            elbo_loss, bow_loss, rc_loss, rc_ppl, kl_loss = sess.run(
                [self.elbo, self.avg_bow_loss, self.avg_rc_loss,
                 self.rc_ppl, self.avg_kld], feed_dict)
            elbo_losses.append(elbo_loss)
            rc_losses.append(rc_loss)
            rc_ppls.append(rc_ppl)
            bow_losses.append(bow_loss)
            kl_losses.append(kl_loss)

        avg_losses = self.print_loss(name, ["elbo_loss", "bow_loss", "rc_loss", "rc_peplexity", "kl_loss"],
                                     [elbo_losses, bow_losses, rc_losses, rc_ppls, kl_losses], "")
        return avg_losses[0]

    def test(self, sess, test_feed, num_batch=None, repeat=5, dest=sys.stdout):
        local_t = 0
        recall_bleus = []
        prec_bleus = []

        while True:
            batch = test_feed.next_batch()
            if batch is None or (num_batch is not None and local_t > num_batch):
                break
            feed_dict = self.batch_2_feed(batch, None, use_prior=True, repeat=repeat)
            word_outs, topic_logits = sess.run([self.dec_out_words, self.topic_logits], feed_dict)
            sample_words = np.split(word_outs, repeat, axis=0)
            sample_topic = np.split(topic_logits, repeat, axis=0)

            true_srcs = feed_dict[self.input_contexts]
            true_src_lens = feed_dict[self.context_lens]
            true_outs = feed_dict[self.output_tokens]
            true_topics = feed_dict[self.output_topics]
            local_t += 1

            if dest != sys.stdout:
                if local_t % (test_feed.num_batch / 10) == 0:
                    print("%.2f >> " % (test_feed.ptr / float(test_feed.num_batch))),

            for b_id in range(test_feed.batch_size):
                # print the dialog context
                dest.write("Batch %d index %d\n" % (local_t, b_id))
                for t_id in range(0, true_src_lens[b_id], 1):
                    src_str = " ".join([self.vocab[e] for e in true_srcs[b_id, t_id].tolist() if e != 0])
                    dest.write("Src %d: %s\n" % (t_id, src_str))
                # print the true outputs
                true_tokens = [self.vocab[e] for e in true_outs[b_id].tolist() if e not in [0, self.eos_id, self.go_id]]
                true_str = " ".join(true_tokens).replace(" ' ", "'")
                topic_str = self.topic_vocab[true_topics[b_id]]
                # print the predicted outputs
                dest.write("Target (%s) >> %s\n" % (topic_str, true_str))
                local_tokens = []
                for r_id in range(repeat):
                    pred_outs = sample_words[r_id]
                    pred_topic = np.argmax(sample_topic[r_id], axis=1)[0]
                    pred_tokens = [self.vocab[e] for e in pred_outs[b_id].tolist() if e != self.eos_id and e != 0]
                    pred_str = " ".join(pred_tokens).replace(" ' ", "'")
                    dest.write("Sample %d (%s) >> %s\n" % (r_id, self.topic_vocab[pred_topic], pred_str))
                    local_tokens.append(pred_tokens)

                max_bleu, avg_bleu = utils.get_bleu_stats(true_tokens, local_tokens)
                recall_bleus.append(max_bleu)
                prec_bleus.append(avg_bleu)
                # make a new line for better readability
                dest.write("\n")

        avg_recall_bleu = float(np.mean(recall_bleus))
        avg_prec_bleu = float(np.mean(prec_bleus))
        avg_f1 = 2*(avg_prec_bleu*avg_recall_bleu) / (avg_prec_bleu+avg_recall_bleu+10e-12)
        report = "Avg recall BLEU %f, avg precision BLEU %f and F1 %f (only 1 reference response. Not final result)" \
                 % (avg_recall_bleu, avg_prec_bleu, avg_f1)
        print(report)
        dest.write(report + "\n")
        print("Done testing")


