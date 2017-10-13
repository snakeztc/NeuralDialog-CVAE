#    Copyright (C) 2017 Tiancheng Zhao, Carnegie Mellon University


class CVAEConfig(object):
    description = None
    update_limit = 3000  # the number of mini-batch before evaluating the model

    vocab_size = 4000  # max vocabulary size

    train_size = 5000000  # valid-dataset size
    valid_size = 2000  # valid-dataset size
    test_size = 500  # test-dataset size

    bow_weights = 0.1  # weight of bow loss

    # how to encode utterance.
    # bow: add word embedding together
    # rnn: RNN utterance encoder
    # bi_rnn: bi_directional RNN utterance encoder
    sent_type = "bi_rnn"

    # latent variable (gaussian variable)
    latent_size = 50  # the dimension of latent variable
    full_kl_step = 50000  # how many batch before KL cost weight reaches 1.0
    dec_keep_prob = 1.0  # do we use word drop decoder [Bowman el al 2015]

    # Network general
    cell_type = "gru"  # gru or lstm
    embed_size = 200  # word embedding size
    cxt_cell_size = 200  # context encoder hidden size
    sent_cell_size = 200  # utterance encoder hidden size
    dec_cell_size = 200  # response decoder hidden size
    max_utt_len = 80  # max number of words in an utterance
    num_layer = 1  # number of context RNN layers

    # Optimization parameters
    op = "adam"
    grad_clip = 5.0  # gradient abs max cut
    init_w = 0.08  # uniform random from [-init_w, init_w]
    batch_size = 64  # mini-batch size
    init_lr = 0.001  # initial learning rate
    lr_hold = 1  # only used by SGD
    lr_decay = 0.6  # only used by SGD
    keep_prob = 1.0  # drop out rate
    improve_threshold = 0.996  # for early stopping
    patient_increase = 2.0  # for early stopping
    early_stop = True
    max_epoch = 100  # max number of epoch of training
    grad_noise = 0.0  # inject gradient noise?
