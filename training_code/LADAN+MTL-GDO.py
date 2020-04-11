import argparse
import random
import os
import warnings
import tensorflow as tf
import time
import pickle as pk
from Model.Topjudge_model import Topjudge
from parser import ConfigParser
import numpy as np
from sklearn import metrics
from law_processed.law_processed import get_law_graph
import keras.layers
from tensorflow.contrib import layers
from Model.LSTM_cell import LSTMDecoder


epsilon=1e-9


def gen_dict(inputs_, law_labels_input_, accu_labels_input_, time_labels_input_):

    feed_dict_ = {fact_input: inputs_, law_labels: law_labels_input_,
                  accu_labels: accu_labels_input_, time_labels: time_labels_input_}

    return feed_dict_


def evaluation_multitask(y, prediction, task_num, correct_tags, total_tags):
    accuracy_ = []
    metrics_acc = []
    for x in range(task_num):
        accuracy_1 = correct_tags[x] / total_tags * 100
        accuracy_metric = metrics.accuracy_score(y[x], prediction[x])
        macro_recall = metrics.recall_score(y[x], prediction[x], average='macro')
        micro_recall = metrics.recall_score(y[x], prediction[x], average='micro')
        macro_precision = metrics.precision_score(y[x], prediction[x], average='macro')
        micro_precision = metrics.precision_score(y[x], prediction[x], average='micro')
        macro_f1 = metrics.f1_score(y[x], prediction[x], average='macro')
        micro_f1 = metrics.f1_score(y[x], prediction[x], average='micro')
        accuracy_.append(accuracy_1)
        metrics_acc.append(
            (accuracy_metric, macro_recall, micro_recall, macro_precision, micro_precision, macro_f1, micro_f1))
    return accuracy_, metrics_acc


def get_safe_shift(logits, mask):
    """
    :param logits: A tf.Tensor of shape [B, TQ, TK] of dtype tf.float32
    :param mask: A tf.Tensor of shape [B, TQ, TK] of dtype tf.float32
    where TQ, TK are the maximum lengths of the queries resp. the keys in the batch
    """

    # Determine minimum
    K_shape=logits.get_shape().as_list()
    mask_shape=mask.get_shape().as_list()
    if mask_shape!=K_shape:
        mask=tf.tile(mask,[1]+[K_shape[1]//mask_shape[1]]+[1]*(len(K_shape)-2))

    logits_min = tf.reduce_min(logits, axis=-1, keepdims=True)      # [B, TQ, 1]
    logits_min = tf.tile(logits_min, multiples=[1]*(len(K_shape)-1)+[K_shape[-1]])  # [B, TQ, TK]

    logits = tf.where(condition=mask > .5, x=logits, y=logits_min)

    # Determine maximum
    logits_max = tf.reduce_max(logits, axis=-1, keepdims=True, name="logits_max")      # [B, TQ, 1]
    logits_shifted = tf.subtract(logits, logits_max, name="logits_shifted")    # [B, TQ, TK]

    return logits_shifted


def padding_aware_softmax(logits, key_mask, query_mask=None):

    logits_shifted=get_safe_shift(logits, key_mask)

    # Apply exponential
    weights_unscaled = tf.exp(logits_shifted)

    # Apply mask
    weights_unscaled = tf.multiply(key_mask, weights_unscaled)     # [B, TQ, TK]

    # Derive total mass
    weights_total_mass = tf.reduce_sum(weights_unscaled, axis=-1, keepdims=True)     # [B, TQ, 1]

    # Avoid division by zero
    if query_mask:
        weights_total_mass = tf.where(condition=tf.equal(query_mask, 1),
                                    x=weights_total_mass,
                                    y=tf.ones_like(weights_total_mass))

    # Normalize weights
    weights = tf.divide(weights_unscaled, weights_total_mass + epsilon)   # [B, TQ, TK]

    return weights


def atten_encoder_mask(Q, K, fc_layer=None, mask=None, weights_regularizer=None, K_ori=False, div_norm=True):
    '''
    :param Q: [..., seq_len_q, F] : the attention vector u
    :param K: [..., seq_len_k, F]
    :param mask:
    :return: a tensor whose size is [..., F]
    '''
    V = K
    K_shape = K.get_shape().as_list() # size[x, y, z]
    if fc_layer is not None:
        K = fc_layer(K)
    else:
        K = layers.fully_connected(K, K_shape[-1], activation_fn=tf.nn.tanh, weights_regularizer=weights_regularizer)  # size: [x, y, z]

    if not K_ori:
        V = K
    # ======================================================
    # Q=tf.transpose(Q,[-1,-2])
    # scores=tf.map_fn(lambda x:x@Q,K,dtype=tf.float32)
    # -------------another implementation-------------------
    scores = tf.reduce_sum(K * Q, -1)  # size: [x, y]
    if div_norm:
        scores = scores/tf.sqrt(tf.cast(K_shape[-1],tf.float32))  # size: [x, y]
    # =======================================================
    # scores=tf.nn.softmax(scores,-2)
    if mask is not None:
        # scores = scores - tf.reduce_max(scores, -1, keepdims=True)  # e^(a-b)=e^a * e^-b
        # exp = tf.exp(scores) * mask  # big scores cause inf in exp
        # scores = exp / (tf.reduce_sum(exp, -1, keepdims=True) + epsilon)  # the score is attention weight a , size: [x, y]
        scores = padding_aware_softmax(scores, mask)
    else:
        scores = tf.nn.softmax(scores, -1)
    return tf.reduce_sum(tf.expand_dims(scores, -1) * V, -2), scores  # size: [x, z]


def run_model(input, mask, model):
    input_shape = input.get_shape().as_list()
    mask = tf.expand_dims(mask, -1)
    mask_shape = mask.get_shape().as_list()
    input = tf.reshape(input, [int(np.prod(input_shape[:-2]))] + input_shape[-2:])
    mask = tf.reshape(mask, [int(np.prod(mask_shape[:-2]))] + mask_shape[-2:])
    out = model(input, mask=mask)
    rep = tf.reshape(out, input_shape[:-1] + [lstm_size * 2])
    return rep


configFilePath = 'config/single.config'
config = ConfigParser(configFilePath)
parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c')
parser.add_argument('--gpu', '-g')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


batch_size = 128
max_epoch = 16
sent_len_fact = 100
doc_len_fact = 15
doc_len_law = 10
sent_len_law = 100
learning_rate = 1e-3

lstm_size = 128
clr_fc1_size = 512
clr_fc2_size = 256
law_relation_threshold = 0.3


task = ['law', 'accu', 'time']

with open('data/w2id_thulac.pkl', 'rb') as f:
    word2id_dict = pk.load(f)
    f.close()

emb_path = 'data/cail_thulac.npy'
word_embedding = np.cast[np.float32](np.load(emb_path))

word_dict_len = len(word2id_dict)
vec_size = 200
shuffle = True

n_law = 103
n_accu = 119
n_term = 12

with tf.Graph().as_default():
    regualrizer = layers.l2_regularizer(0.)
    fact_input = tf.placeholder(tf.int32, [batch_size, doc_len_fact, sent_len_fact], name='fact')
    law_labels = tf.placeholder(tf.int32, [batch_size], name='fact')
    accu_labels = tf.placeholder(tf.int32, [batch_size], name='fact')
    time_labels = tf.placeholder(tf.int32, [batch_size], name='fact')

    law_input, graph_list_1, graph_membership, neigh_index = get_law_graph(law_relation_threshold, 'data/w2id_thulac.pkl', 15, 100)

    fact_mask = tf.cast(tf.cast(fact_input - word2id_dict['BLANK'], tf.bool), tf.float32)
    fact_sent_len = tf.reduce_sum(fact_mask, -1)
    fact_doc_mask = tf.cast(tf.cast(fact_sent_len, tf.bool), tf.float32)
    fact_doc_len = tf.reduce_sum(fact_doc_mask, -1)

    law_mask = tf.cast(tf.cast(law_input - word2id_dict['BLANK'], tf.bool), tf.float32)
    law_sent_len = tf.reduce_sum(law_mask, -1)
    law_doc_mask = tf.cast(tf.cast(law_sent_len, tf.bool), tf.float32)
    law_doc_len = tf.reduce_sum(law_doc_mask, -1)

    # word_embedding_1 = tf.Variable(name="word_1", initial_value=tf.random_uniform([word_dict_len - 1, vec_size]), dtype=tf.float32, trainable=True)
    # word_embedding_0 = tf.Variable(name="word_0", initial_value=tf.zeros([1, vec_size]), dtype=tf.float32, trainable=True)
    # word_embedding_uniform = tf.concat([word_embedding_0, word_embedding_1], axis=0)

    fact_description = tf.nn.embedding_lookup(word_embedding, fact_input)
    law_description = tf.nn.embedding_lookup(word_embedding, law_input)

    max_graph = len(graph_list_1)
    deg_list = [len(neigh_index[i]) for i in range(n_law)]
    graph_list = list(zip(*graph_membership))[1]

    gold_matrix_law = tf.one_hot(law_labels, 103, dtype=tf.float32)
    gold_matrix_accu = tf.one_hot(accu_labels, 119, dtype=tf.float32)
    gold_matrix_time = tf.one_hot(time_labels, 12, dtype=tf.float32)

    #############----------------------###################
    graph_label = tf.dynamic_partition(tf.transpose(gold_matrix_law, [1, 0]), graph_list, max_graph)  # size: [batch_size, graph_num, N_each_graph])
    label = []
    for i in range(max_graph):
        label.append(tf.reduce_sum(graph_label[i], 0, keepdims=True))

    graph_label = tf.transpose(tf.concat(label, 0), [1, 0])  # size: [batch_size, graph_num]
    #############----------------------###################

    neigh_index = sorted(neigh_index.items(), key=lambda x: len(x[1]))
    max_deg = len(neigh_index[-1][1])
    t = 0
    adj_list = [[]]
    for i in range(n_law):
        each = neigh_index[i]
        if len(each[1]) != t:
            for j in range(t, len(each[1])):
                adj_list.append([])
            t = len(each[1])
        adj_list[-1].append(each[1])

    u_aw = tf.get_variable('u_aw', shape=[1, lstm_size * 2], initializer=layers.xavier_initializer())
    u_as = tf.get_variable('u_as', shape=[1, lstm_size * 2], initializer=layers.xavier_initializer())

    Fully_atten_sent_1 = keras.layers.Dense(lstm_size * 2, name='Fully_atten_sent_1')
    Fully_atten_doc_1 = keras.layers.Dense(lstm_size * 2, name='Fully_atten_doc_1')

    model = keras.Sequential([keras.layers.Bidirectional(keras.layers.GRU(lstm_size, return_sequences=True),merge_mode='concat')])
    rep_law = run_model(law_description, law_mask, model)
    rep_fact = run_model(fact_description, fact_mask, model)

    # rep_law_ = tf.reduce_mean(rep_law, -2)
    # rep_fact_ = tf.reduce_mean(rep_fact, -2)

    rep_law_, _ = atten_encoder_mask(u_aw, rep_law, Fully_atten_sent_1, law_mask, K_ori=True)
    rep_fact_, _ = atten_encoder_mask(u_aw, rep_fact, Fully_atten_sent_1, fact_mask, K_ori=True)

    model_1 = keras.Sequential([keras.layers.Bidirectional(keras.layers.GRU(lstm_size, return_sequences=True), merge_mode='concat')])
    rep_law_1 = run_model(rep_law_, law_doc_mask, model_1)
    rep_fact_1 = run_model(rep_fact_, fact_doc_mask, model_1)

    # rep_law_1 = tf.reduce_mean(rep_law_1, -2)
    # rep_fact_1 = tf.reduce_mean(rep_fact_1, -2)
    rep_law_1, _ = atten_encoder_mask(u_as, rep_law_1, Fully_atten_doc_1, law_doc_mask, K_ori=True)
    rep_fact_1, _ = atten_encoder_mask(u_as, rep_fact_1, Fully_atten_doc_1, fact_doc_mask, K_ori=True)

    with tf.name_scope('interaction'):

        indices = tf.dynamic_partition(tf.range(n_law), deg_list, max_deg + 1)
        law_representation = tf.dynamic_partition(rep_law_1, graph_list, max_graph)

        atten_list = []
        for i in range(max_graph):
            # u = tf.reduce_max(law_representation[i], 0)  # law_representation[i]: [n, law_size]
            # u_2 = tf.reduce_min(law_representation[i], 0)
            u = tf.reduce_mean(law_representation[i], 0)
            atten_list.append(
                tf.concat([u], -1))

    with tf.name_scope('law_re_encoder'):
        law_u = tf.gather(atten_list, graph_list)  # size:[183, law_size]; law: [183, x, y, word_size]
        Fully_connected_1 = keras.layers.Dense(lstm_size *2)
        Fully_connected_2 = keras.layers.Dense(lstm_size *2)
        u_law_w = tf.reshape(Fully_connected_1(law_u), [-1, 1, 1, lstm_size *2])
        u_law_s = tf.reshape(Fully_connected_2(law_u), [-1, 1, lstm_size *2])

        Fully_atten_sent_2 = keras.layers.Dense(lstm_size *2, kernel_regularizer=regualrizer)
        Fully_atten_doc_2 = keras.layers.Dense(lstm_size *2, kernel_regularizer=regualrizer)

        # model_2 = keras.Sequential(
        #     [keras.layers.Bidirectional(keras.layers.LSTM(lstm_size, return_sequences=True), merge_mode='concat')])
        # rep_law = run_model(law_description, law_mask, model_2)
        rep_law, _ = atten_encoder_mask(u_law_w, rep_law, Fully_atten_sent_2, law_mask, K_ori=True)

        model_3 = keras.Sequential(
            [keras.layers.Bidirectional(keras.layers.GRU(lstm_size, return_sequences=True), merge_mode='concat')])
        rep_law_2 = run_model(rep_law, law_doc_mask, model_3)
        rep_law_2, _ = atten_encoder_mask(u_law_s, rep_law_2, Fully_atten_doc_2, law_doc_mask, K_ori=True)

    with tf.name_scope('fact_re_encoder'):

        Fully_connected_graph = keras.layers.Dense(max_graph)
        fact_graph_choose_1 = Fully_connected_graph(rep_fact_1)
        fact_graph_choose = tf.nn.softmax(fact_graph_choose_1, -1)

        # graph_chose_loss = tf.losses.softmax_cross_entropy(graph_label, fact_graph_choose_1)
        graph_chose_loss = tf.nn.softmax_cross_entropy_with_logits(logits=fact_graph_choose_1, labels=graph_label)
        graph_chose_loss = tf.reduce_sum(graph_chose_loss)/128.0
        graph_L = tf.arg_max(graph_label, -1)
        correct_graph = tf.nn.in_top_k(fact_graph_choose_1, graph_L, 1)

        #------------------------------------one-hot-----------------------------#

        fact_graph_choose = tf.where(fact_graph_choose == tf.reduce_max(fact_graph_choose, -1), tf.ones_like(fact_graph_choose), tf.zeros_like(fact_graph_choose))

        ###########################################################################

        atten_tensor = tf.reshape(tf.concat(atten_list, 0), [-1, lstm_size * 2])
        u_fact = fact_graph_choose @ atten_tensor

        u_fact_w = tf.reshape(Fully_connected_1(u_fact), [-1, 1, 1, lstm_size * 2])
        u_fact_s = tf.reshape(Fully_connected_2(u_fact), [-1, 1, lstm_size * 2])

        # rep_fact = run_model(fact_description, fact_mask, model_2)
        rep_fact, _ = atten_encoder_mask(u_fact_w, rep_fact, Fully_atten_sent_2, fact_mask, K_ori=True)

        rep_fact_2 = run_model(rep_fact, fact_doc_mask, model_3)
        rep_fact_2, _ = atten_encoder_mask(u_fact_s, rep_fact_2, Fully_atten_doc_2, fact_doc_mask, K_ori=True)

        fact_repr = tf.concat([rep_fact_1, rep_fact_2], -1)  # size: [batch_size, 2 * fact_size]
        law_repr = tf.concat([rep_law_1, rep_law_2], -1)  # size: [183 , 2 * law_size]

    Full_law_1 = keras.layers.Dense(clr_fc1_size)
    Full_law_2 = keras.layers.Dense(n_law)
    law_output = Full_law_2(tf.nn.relu(Full_law_1(law_repr)))
    loss_law_article = tf.losses.softmax_cross_entropy(tf.one_hot(tf.range(n_law), n_law), law_output)

    decoder_1_ = keras.layers.Dense(256)
    decoder_2_ = keras.layers.Dense(256)
    decoder_3_ = keras.layers.Dense(256)

    decoder_1 = keras.layers.Dense(103)
    decoder_2 = keras.layers.Dense(119)
    decoder_3 = keras.layers.Dense(12)
    output_task1, output_task2, output_task3 = [decoder_1(tf.nn.relu(decoder_1_(fact_repr))),
                                                decoder_2(tf.nn.relu(decoder_2_(fact_repr))),
                                                decoder_3(tf.nn.relu(decoder_3_(fact_repr)))]
    '''
        output_task1: embeddings for law prediction
        output_task2: embeddings for accu prediction
        output_task3: embeddings for time prediction
        '''

    law_prob = tf.nn.softmax(output_task1, -1)
    accu_prob = tf.nn.softmax(output_task2, -1)
    time_prob = tf.nn.softmax(output_task3, -1)

    law_predictions = tf.argmax(law_prob, 1)
    accu_predictions = tf.argmax(accu_prob, 1)
    time_predictions = tf.argmax(time_prob, 1)

    loss_1 = tf.nn.softmax_cross_entropy_with_logits(logits=output_task1, labels=gold_matrix_law)
    loss_2 = tf.nn.softmax_cross_entropy_with_logits(logits=output_task2, labels=gold_matrix_accu)
    loss_3 = tf.nn.softmax_cross_entropy_with_logits(logits=output_task3, labels=gold_matrix_time)

    law_loss = tf.reduce_sum(loss_1)
    accu_loss = tf.reduce_sum(loss_2)
    time_loss = tf.reduce_sum(loss_3)

    # loss = (law_loss + accu_loss + time_loss) / batch_size + loss_law_article / 103.0
    loss = (law_loss + accu_loss + time_loss) / batch_size + loss_law_article / 103.0 + 0.1 * graph_chose_loss
    tf.add_to_collection('losses_1', tf.contrib.layers.l2_regularizer(0.0001)(loss_1))
    tf.add_to_collection('losses_2', tf.contrib.layers.l2_regularizer(0.0001)(loss_2))
    tf.add_to_collection('losses_3', tf.contrib.layers.l2_regularizer(0.0001)(loss_3))

    loss_total = loss + tf.add_n(tf.get_collection('losses_1')) + tf.add_n(tf.get_collection('losses_2')) + tf.add_n(
        tf.get_collection('losses_3'))

    global_step = tf.Variable(0, trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=.9, beta2=.999, epsilon=1e-7)
    train_op = optimizer.minimize(loss_total, global_step=global_step)

    correct_law = tf.nn.in_top_k(output_task1, law_labels, 1)
    correct_accu = tf.nn.in_top_k(output_task2, accu_labels, 1)
    correct_time = tf.nn.in_top_k(output_task3, time_labels, 1)

    ###########################-----------graph built over-----------###############################

    initializer = tf.global_variables_initializer()

    # vars_ = {}
    # for var in tf.global_variables():
    #     # print(var)
    #     vars_[var.name.split(":")[0]] = var
    # saver = tf.train.Saver(vars_)

    config_tf = tf.ConfigProto()
    config_tf.gpu_options.per_process_gpu_memory_fraction = 0.25
    config_tf.gpu_options.allow_growth = True

    sess = tf.Session(config=config_tf)
    sess.run(initializer)

    total_loss = 0.0
    graph_chose_total = 0.0
    ave_graph_acc = 0.0
    start_time = time.time()

    ############################--------------initialized over---------------########################
    f_train = pk.load(open('legal_basis_data/train_processed_thulac_Legal_basis.pkl', 'rb'))
    f_valid = pk.load(open('legal_basis_data/valid_processed_thulac_Legal_basis.pkl', 'rb'))
    f_test = pk.load(open('legal_basis_data/test_processed_thulac_Legal_basis.pkl', 'rb'))

    train_step = int(len(f_train['fact_list']) / batch_size) + 1
    lose_num_train = train_step * batch_size - len(f_train['fact_list'])

    valid_step = int(len(f_valid['fact_list']) / batch_size) + 1
    lose_num_valid = valid_step * batch_size - len(f_valid['fact_list'])

    test_step = int(len(f_test['fact_list']) / batch_size) + 1
    lose_num_test = test_step * batch_size - len(f_test['fact_list'])

    fact_train = f_train['fact_list']
    law_labels_train = f_train['law_label_lists']
    accu_label_train = f_train['accu_label_lists']
    term_train = f_train['term_lists']

    if shuffle:
        index = [i for i in range(len(f_train['term_lists']))]
        random.shuffle(index)
        fact_train = [fact_train[i] for i in index]
        law_labels_train = [law_labels_train[i] for i in index]
        accu_label_train = [accu_label_train[i] for i in index]
        term_train = [term_train[i] for i in index]

        for epoch in range(max_epoch):
            for i in range(train_step):
                if i == train_step - 1:
                    inputs = np.array(fact_train[i * batch_size:] + fact_train[:lose_num_train], dtype='int32')
                    law_labels_input = np.array(law_labels_train[i * batch_size:] + law_labels_train[:lose_num_train],
                                                dtype='int32')
                    accu_labels_input = np.array(accu_label_train[i * batch_size:] + accu_label_train[:lose_num_train],
                                                 dtype='int32')
                    time_labels_input = np.array(term_train[i * batch_size:] + term_train[:lose_num_train],
                                                 dtype='int32')
                else:
                    inputs = np.array(fact_train[i * batch_size: (i + 1) * batch_size], dtype='int32')
                    law_labels_input = np.array(law_labels_train[i * batch_size: (i + 1) * batch_size], dtype='int32')
                    accu_labels_input = np.array(accu_label_train[i * batch_size: (i + 1) * batch_size], dtype='int32')
                    time_labels_input = np.array(term_train[i * batch_size: (i + 1) * batch_size], dtype='int32')
                # print(inputs.shape)
                feed_dict = gen_dict(inputs, law_labels_input, accu_labels_input, time_labels_input)
                correct_graph_ = sess.run([correct_graph], feed_dict=feed_dict)
                loss_value, _, graph_chose_value = sess.run([loss_total, train_op, graph_chose_loss],
                                                            feed_dict=feed_dict)
                total_loss += loss_value
                graph_chose_total += graph_chose_value
                ave_graph_acc += np.sum(np.cast[np.int32](correct_graph_)) / 128.0

                if (i + 1) == train_step:
                    duration = time.time() - start_time
                    start_time = time.time()
                    print('Step %d: loss = %.2f (%.3f sec)' % (i, total_loss, duration))
                    losses = total_loss
                    total_loss = 0.0

                    print(graph_chose_total)
                    graph_chose_total = 0.0

                    print(ave_graph_acc / train_step)
                    ave_graph_acc = 0.0
            ############################----------the following is valid prediction-----------------###############################
            predic_law, predic_accu, predic_time = [], [], []
            y_law, y_accu, y_time = [], [], []
            time_correct = []
            loss_sum = 0
            total_tags = 0.0
            correct_tags_law = 0
            correct_tags_accu = 0
            correct_tags_time = 0
            for i in range(valid_step):
                if i == valid_step - 1:
                    inputs = np.array(f_valid['fact_list'][i * batch_size:] + f_valid['fact_list'][:lose_num_valid],
                                      dtype='int32')
                    law_labels_input = np.array(
                        f_valid['law_label_lists'][i * batch_size:] + f_valid['law_label_lists'][:lose_num_valid],
                        dtype='int32')
                    accu_labels_input = np.array(
                        f_valid['accu_label_lists'][i * batch_size:] + f_valid['accu_label_lists'][:lose_num_valid],
                        dtype='int32')
                    time_labels_input = np.array(
                        f_valid['term_lists'][i * batch_size:] + f_valid['term_lists'][:lose_num_valid],
                        dtype='int32')
                else:
                    inputs = np.array(f_valid['fact_list'][i * batch_size: (i + 1) * batch_size], dtype='int32')
                    law_labels_input = np.array(f_valid['law_label_lists'][i * batch_size: (i + 1) * batch_size],
                                                dtype='int32')
                    accu_labels_input = np.array(f_valid['accu_label_lists'][i * batch_size: (i + 1) * batch_size],
                                                 dtype='int32')
                    time_labels_input = np.array(f_valid['term_lists'][i * batch_size: (i + 1) * batch_size],
                                                 dtype='int32')

                feed_dict_valid = gen_dict(inputs, law_labels_input, accu_labels_input, time_labels_input)
                num_y = batch_size
                if i + 1 == valid_step:
                    num_y = batch_size - lose_num_valid

                total_tags += num_y
                correct_law_, correct_accu_, correct_time_, predic_law_, predic_accu_, predic_time_, y_law_, y_accu_, y_time_ = sess.run(
                    (correct_law, correct_accu, correct_time, law_predictions, accu_predictions, time_predictions,
                     law_labels, accu_labels, time_labels), feed_dict=feed_dict_valid)

                predic_law += list(predic_law_[:num_y])
                predic_accu += list(predic_accu_[:num_y])
                predic_time += list(predic_time_[:num_y])

                y_law += list(y_law_[:num_y])
                y_accu += list(y_accu_[:num_y])
                y_time += list(y_time_[:num_y])
                time_correct += list(correct_time_[:num_y])

                correct_tags_law += np.sum(np.cast[np.int32](correct_law_[:num_y]))
                correct_tags_accu += np.sum(np.cast[np.int32](correct_accu_[:num_y]))
                correct_tags_time += np.sum(np.cast[np.int32](correct_time_[:num_y]))

            prediction = [predic_law, predic_accu, predic_time]
            y = [y_law, y_accu, y_time]
            correct_tags = [correct_tags_law, correct_tags_accu, correct_tags_time]
            print(len(time_correct))
            accuracy, metric = evaluation_multitask(y, prediction, 3, correct_tags, total_tags)
            print('Now_epoch is: {}'.format(epoch))
            for i in range(3):
                print('Accuracy for {} prediction is: '.format(task[i]), accuracy[i])
                print('Other metrics for {} prediction is: '.format(task[i]), metric[i])

            print('\n')

            ############################----------the following is valid prediction-----------------###############################
            predic_law, predic_accu, predic_time = [], [], []
            y_law, y_accu, y_time = [], [], []
            time_correct = []
            loss_sum = 0
            total_tags = 0.0
            correct_tags_law = 0
            correct_tags_accu = 0
            correct_tags_time = 0
            for i in range(test_step):
                if i == test_step - 1:
                    inputs = np.array(f_test['fact_list'][i * batch_size:] + f_test['fact_list'][:lose_num_test],
                                      dtype='int32')
                    law_labels_input = np.array(
                        f_test['law_label_lists'][i * batch_size:] + f_test['law_label_lists'][:lose_num_test],
                        dtype='int32')
                    accu_labels_input = np.array(
                        f_test['accu_label_lists'][i * batch_size:] + f_test['accu_label_lists'][:lose_num_test],
                        dtype='int32')
                    time_labels_input = np.array(
                        f_test['term_lists'][i * batch_size:] + f_test['term_lists'][:lose_num_test],
                        dtype='int32')
                else:
                    inputs = np.array(f_test['fact_list'][i * batch_size: (i + 1) * batch_size], dtype='int32')
                    law_labels_input = np.array(f_test['law_label_lists'][i * batch_size: (i + 1) * batch_size],
                                                dtype='int32')
                    accu_labels_input = np.array(f_test['accu_label_lists'][i * batch_size: (i + 1) * batch_size],
                                                 dtype='int32')
                    time_labels_input = np.array(f_test['term_lists'][i * batch_size: (i + 1) * batch_size],
                                                 dtype='int32')

                feed_dict_test = gen_dict(inputs, law_labels_input, accu_labels_input, time_labels_input)
                num_y = batch_size
                if i + 1 == test_step:
                    num_y = batch_size - lose_num_test

                total_tags += num_y
                correct_law_, correct_accu_, correct_time_, predic_law_, predic_accu_, predic_time_, y_law_, y_accu_, y_time_ = sess.run(
                    (correct_law, correct_accu, correct_time, law_predictions, accu_predictions, time_predictions,
                     law_labels, accu_labels, time_labels), feed_dict=feed_dict_test)

                predic_law += list(predic_law_[:num_y])
                predic_accu += list(predic_accu_[:num_y])
                predic_time += list(predic_time_[:num_y])

                y_law += list(y_law_[:num_y])
                y_accu += list(y_accu_[:num_y])
                y_time += list(y_time_[:num_y])
                time_correct += list(correct_time_[:num_y])

                correct_tags_law += np.sum(np.cast[np.int32](correct_law_[:num_y]))
                correct_tags_accu += np.sum(np.cast[np.int32](correct_accu_[:num_y]))
                correct_tags_time += np.sum(np.cast[np.int32](correct_time_[:num_y]))

            prediction = [predic_law, predic_accu, predic_time]
            y = [y_law, y_accu, y_time]
            correct_tags = [correct_tags_law, correct_tags_accu, correct_tags_time]

            accuracy, metric = evaluation_multitask(y, prediction, 3, correct_tags, total_tags)
            print('Now_testing')
            for i in range(3):
                print('Accuracy for {} prediction is: '.format(task[i]), accuracy[i])
                print('Other metrics for {} prediction is: '.format(task[i]), metric[i])

            print('\n')
