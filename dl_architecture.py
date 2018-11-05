from keras.layers import Conv1D
from keras.models import Model
from keras.layers import Input, Embedding, Dense, Dropout, Concatenate, BatchNormalization, MaxPooling1D, Flatten
from keras.layers.merge import concatenate
from keras.optimizers import Adam
import numpy as np


def build_model(unigrams_shape, num_classes, charvec_shape, char_vocab_size):

    inputs = []
    optimizer = Adam(lr=0.0008)

    #tfidf matrix input
    tfidf_matrix = Input(shape=(unigrams_shape,))
    inputs.append(tfidf_matrix)

    # char level
    char_sequence_input = Input(shape=(charvec_shape,), dtype='int32')
    char_dense = charConv(char_sequence_input, charvec_shape, char_vocab_size)
    inputs.append(char_sequence_input)
    mergedconv = concatenate([tfidf_matrix, char_dense])
    dense_all = Dense(256, activation='relu')(mergedconv)
    dropout = Dropout(0.4)(dense_all)
    dense3 = Dense(num_classes, activation='softmax')(dropout)
    model = Model(inputs=inputs, outputs=dense3)
    model.summary()
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['acc'])
    return model


def make_charvec(data, train=True, char_vocab={}, max_text_len=0):
    counter = 2
    if train:
        all_texts = ' '.join(data)
        for char in all_texts:
            if char not in char_vocab:
                char_vocab[char] = counter
                counter += 1
        max_text_len = max([len(text) for text in data])
    else:
        if char_vocab == [] or max_text_len == 0:
            raise Exception('You have to add char vocabulary and max train text len as input')
    output_chars = np.zeros((len(data), max_text_len), dtype='int32')
    for i in range(len(data)):
        for j in range(max_text_len):
            if j < len(data[i]):
                if data[i][j] in char_vocab:
                    output_chars[i][j] = char_vocab[data[i][j]]
                else:
                    output_chars[i][j] = 1

    return output_chars, char_vocab, max_text_len


def charConv(sequence_input, max_seq_length, vocab_size):
    convs = []
    filter_sizes = [4, 5]

    embedding_layer = Embedding(vocab_size, 200, input_length=max_seq_length,)(sequence_input)

    for fsz in filter_sizes:
        l_conv = Conv1D(filters=172, kernel_size=fsz, activation='relu')(embedding_layer)
        l_bn = BatchNormalization()(l_conv)
        l_pool = MaxPooling1D(fsz)(l_bn)
        convs.append(l_pool)

    l_merge = Concatenate(axis=1)(convs)
    l_cov1 = Conv1D(200, 5, activation='relu')(l_merge)
    l_bn1 = BatchNormalization()(l_cov1)
    l_pool1 = MaxPooling1D(40)(l_bn1)
    l_dropout = Dropout(0.4)(l_pool1)
    l_flat = Flatten()(l_dropout)
    return l_flat





