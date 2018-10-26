# -*- coding: utf-8 -*-
import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import numpy as np
from dl_architecture import make_charvec, build_model
from keras.callbacks import ModelCheckpoint
from keras import backend as K

from sklearn.preprocessing import Normalizer
from sklearn import pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import preprocessing
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from collections import defaultdict
import gc


def remove_email(text, replace_token):
    return re.sub(r'[\w\.-]+@[\w\.-]+', replace_token, text)


def remove_url(text, replace_token):
    regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.sub(regex, replace_token, text)


def preprocess(df_data):
    df_data['text_clean'] = df_data['text'].map(lambda x: remove_url(x, "HTTPURL"))
    df_data['text_clean'] = df_data['text_clean'].map(lambda x: remove_email(x, 'EMAIL'))
    return df_data


def preprocess_data(df_data, target, drop, tags_to_idx = []):

    df_data = preprocess(df_data)

    # shuffle the corpus and optionaly choose the chunk you want to use if you don't want to use the whole thing - will be much faster
    df_data = df_data.sample(frac=1, random_state=1)
    tags = df_data[target].tolist()

    if len(tags_to_idx) < 1:
        tags_to_idx = list(set(df_data[target].tolist()))
    df_data = df_data.drop([target], axis=1)
    if len(drop) > 0:
        df_data = df_data.drop(drop, axis=1)

    y = np.array([tags_to_idx.index(tmp_y) for tmp_y in tags])
    return df_data, y, tags_to_idx


class text_col(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    def fit(self, x, y=None):
        return self
    def transform(self, data_dict):
        return data_dict[self.key]


#fit and transform numeric features, used in scikit Feature union
class digit_col(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self
    def transform(self, hd_searches):
        d_col_drops=['text', 'no_punctuation', 'no_stopwords', 'text_clean', 'affixes', 'affix_punct']
        hd_searches = hd_searches.drop(d_col_drops,axis=1).values
        scaler = preprocessing.MinMaxScaler().fit(hd_searches)
        return scaler.transform(hd_searches)


def train(xtrain, ytrain, xval, yval, lang, tags_to_idx):
    checkpointer = ModelCheckpoint(filepath="./models/model_" + lang + "_weights.hdf5",
                                   verbose=1,
                                   monitor="val_acc",
                                   save_best_only=True,
                                   mode="max")


    #print("Train and dev shape: ", xtrain.shape, xval.shape)
    counts = defaultdict(int)
    for c in ytrain.tolist():
        counts[c] += 1


    if lang!='all':
        character_vectorizer = CountVectorizer(analyzer='char', ngram_range=(3,6), lowercase=False, min_df=5, max_df=0.3)
    else:
        character_vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(3,5), lowercase=False, min_df=5, max_df=0.3)

    tfidf_transformer = TfidfTransformer(sublinear_tf=True)

    tfidf_matrix = pipeline.Pipeline([
        ('character', pipeline.Pipeline(
            [('s5', text_col(key='text_clean')), ('character_vectorizer', character_vectorizer),
             ('tfidf_character', tfidf_transformer)])),
        ('scale', Normalizer())])

    tfidf_matrix = tfidf_matrix.fit(xtrain)
    tfidf_matrix_test = tfidf_matrix.transform(xtrain)
    print('tfidf matrix size: ', tfidf_matrix_test.shape)
    ngrams_matrix_shape = tfidf_matrix_test.shape[1]
    tfidf_matrix_val = tfidf_matrix.transform(xval)

    charvec, char_vocab, max_train_len_char = make_charvec(xtrain.text_clean.tolist())
    char_vocab_size = len(char_vocab) + 2
    charvec_shape = charvec.shape[1]
    charvec_val, _, _ = make_charvec(xval.text_clean.tolist(), train=False, char_vocab=char_vocab, max_text_len=max_train_len_char)

    num_classes = len(set(yval.tolist()))

    textmodel_data = ngrams_matrix_shape, num_classes, charvec_shape, char_vocab_size, tfidf_matrix, char_vocab, max_train_len_char, tags_to_idx

    with open('models/model_' + lang + '_data.pk', 'wb') as f:
        pickle.dump(textmodel_data, f, protocol=2)

    if lang != 'all':
        num_epoch = 20
    else:
        num_epoch = 10
    model = build_model(ngrams_matrix_shape, num_classes, charvec_shape, char_vocab_size)
    model.fit([tfidf_matrix_test, charvec], ytrain, validation_data=([tfidf_matrix_val, charvec_val], yval), batch_size=16, epochs=num_epoch, verbose=0, callbacks=[checkpointer])

    K.clear_session()
    gc.collect()

    return model


def test_trained_model(data_test, target, drop, lang):
    textmodel_data = pickle.load(open('models/model_' + lang + '_data.pk', 'rb'))

    unigrams_shape, num_classes, charvec_shape, char_vocab_size,tfidf_matrix, char_vocab, max_train_len_char, tags_to_idx = textmodel_data
    xtest, ytest, _ = preprocess_data(data_test, target, drop, tags_to_idx=tags_to_idx)
    tfidf_matrix_test = tfidf_matrix.transform(xtest)
    charvec_test, _, _ = make_charvec(xtest.text_clean.tolist(), train=False, char_vocab=char_vocab, max_text_len=max_train_len_char)

    if lang != 'all':
        model = build_model(unigrams_shape, num_classes, charvec_shape, char_vocab_size)
    else:
        model = build_model(unigrams_shape, num_classes, charvec_shape, char_vocab_size)
    model.load_weights('{}.hdf5'.format('models/model_' + lang + '_weights'))

    predictions = model.predict([tfidf_matrix_test, charvec_test]).argmax(axis=-1)
    macro = str(f1_score(ytest, predictions, average='macro'))
    micro = str(f1_score(ytest, predictions, average='micro'))
    weighted = str(f1_score(ytest, predictions, average='weighted'))
    accuracy = str(accuracy_score(ytest, predictions))
    print('Test F1 macro:', macro)
    print('Test F1 micro:', micro)
    print('Test F1 weighted:', weighted)
    print('Test accuracy:', accuracy)
    print('Test confusion matrix:', confusion_matrix(ytest, predictions))


def test_all(data_test, target, drop, langs=['es','fa','fr','idmy','pt','slavic']):

    textmodel_data_all = pickle.load(open('models/model_all_data.pk', 'rb'))
    unigrams_shape, num_classes, charvec_shape, char_vocab_size, tfidf_matrix, char_vocab, max_train_len_char, group_tags_to_idx = textmodel_data_all
    xtest, ytest, _ = preprocess_data(data_test, target, drop, tags_to_idx=group_tags_to_idx)
    tfidf_matrix_test = tfidf_matrix.transform(xtest)

    charvec_test, _, _ = make_charvec(xtest.text_clean.tolist(), train=False, char_vocab=char_vocab, max_text_len=max_train_len_char)
    model = build_model(unigrams_shape, num_classes, charvec_shape, char_vocab_size)
    model.load_weights('{}.hdf5'.format('models/model_all_weights'))

    predictions = model.predict([tfidf_matrix_test, charvec_test]).argmax(axis=-1)
    print('Test F1 macro lang group:', f1_score(ytest, predictions, average='macro'))
    print('Test F1 micro lang group:', f1_score(ytest, predictions, average='micro'))
    print('Test F1 weighted lang group:', f1_score(ytest, predictions, average='weighted'))
    print('Test accuracy lang group:', accuracy_score(ytest, predictions))
    print('Test confusion matrix lang group:', confusion_matrix(ytest, predictions))

    df_predictions = pd.DataFrame({'lang_group_pred': predictions})
    xtest.reset_index(drop=True, inplace=True)
    df_true = pd.DataFrame({'lang_group': ytest})
    df_data = pd.concat([xtest, df_true, df_predictions], axis=1)

    K.clear_session()
    gc.collect()
    all_predictions = []
    for lang in langs:
        lang_idx = group_tags_to_idx.index(lang)
        filtered_data = df_data.loc[df_data['lang_group_pred'] == lang_idx]
        textmodel_data = pickle.load(open('models/model_' + lang + '_data.pk', 'rb'))
        unigrams_shape, num_classes, charvec_shape, char_vocab_size, tfidf_matrix, char_vocab, max_train_len_char, tags_to_idx = textmodel_data
        tfidf_matrix_test = tfidf_matrix.transform(filtered_data).toarray()
        charvec_test, _, _ = make_charvec(filtered_data.text_clean.tolist(), train=False, char_vocab=char_vocab, max_text_len=max_train_len_char)
        model = build_model(unigrams_shape, num_classes, charvec_shape, char_vocab_size)
        model.load_weights('{}.hdf5'.format('models/model_' + lang + '_weights'))
        predictions = model.predict([tfidf_matrix_test, charvec_test]).argmax(axis=-1)
        predictions = np.array([tags_to_idx[prediction] for prediction in predictions])
        df_predictions = pd.DataFrame({'predictions': predictions})
        df_predictions.reset_index(drop=True, inplace=True)
        ytest = filtered_data.variety
        df_ytest = pd.DataFrame({'y': ytest})
        df_ytest.reset_index(drop=True, inplace=True)
        results = pd.concat([df_ytest, df_predictions], axis=1)
        all_predictions.append(results)
    all_data = pd.concat(all_predictions, axis=0)
    all_y = all_data.y
    all_preds = all_data.predictions
    print('Test all macro F1 score:', f1_score(all_y, all_preds, average='macro'))
    print('Test all micro F1 score:', f1_score(all_y, all_preds, average='micro'))
    print('Test all weighted F1 score:', f1_score(all_y, all_preds, average='weighted'))
    print('Test all accuracy score:',  accuracy_score(all_y, all_preds))
    print('Test all confusion matrix score:', confusion_matrix(all_y, all_preds))












