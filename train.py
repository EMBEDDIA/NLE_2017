# -*- coding: utf-8 -*-

import argparse
import time
import pandas as pd
from language_variety import train, test_trained_model, test_all, preprocess_data


def read_dslccv40_corpus(input, directory, name):
    data = [['text', 'variety']]
    with open(input, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if len(line.split('\t')) != 2:
                print(line)
            text, c = line.split('\t')
            data.append([text, c])
    headers = data.pop(0)
    df = pd.DataFrame(data, columns=headers)
    df.to_csv(directory + name +'.csv', encoding="utf8", sep='\t', index=False)
    return df


def read_adi_corpus(input):
    data = [['text', 'variety']]
    with open(input, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            text, q, c = line.split('\t')
            data.append([text, c])
    headers = data.pop(0)
    df = pd.DataFrame(data, columns=headers)
    return df


def read_corpus(input):
    data = [['text', 'variety']]
    with open(input, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            text, c = line.split('\t')
            data.append([text, c])
    headers = data.pop(0)
    df = pd.DataFrame(data, columns=headers)
    return df


def split_dataset(df_data, output, name='test'):
    groups = {'slavic':['hr', 'bs', 'sr'], 'idmy': ['id', 'my'], 'pt': ['pt-PT', 'pt-BR'], 'es': ['es-AR', 'es-ES', 'es-PE'], 'fa': ['fa-IR', 'fa-AF'], 'fr': ['fr-FR', 'fr-CA']}
    for group, langs in groups.items():
        print('Generating ' + name + ' file for ' + group + ' language group, containing following labels: ', langs)
        filtered_data = df_data.loc[df_data['variety'].isin(langs)]
        print('Dataset size: ', filtered_data.shape[0])
        filtered_data.to_csv(output + group + '_' + name + '.csv', encoding='utf8', sep='\t', index=False)


def add_language_group(df_data):
    groups = {'hr':'slavic', 'bs': 'slavic', 'sr':'slavic', 'id':'idmy', 'my':'idmy', 'pt-PT':'pt', 'pt-BR':'pt',
              'es-AR':'es', 'es-ES':'es', 'es-PE':'es', 'fa-IR':'fa', 'fa-AF':'fa', 'fr-FR':'fr', 'fr-CA':'fr'}
    df_data['lang_group'] = df_data['variety'].map(lambda x: groups[x])
    return df_data



if __name__ == '__main__':
    start_time = time.time()
    # run from command line
    # e.g. python3 train.py --dslcc

    argparser = argparse.ArgumentParser(description='Language variety classification')

    argparser.add_argument('-d', '--data_directory', type=str,
                           default='data/dslccv4.0/',
                           help='Data directory')
    argparser.add_argument('-x', '--train_corpus', type=str,
                           default='data/dslccv4.0/DSL-TRAIN.txt',
                           help='Path to train corpus - first column should be text, second a label. Columns should be separated by tab.')
    argparser.add_argument('-z', '--dev_corpus', type=str,
                           default='data/dslccv4.0/DSL-DEV.txt',
                           help='Path to development corpus - first column should be text, second a label. Columns should be separated by tab.')
    argparser.add_argument('-y', '--test_corpus', type=str,
                           default='data/dslccv4.0/DSL-TEST-GOLD.txt',
                           help='Path to test corpus - first column should be text, second a label. Columns should be separated by tab.')
    argparser.add_argument('-w', '--weighting', type=str,
                           default='tfidf',
                           help='Weighting scheme. Can either be "tfidf" or "bm25".')
    argparser.add_argument('-e', '--experiment', type=str,
                           default='DSLCC',
                           help='Default is "DSLCC". Other allowed values are: "GDIC", "ADIC" and "OTH" (for custom datasets).')


    args = argparser.parse_args()
    train_data_all = args.train_corpus
    dev_data_all = args.dev_corpus
    test_data_all = args.test_corpus
    directory = args.data_directory
    experiment = args.experiment
    weighting = args.weighting

    assert experiment in ['DSLCC', 'ADIC', 'GDIC', 'OTH'], "Invalid experiemnt argument!"
    assert weighting in ['tfidf', 'bm25'], "Invalid weighting argument!"

    if experiment == 'DSLCC':

        train_data_all = read_dslccv40_corpus('data/dslccv4.0/DSL-TRAIN.txt', 'data/dslccv4.0/', name='train')
        split_dataset(train_data_all, directory, name='train')

        dev_data_all = read_dslccv40_corpus('data/dslccv4.0/DSL-DEV.txt', 'data/dslccv4.0/', name='dev')
        split_dataset(dev_data_all, directory, name='dev')

        test_data_all = read_dslccv40_corpus('data/dslccv4.0/DSL-TEST-GOLD.txt', 'data/dslccv4.0/', name='test')
        split_dataset(test_data_all, directory, name='test')

        langs = ['slavic', 'es', 'fa', 'fr','idmy','pt']
        for lang in langs:
            print()
            print('Training model for ' + lang)
            print()

            train_file = lang + '_train.csv'
            dev_file = lang + '_dev.csv'
            test_file = lang + '_test.csv'
            target = 'variety'
            drop = []

            data_train = pd.read_csv(directory + train_file, encoding='utf8', sep='\t')
            data_val = pd.read_csv(directory + dev_file, encoding='utf8', sep='\t')
            data_test = pd.read_csv(directory + test_file, encoding='utf8', sep='\t')
            #print(data_test.shape, data_train.columns)

            data_train = data_train.sample(frac=1, random_state=1)

            xtrain, ytrain, tags_to_idx = preprocess_data(data_train, target, drop)
            xval, yval, _ = preprocess_data(data_val, target, drop, tags_to_idx=tags_to_idx)
            xtest, ytest, _ = preprocess_data(data_test, target, drop, tags_to_idx=tags_to_idx)

            train(xtrain, ytrain, xval, yval, lang=lang, tags_to_idx=tags_to_idx, weighting=weighting)

            print()
            print('Testing model for ' + lang)
            print()

            test_trained_model(data_test, target, drop, lang, weighting=weighting)

        print()
        print('Training general model')
        print()

        lang = 'all'
        target = 'lang_group'
        drop = []

        data_train = add_language_group(train_data_all)
        data_val = add_language_group(dev_data_all)
        data_test = add_language_group(test_data_all)

        #shuffle corpus
        data_train = data_train.sample(frac=1, random_state=1)

        xtrain, ytrain, tags_to_idx = preprocess_data(data_train, target, drop)
        xval, yval, _ = preprocess_data(data_val, target, drop, tags_to_idx=tags_to_idx)
        xtest, ytest, _ = preprocess_data(data_test, target, drop, tags_to_idx=tags_to_idx)

        train(xtrain, ytrain, xval, yval, lang=lang, tags_to_idx=tags_to_idx, weighting=weighting)

        print()
        print('Testing general model')
        print()

        test_trained_model(data_test, target, drop, lang, weighting=weighting)

        print()
        print('Testing total 2 step system')
        print()

        test_all(data_test, target, drop, weighting=weighting)

    else:
        if experiment == 'ADIC':
            target = 'variety'
            lang = 'ar'
            drop = []

            data_train_valid = read_adi_corpus('data/adic/Q-train.txt')
            data_train_valid = data_train_valid.sample(frac=1)
            data_train = data_train_valid[:int(data_train_valid.shape[0] * 0.9)]
            data_val = data_train_valid[int(data_train_valid.shape[0] * 0.9):]
            data_test = read_adi_corpus('data/adic/Q-gold.txt')


        elif experiment == 'GDIC':
            target = 'variety'
            lang = 'sg'
            drop = []

            data_train = read_corpus('data/gdic/train.txt')
            data_valid = read_corpus('data/gdic/dev.txt')
            data_train_valid = pd.concat([data_train, data_valid])
            data_train_valid = data_train_valid.sample(frac=1)
            data_train = data_train_valid[:int(data_train_valid.shape[0] * 0.9)]
            data_val = data_train_valid[int(data_train_valid.shape[0] * 0.9):]
            data_test = read_corpus('data/gdic/gold.txt')
            data_test = data_test[data_test[target] != 'XY']

        elif experiment == 'OTH':
            target = 'variety'
            lang = 'oth'
            drop = []

            data_train = read_corpus(train_data_all)
            data_train = data_train.sample(frac=1)
            data_val = read_corpus(dev_data_all)
            data_test = read_corpus(test_data_all)


        xtrain, ytrain, tags_to_idx = preprocess_data(data_train, target, drop)
        xval, yval, _ = preprocess_data(data_val, target, drop, tags_to_idx=tags_to_idx)
        xtest, ytest, _ = preprocess_data(data_test, target, drop, tags_to_idx=tags_to_idx)

        train(xtrain, ytrain, xval, yval, lang=lang, tags_to_idx=tags_to_idx, weighting=weighting)

        print()
        print('Testing model for ' + lang)
        print()

        test_trained_model(data_test, target, drop, lang, weighting=weighting)


    print("--- Model creation in minutes ---", round(((time.time() - start_time) / 60), 2))
    print("--- Training & Testing in minutes ---", round(((time.time() - start_time) / 60), 2))





