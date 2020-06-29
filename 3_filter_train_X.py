#!/usr/bin/env python3

import pandas as pd
from sklearn.feature_selection import SelectKBest, SelectPercentile, chi2, f_classif
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import VarianceThreshold

from config import dword_universe_filepath, train_master_db_filepath, train_X_filepath, words_tfidf_filename, dword_universe_flatten_filepath, test_master_db_filepath, avclass_train_y_filepath, avclass_test_y_filepath, index_column_name, df_splitted_filepath, maximum_amount_of_tfidf_features, min_tfidf_value

from lib.loader import load_obj, save_obj
from lib.dataset_utils import get_words_tfidf, get_best_words, remove_words_under_X_appearances
from lib.dataframe_handler import remove_duplicated_columns, clean_dword_universe

import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


def get_filter_on_dict(test_master_db, words):

    filtered_db = []
    words.append('file_name')
    for report in test_master_db:
        new_report = {}
        for word in words:
            new_report[word] = report.get(word) or 0

        filtered_db.append(new_report)

    return filtered_db


def get_keys_universe(test_master_db):
    result = []
    for report in test_master_db:
        result += report.keys()
    return set(result)


def get_test_X(test_master_db, train_X):
    test_X_words = get_keys_universe(test_master_db)
    test_master_db = get_filter_on_dict(test_master_db, list(train_X.columns))
    test_X = pd.DataFrame(test_master_db).set_index('file_name')
    test_X = test_X.reindex(columns=train_X.columns.tolist(), fill_value=0)

    return test_X


def split_X(train_X, test_X, train_y, test_y):

    data = {
        'train': {
            'X': train_X,  # .fillna(0),
            'y': train_y,  # .fillna(0)
        },
        'test': {
            'X': test_X,  # .fillna(0),
            'y': test_y  # , #.fillna(0)
        }
    }

    return data


def best_X_tfidf_features(df_tfidf, min_tfidf_value=min_tfidf_value, maximum_amount_of_tfidf_features=maximum_amount_of_tfidf_features):
    features_above_threshold = df_tfidf.max()[df_tfidf.max(
    ) > min_tfidf_value].sort_values(ascending=False)

    return features_above_threshold[:maximum_amount_of_tfidf_features]


def get_df_tfidf(train_X, train_y):
    tfidf_t = TfidfTransformer()
    tfidf_t.fit(train_X, train_y)
    df_tfidf = tfidf_t.transform(train_X, train_y)
    df_tfidf = pd.DataFrame(df_tfidf.toarray(), columns=list(
        train_X.columns.values.tolist()), index=train_X.index)
    return tfidf_t, df_tfidf


if __name__ == '__main__':

    train_master_db = load_obj(train_master_db_filepath)
    test_master_db = load_obj(test_master_db_filepath)
    train_y = load_obj(avclass_train_y_filepath)
    test_y = load_obj(avclass_test_y_filepath)

    #################################################
    # Train_X - From Dictionare to DataFrame        #
    #################################################
    train_X = pd.DataFrame(train_master_db)
    train_X = train_X.set_index('file_name')
    train_X = train_X.fillna(0)
    train_master_db = None

    #################################################
    # Test_X
    test_X = get_test_X(test_master_db, train_X)
    test_master_db = None
    # train_X
    train_X = train_X[~train_X.index.isin(test_X.index)]

    #################################################
    # Train_X - Filter 2: Remove duplicated columns #
    #################################################
    train_X = remove_duplicated_columns(train_X)

    ################################################
    # Filter 3: Chi^2
    ################################################
    percentile = 50
    SelectPercentile(chi2, percentile=percentile).fit_transform(train_X, train_y)

    #################################################
    # Train_X - Filter 4: TFIDF filter              #
    #################################################
    tfidf_transformer, df_tfidf = get_df_tfidf(train_X, train_y)
    best_tfidf_features = best_X_tfidf_features(tfidf_transformer, df_tfidf)
    train_X = train_X[best_tfidf_features.index]

    ##########
    test_X = test_X[train_X.columns]

    #######################################################
    df_splitted = split_X(train_X, test_X, train_y, test_y)

    save_obj(train_X, train_X_filepath)
    save_obj(df_splitted, df_splitted_filepath)

    exit(0)
