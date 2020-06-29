#!/usr/bin/env python3

from collections import Counter

from lib.ml_math import tfidf
from lib.utils import timing
from config import min_word_appearances, verbose, amount_of_words_to_keep, index_column_name


def get_words_under_appearances(
        dword_universe,
        min_word_appearances=min_word_appearances):

    result = [word for word, appearances in dword_universe.items()
              if appearances < min_word_appearances]

    return result


def remove_words_under_X_appearances(master_db, dword_universe, min_word_appearances=min_word_appearances):

    print('Removing words appearing less than ', min_word_appearances, ' times.')
    print('Initial dword_universe words: ', len(dword_universe))
    words_to_remove = get_words_under_appearances(
        dword_universe, min_word_appearances)

    print('Removing ', len(words_to_remove), ' words from master_db and dword_universe.')
    for pos, word in enumerate(words_to_remove):
        print("\r\t> Progress\t:{:.2%}".format((pos)/len(words_to_remove)), end='', flush=True)
        del dword_universe[word]
        for item in master_db:
            item.pop(word, None)

    return


def _get_keys_master_db(master_db):
    keys = []
    for report in master_db:
        keys += list(report.keys())

    return list(set(keys))


@timing
def get_dword_universe(master_db):
    if verbose:
        print("Creating dword universe..")
    result = {}

    keys = _get_keys_master_db(master_db)
    result = {key: 0 for key in keys if key != index_column_name}

    for pos, report in enumerate(master_db):
        print("\r\t> Progress\t:{:.2%}".format((pos)/len(master_db)), end='', flush=True)

        for word in report:
            if word != index_column_name:
                result[word] += report[word]

    return result


def _amount_of_samples_w_word(reports_db, word):

    amount_of_samples = \
        len([report for report in reports_db
             if word in report.keys()
             ])
    return amount_of_samples


def get_words_tfidf(dword, reports_db, amount_of_samples=amount_of_words_to_keep):
    '''Returns the words_tfidf dict, containing the tfidf value for each word.
    '''

    words_in_the_document = sum(dword.values())
    words_tfidf = {}
    for word, repetitions in dword.items():
        num_samples_with_this_word = \
            _amount_of_samples_w_word(
                reports_db, word)

        words_tfidf[word] = tfidf(word, repetitions,
                                  words_in_the_document,
                                  amount_of_samples,
                                  num_samples_with_this_word)

    return words_tfidf


def sorted_dict_by_value(dict_to_sort, reverse=True):
    '''Sort a dictionare by its value. Yes, it's possible.

    reverse=True, from top to 0.
            False, from 0 to top.
    '''
    return {k: v for k, v in sorted(dict_to_sort.items(),
                                    key=lambda item: item[1],
                                    reverse=reverse)}


def _get_X_best_words(words_tfidf, amount_of_best_words):

    sorted_dict = sorted_dict_by_value(words_tfidf)

    best_words = list(sorted_dict.keys())[:amount_of_best_words]
    worst_words = list(set(sorted_dict.keys()) - set(best_words))

    return best_words, worst_words


def get_best_words(words_tfidf, reports_db,
                   dword_universe,
                   words_to_keep=amount_of_words_to_keep):
    ''' Remove all but the best words_to_keep words from the dataset
    '''

    best_words, worst_words = _get_X_best_words(words_tfidf, amount_of_words_to_keep)
    if verbose:
        print("Best words: " + str(len(best_words)) + ". Worst words: " + str(len(worst_words)))

    new_dword_universe = {word: dword_universe[word] for word in best_words}

    new_master_db = {
        report['file_name']: {
            word: report[word]
            for word in set(best_words) & set(report.keys())
        }
        for report in reports_db
    }

    return new_master_db, new_dword_universe
