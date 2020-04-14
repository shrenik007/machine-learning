import os
from collections import Counter

import math


def get_words_from_document(path):
    f = open(path, "r", encoding="utf8")
    document_string = f.read()
    words = document_string.split()
    f.close()
    return words


def get_test_data(list_of_docs, testing_data_path, kwargs):
    test_data_result = []
    for file in list_of_docs:
        positive_words = get_words_from_document(testing_data_path + file)
        test_document_pos_probability = kwargs['positive_documents_prob']
        test_document_neg_probability = kwargs['negative_documents_prob']
        for word in positive_words:
            if word in kwargs['pos_conditional_prob_dict']:
                test_document_pos_probability += kwargs['pos_conditional_prob_dict'][word]
            if word in kwargs['neg_conditional_prob_dict']:
                test_document_neg_probability += kwargs['neg_conditional_prob_dict'][word]
        test_data_result.append(test_document_pos_probability < test_document_neg_probability)
    return test_data_result


def main_function():
    positive_training_data_path = 'data/train/pos/'
    negative_training_data_path = 'data/train/neg/'
    list_of_positive_docs = os.listdir(positive_training_data_path)
    list_of_negative_docs = os.listdir(negative_training_data_path)
    number_pos_docs = len(list_of_positive_docs)
    number_neg_docs = len(list_of_negative_docs)
    total_documents = number_pos_docs + number_neg_docs
    vocabulary_word_set = set()
    list_of_positive_words = []
    list_of_negative_words = []
    for file in list_of_positive_docs:
        positive_words = get_words_from_document(positive_training_data_path+file)
        list_of_positive_words.extend(positive_words)
        vocabulary_word_set.update(positive_words)
    for file in list_of_negative_docs:
        negative_words = get_words_from_document(negative_training_data_path+file)
        list_of_negative_words.extend(negative_words)
        vocabulary_word_set.update(negative_words)
    positive_frequency = dict.fromkeys(vocabulary_word_set, 0)
    negative_frequency = dict.fromkeys(vocabulary_word_set, 0)

    counter_positive_frequency = dict(Counter(list_of_positive_words))
    counter_negative_frequency = dict(Counter(list_of_negative_words))

    positive_frequency = {k: counter_positive_frequency[k] if k in counter_positive_frequency else 0 for k in positive_frequency}
    negative_frequency = {k: counter_negative_frequency[k] if k in counter_negative_frequency else 0 for k in negative_frequency}

    pos_conditional_prob_dict = {}
    neg_conditional_prob_dict = {}
    len_of_pos_words = len(list_of_positive_words)
    len_of_neg_words = len(list_of_negative_words)
    V = len(vocabulary_word_set)
    for key in vocabulary_word_set:
        if key == 'classic!)':
            print(key)
        pos_conditional_prob_dict[key] = get_conditional_probability(positive_frequency[key], len_of_pos_words, V)
        neg_conditional_prob_dict[key] = get_conditional_probability(negative_frequency[key], len_of_neg_words, V)

    positive_testing_data_path = 'data/test/pos/'
    negative_testing_data_path = 'data/test/neg/'

    list_of_positive_docs = os.listdir(positive_testing_data_path)
    list_of_negative_docs = os.listdir(negative_testing_data_path)

    positive_documents_prob = math.log(number_pos_docs / total_documents)
    negative_documents_prob = math.log(number_neg_docs / total_documents)
    positive_test_data_result = []
    negative_test_data_result = []

    list_of_docs = []
    kwargs_dict = {
        'positive_documents_prob': positive_documents_prob,
        'negative_documents_prob': negative_documents_prob,
        'pos_conditional_prob_dict': pos_conditional_prob_dict,
        'neg_conditional_prob_dict': neg_conditional_prob_dict,
    }
    positive_test_data_result = get_test_data(list_of_positive_docs, positive_testing_data_path, kwargs_dict)
    print(len(positive_test_data_result), positive_test_data_result.count(True), positive_test_data_result.count(False))

    negative_test_data_result = get_test_data(list_of_negative_docs, negative_testing_data_path, kwargs=kwargs_dict)
    print(len(negative_test_data_result), negative_test_data_result.count(True), negative_test_data_result.count(False))


def get_conditional_probability(value, len_of_pos_words, V):
    return math.log((value + 1) / (len_of_pos_words + V))


if __name__ == '__main__':
    main_function()
