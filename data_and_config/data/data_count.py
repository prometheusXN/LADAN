import numpy as np
import json,re,pickle
import tensorflow as tf
from collections import Iterable


prefix = ['train','valid','test']
# with open('exercise_contest/data_{}.json'.format(prefix[0]), 'r', encoding='utf-8') as f:
#     example = json.loads(f.readline()) # keys = ['fact', 'meta'['relaevant_articles', 'accusation', 'term_of_imprisonment']]
#     example_articles = example['meta']['relevant_articles']
#     example_accusation = example['meta']['accusation']
#     print(len(example_articles))
#     print(len(example_accusation))

dict_articles = {}
dict_accusation = {}
for i in range(len(prefix)):
    with open('data_{}.json'.format(prefix[i]), 'r', encoding='utf-8') as f:
        for line in f:
            example = json.loads(line)
            example_articles = example['meta']['relevant_articles']
            example_accusation = example['meta']['accusation']
            example_fact = example['fact']
            if len(example_articles) == 1 and len(example_accusation) == 1 and '二审' not in example_fact:
                if dict_articles.__contains__(example_articles[0]):
                    dict_articles[example_articles[0]] += 1
                else:
                    dict_articles.update({example_articles[0]: 1})

                if dict_accusation.__contains__(example_accusation[0]):
                    dict_accusation[example_accusation[0]] += 1
                else:
                    dict_accusation.update({example_accusation[0]: 1})
        f.close()
    print('The {} dataset is read over'.format(prefix[i]))

print(dict_articles)
print(dict_accusation)


def filter_dict(data_dict):
    return {k: v for k, v in data_dict.items() if v >= 100}

def sum_dict(data_dict):
    sum = 0
    for k,v in data_dict.items():
        sum+= v
    return sum

print(sum_dict(dict_articles))
print(sum_dict(dict_accusation))

dict_articles = filter_dict(dict_articles)
dict_accusation= filter_dict(dict_accusation)

articles_sum = sum_dict(dict_articles)
accusation_sum = sum_dict(dict_accusation)

print('\n')
print(dict_articles)
print('articles_num: '+ str(len(dict_articles)))
print('article_sum: ' + str(articles_sum))

print(dict_accusation)
print('accusation_num='+ str(len(dict_accusation)))
print('accusation_sum: ' + str(accusation_sum))
print('\n')

def reset_dict(data_dict):
    return {k: 0 for k, v in data_dict.items()}

while articles_sum!=accusation_sum:
    dict_accusation = reset_dict(dict_accusation)
    dict_articles = reset_dict(dict_articles)
    print(dict_articles)
    print(dict_accusation)
    for i in range(len(prefix)):
        with open('data_{}.json'.format(prefix[i]), 'r', encoding='utf-8') as f:
            for line in f:
                example = json.loads(line)
                example_articles = example['meta']['relevant_articles']
                example_accusation = example['meta']['accusation']
                if len(example_articles) == 1 and len(example_accusation) == 1 and '二审' not in example_fact:
                    if dict_articles.__contains__(example_articles[0]) and dict_accusation.__contains__(example_accusation[0]):
                        dict_articles[example_articles[0]] += 1
                        dict_accusation[example_accusation[0]] += 1
                    else:
                        continue
            f.close()
        print('The {} dataset is read over'.format(prefix[i]))

    print(dict_articles)
    print(dict_accusation)

    print(len(dict_articles))
    print(len(dict_accusation))

    dict_articles = filter_dict(dict_articles)
    dict_accusation = filter_dict(dict_accusation)

    articles_sum = sum_dict(dict_articles)
    accusation_sum = sum_dict(dict_accusation)

    print('\n')
    print(dict_articles)
    print('articles_num: '+ str(len(dict_articles)))
    print('article_sum: ' + str(articles_sum))

    print(dict_accusation)
    print('accusation_num='+ str(len(dict_accusation)))
    print('accusation_sum: ' + str(accusation_sum))


print('\n')
print('final counter:')
print(dict_articles)
print('articles_num: ' + str(len(dict_articles)))
print('article_sum: ' + str(articles_sum))

print(dict_accusation)
print('accusation_num=' + str(len(dict_accusation)))
print('accusation_sum: ' + str(accusation_sum))