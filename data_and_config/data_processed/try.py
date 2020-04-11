import pickle as pk
import numpy as np
import tensorflow as tf
import keras
import random
import json
from string import punctuation

# add_punc='，。、【 】 “”：；（）《》‘’{}？！⑦()、%^>℃：.”“^-——=&#@￥'
# all_punc = punctuation + add_punc
# file_list = ['train', 'valid', 'test']
#
# def punc_delete(fact_list):
#     fact_filtered = []
#     for word in fact_list:
#         fact_filtered.append(word)
#         if word in all_punc:
#             fact_filtered.remove(word)
#     return fact_filtered
#
#
# f_valid = pk.load(open('valid_processed_punc_delete.pkl', 'rb'))
# print(f_valid['term_lists'])

lidt = []

if len(lidt)==0:
    print(1)

# with open('../data/{}_cs.json'.format(file_list[0]), 'r', encoding='utf-8') as f:
#     for i in range(1):
#         line = json.loads(f.readline())
#         fact = line['fact_cut'].strip().split(' ')
#         print(fact)
#         fact = punc_delete(fact)
#         print(fact)