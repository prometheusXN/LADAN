import pickle as pk
import numpy as np
import json
from string import punctuation

add_punc='，。、【 】 “”：；（）《》‘’{}？！⑦()、%^>℃：.”“^-——=&#@￥'
all_punc = punctuation + add_punc
max_length = 512

stop_word_file = ''


def punc_delete(fact_list):
    fact_filtered = []
    for word in fact_list:
        fact_filtered.append(word)
        if word in all_punc:
            fact_filtered.remove(word)
    return fact_filtered


with open('../data/w2id_thulac.pkl', 'rb') as f:
    word2id_dict = pk.load(f)
    f.close()

# print(word2id_dict)

file_list = ['train', 'valid', 'test']


for i in range(len(file_list)):
    fact_lists = []
    law_label_lists = []
    accu_label_lists = []
    term_lists = []
    num = 0
    with open('../data/{}_cs.json'.format(file_list[i]), 'r', encoding='utf-8') as f:
        idx = 0
        for line in f.readlines():
            idx += 1
            line = json.loads(line)
            fact = line['fact_cut'].strip().split(' ')
            fact = punc_delete(fact)
            id_list = []
            word_num = 0
            for j in range(int(min(len(fact), max_length))):
                if fact[j] in word2id_dict:
                    id_list.append(int(word2id_dict[fact[j]]))
                    word_num += 1
                else:
                    id_list.append(int(word2id_dict['UNK']))
            while len(id_list) < 512:
                id_list.append(int(word2id_dict['BLANK']))

            if word_num <= 10:
                print(fact)
                print(idx)
                continue

            id_numpy = np.array(id_list)

            fact_lists.append(id_numpy)
            law_label_lists.append(line['law'])
            accu_label_lists.append(line['accu'])
            term_lists.append(line['term'])
            num+=1
        f.close()
    data_dict = {'fact_list': fact_lists, 'law_label_lists': law_label_lists, 'accu_label_lists': accu_label_lists, 'term_lists': term_lists}
    pk.dump(data_dict, open('{}_processed_thulac.pkl'.format(file_list[i]), 'wb'))
    print(num)
    print('{}_dataset is processed over'.format(file_list[i])+'\n')