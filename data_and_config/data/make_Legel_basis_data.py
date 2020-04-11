import thulac
import jieba
import json
import pickle as pk
import numpy as np
from string import punctuation

add_punc='，。、【 】 “”：；（）《》‘’{}？！⑦()、%^>℃：.”“^-——=&#@￥'
all_punc = punctuation + add_punc
doc_len = 15
sent_len = 100


def punc_delete(fact_list):
    fact_filtered = []
    for word in fact_list:
        fact_filtered.append(word)
        if word in all_punc:
            fact_filtered.remove(word)
    return fact_filtered


def hanzi_to_num(hanzi_1):
    # for num<10000
    hanzi = hanzi_1.strip().replace('零', '')
    if hanzi == '':
        return str(int(0))
    d = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '': 0}
    m = {'十': 1e1, '百': 1e2, '千': 1e3, }
    w = {'万': 1e4, '亿': 1e8}
    res = 0
    tmp = 0
    thou = 0
    for i in hanzi:
        if i not in d.keys() and i not in m.keys() and i not in w.keys():
            return hanzi

    if (hanzi[0]) == '十': hanzi = '一' + hanzi
    for i in range(len(hanzi)):
        if hanzi[i] in d:
            tmp += d[hanzi[i]]
        elif hanzi[i] in m:
            tmp *= m[hanzi[i]]
            res += tmp
            tmp = 0
        else:
            thou += (res + tmp) * w[hanzi[i]]
            tmp = 0
            res = 0
    return int(thou + res + tmp)


def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords


def get_cutter(dict_path="../law_processed/Thuocl_seg.txt", mode='thulac', stop_words_filtered=True):
    if stop_words_filtered:
        stopwords = stopwordslist('../law_processed/stop_word.txt')  # 这里加载停用词的路径
    else:
        stopwords = []
    if mode == 'jieba':
        jieba.load_userdict(dict_path)
        return lambda x: [a for a in list(jieba.cut(x)) if a not in stopwords]
    elif mode == 'thulac':
        thu = thulac.thulac(user_dict=dict_path, seg_only=True)
        return lambda x: [a for a in thu.cut(x, text=True).split(' ') if a not in stopwords]


def seg_sentence(sentence, cut):
    # cut=get_cutter()
    # sentence_seged = thu.cut(sentence.strip(), text=True).split(' ')
    sentence_seged = cut(sentence)
    # print(sentence_seged)
    outstr = []
    for word in sentence_seged:
        if word != '\t':
            word = str(hanzi_to_num(word))
            outstr.append(word)
            # outstr += " "
    return outstr


def lookup_index_for_sentences(sentences, word2id, doc_len, sent_len):
    item_num = 0
    res = []
    if len(sentences) == 0:
        tmp = [word2id['BLANK']] * sent_len
        res.append(np.array(tmp))
    else:
        for sent in sentences:
            sent = punc_delete(sent)
            tmp = [word2id['BLANK']] * sent_len
            for i in range(len(sent)):
                if i >= sent_len:
                    break
                try:
                    tmp[i] = word2id[sent[i]]
                    item_num += 1
                except KeyError:
                    tmp[i] = word2id['UNK']

            res.append(np.array(tmp))
    if len(res) < doc_len:
        res = np.concatenate([np.array(res), word2id['BLANK'] * np.ones([doc_len - len(res), sent_len], dtype=np.int)], 0)
    else:
        res = np.array(res[:doc_len])

    return res, item_num


def sentence2index_matrix(sentence, word2id, doc_len, sent_len, cut):
    sentence = sentence.replace(' ', '')
    sent_words, sent_n_words = [], []
    for i in sentence.split('。'):
        if i != '':
            sent_words.append((seg_sentence(i, cut)))
    index_matrix, item_num = lookup_index_for_sentences(sent_words, word2id, doc_len, sent_len)
    return index_matrix, item_num, sent_words


with open('../data/w2id_thulac.pkl', 'rb') as f:
    word2id_dict = pk.load(f)
    f.close()

file_list = ['train', 'valid', 'test']
cut = get_cutter(stop_words_filtered= False)

for i in range(len(file_list)):
    fact_lists = []
    law_label_lists = []
    accu_label_lists = []
    term_lists = []
    num = 0

    with open('../data/{}_cs.json'.format(file_list[i]), 'r', encoding= 'utf-8') as f:
        idx = 0
        for line in f.readlines():
            idx += 1
            line = json.loads(line)
            fact = line['fact_cut']
            sentence, word_num, sent_words = sentence2index_matrix(fact, word2id_dict, doc_len, sent_len, cut)

            if word_num <= 10:
                print(fact)
                print(sent_words)
                print(idx)
                continue

            fact_lists.append(sentence)
            law_label_lists.append(line['law'])
            accu_label_lists.append(line['accu'])
            term_lists.append(line['term'])
            num += 1
        f.close()
    data_dict = {'fact_list': fact_lists, 'law_label_lists': law_label_lists, 'accu_label_lists': accu_label_lists, 'term_lists': term_lists}
    pk.dump(data_dict, open('../legal_basis_data/{}_processed_thulac_Legal_basis.pkl'.format(file_list[i]), 'wb'))
    print(num)
    print('{}_dataset is processed over'.format(file_list[i])+'\n')