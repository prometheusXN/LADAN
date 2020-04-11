import json
import pickle
import numpy as np



def word2id_generator(data_list):
    word_dict = {}
    w2id = {"BLANK": 0}
    for i in range(len(data_list)):
        with open('{}_cs.json'.format(data_list[i]), 'r', encoding='utf-8') as f:
            for line in f:
                example = json.loads(line)
                example_fact = example['fact_cut'].split()
                for j in example_fact:
                    if word_dict.__contains__(j):
                        word_dict[j] += 1
                    else:
                        word_dict.update({j: 1})
                    # print(example_fact)
            f.close()
            print('{} dataset is read over'.format(data_list[i]))
            print(len(word_dict))

    word_dict, loss_dict = filter_dict(word_dict)

    i = 1
    for k, v in word_dict.items():
        w2id.update({k: i})
        i += 1

    print(w2id)
    print(len(w2id))
    w2id.update({"UNK": i})
    return w2id


def filter_dict(data_dict):
    return {k: v for k, v in data_dict.items() if v >= 25}, {k: v for k, v in data_dict.items() if v < 25}


if __name__ == '__main__':

    datalist = ['train', 'valid']
    word_2id_file = "word2id_frequency25_train_valid.pkl"

    out_file = open(word_2id_file , 'wb')
    word2id = word2id_generator(datalist)
    pickle.dump(word2id, out_file)

    # word2id_dict = pickle.loads(open(word_2id_file, 'rb'))


