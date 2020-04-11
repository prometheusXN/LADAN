import pickle as pk
import numpy as np

with open('train_processed_thulac.pkl', 'rb') as f:
    train_data_dict = pk.load(f)
    fact_lists = train_data_dict['fact_list']
    law_label_lists = train_data_dict['law_label_lists']
    accu_label_lists = train_data_dict['accu_label_lists']
    term_lists = train_data_dict['term_lists']
    f.close()

max_count = 3758

print(len(fact_lists))

count_num = [0 for i in range(103)]
sample_list = [[] for j in range(103)]

print(count_num)
for i in range(103):
    for j in range(len(fact_lists)):
        data_list = [fact_lists[j], law_label_lists[j], accu_label_lists[j], term_lists[j]]
        if law_label_lists[j] == i:
            count_num[i] += 1
            sample_list[i].append(data_list)
print(count_num)
print(len(sample_list))
print(len(sample_list[0]))
for i in range(len(sample_list)):
    if len(sample_list[i]) < max_count:
        epoch = int(max_count/len(sample_list[i]))
        addition = max_count - epoch * len(sample_list[i])
        list = []
        for j in range(epoch):
            list += sample_list[i]
        list += sample_list[i][:addition]
        sample_list[i] = list
        # print(len(sample_list[i]))

fact_lists, law_label_lists, accu_label_lists, term_lists = [], [], [], []

for i in range(len(sample_list)):
    for j in range(len(sample_list[i])):
        fact, law_label, accu_label, term = sample_list[i][j]
        fact_lists.append(fact)
        law_label_lists.append(law_label)
        accu_label_lists.append(accu_label)
        term_lists.append(term)

print(len(fact_lists))

data_dict = {'fact_list': fact_lists, 'law_label_lists': law_label_lists, 'accu_label_lists': accu_label_lists, 'term_lists': term_lists}

pk.dump(data_dict, open('train_processed_thulac_fit.pkl', 'wb'))

