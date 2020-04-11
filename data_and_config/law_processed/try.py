import pickle as pk

law_file_order = pk.load(open('../law_processed/law_label2index.pkl', 'rb'))
print(law_file_order)