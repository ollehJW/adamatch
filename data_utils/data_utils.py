# %%
import numpy as np
import pandas as pd
import os
import random
from collections import defaultdict

# %%
def prepare_data(train_path, remove_filename_list=[]):
    if 'csv' == train_path.split('.')[-1]:
        files_df = pd.read_csv(train_path)
        files = files_df['path'].to_list()
        labels = files_df['new_label'].to_list()
    else:
        files = []
        for ty in os.listdir(train_path):
            filelist = os.listdir(os.path.join(train_path, ty))
            for i, file in enumerate(filelist):
                if file.endswith('.wav') | file.endswith('.tdms'):
                    if file not in remove_filename_list:
                        files.append(os.path.join(train_path, ty, file))

        labels = [file.split('/')[-2] for file in files]

    uni_label = np.unique(labels)
    print(uni_label)
    y = np.array([np.eye(len(uni_label))[np.where(uni_label==label)].reshape(-1) for label in labels])
    print(y.shape)

    return files, uni_label, y

def make_mixup_fileset(file_name_list, category_list, file_label_list, seed, mixup = True):
    np.random.seed(seed)
    
    category_index = defaultdict(list)
    for cat in category_list:
        category_index[cat] = list(np.where(file_label_list[:, list(category_list).index(cat)] == 1)[0])

    mix_up_set = list(range(len(file_name_list)))
    if mixup:    
        for cat in category_list[:-1]:
            if len(category_index[cat]) > 0:
                aug_num = len(category_index['OK']) - len(category_index[cat])
                OK_sample = np.random.choice(category_index['OK'],aug_num, replace=False)
                NG_sample = np.random.choice(category_index[cat],aug_num, replace=True)
                for i in range(aug_num):
                    mix_up_set.append([NG_sample[i], OK_sample[i]])

    random.shuffle(mix_up_set)
    return mix_up_set

