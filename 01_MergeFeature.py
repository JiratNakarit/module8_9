import os
import pandas as pd
import numpy as np

data_path = os.path.abspath(os.path.join('data/feature'))

def files(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file

list_file = []
for file in files(data_path):
    list_file.append(file)

for i in range(len(list_file)):
    data_dir = os.path.abspath(os.path.join('data/feature'))
    data_path = os.path.join(data_dir, list_file[i])
    df = pd.read_csv(data_path)

    if i != 0:
        df_merge = pd.concat([df,df_complete])
        df_complete = df_merge
    else:
        df_complete = df

df_complete.to_csv('out/01_MergeFeature.csv', encoding='utf-8')
