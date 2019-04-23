import json
import sklearn.cluster as clu
import numpy as np
import pandas as pd

cluster = 4

with open('/Users/maeg/PycharmProjects/fisrtApp/Total_labels/long_sample.json') as json_file:
    json_data = json.load(json_file)
    X = pd.DataFrame.from_records(json_data)
    X = pd.DataFrame.transpose(X)
    X = pd.DataFrame.drop(X, ['name'])
    X = pd.DataFrame.fillna(X, value=0)
    X_out = pd.DataFrame.to_string(X)

    model = clu.spectral_clustering(n_clusters=cluster).fit(X)

    result = pd.DataFrame(X[0])
    result[0] = model.labels_
    print(result)
    result.to_csv('/Users/maeg/PycharmProjects/fisrtApp/result/spec_sample.csv')

