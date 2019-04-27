import json
import sklearn.cluster as clu
import pandas as pd

cluster = 10

with open('/Users/maeg/PycharmProjects/fisrtApp/Total_labels/Total_labels_short.json') as json_file:
    json_data = json.load(json_file)
    X = pd.DataFrame.from_records(json_data)
    X = pd.DataFrame.transpose(X)
    X = pd.DataFrame.drop(X, ['name'])
    X = pd.DataFrame.fillna(X, value=0)
    X_out = pd.DataFrame.to_string(X)

    print("before clustering")

    kmeans = clu.k_means(X, n_clusters=cluster)

    result=pd.DataFrame(X[0])

    result[0] = kmeans[1]
    print(result)
    result.to_csv('/Users/maeg/PycharmProjects/fisrtApp/result/kmeans.csv')

