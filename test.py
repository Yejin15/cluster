import json
import sklearn.cluster as clu
import pandas as pd
import time
start = time.time()
cluster = 10

with open('/Users/maeg/PycharmProjects/fisrtApp/Total_labels/long_sample.json') as json_file:
    json_data = json.load(json_file)
    print('loaded')
    print(time.time() - start)
    X = pd.DataFrame.from_records(json_data)
    print('1')
    print(time.time() - start)
    X = pd.DataFrame.transpose(X)
    print('2')
    print(time.time() - start)
    X = pd.DataFrame.drop(X, ['name'])
    print('3')
    print(time.time() - start)
    X = pd.DataFrame.fillna(X, value=0)
    X = X.astype('bool')
    print("before clustering")
    print(time.time() - start)

    print(X)

    kmeans = clu.k_means(X, n_clusters=cluster)

    result=pd.DataFrame(X[0])

    result[0] = kmeans[1]
    print(result)
    result.to_csv('/Users/maeg/PycharmProjects/fisrtApp/result/test.csv')
    print(time.time() - start)