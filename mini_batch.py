import json
import sklearn.cluster as clu
import pandas as pd
import time
start = time.time()
cluster = 10

with open('/Users/maeg/PycharmProjects/fisrtApp/Total_labels/Total_labels_short.json') as json_file:
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

    model = clu.SpectralClustering(n_clusters=cluster).fit(X)

    result = pd.DataFrame(X[0])
    result[0] = model.labels_
    print(result)
    result.to_csv('/Users/maeg/PycharmProjects/fisrtApp/result/spectral.csv')

    model = clu.affinity_propagation(n_clusters=cluster).fit(X)

    result = pd.DataFrame(X[0])
    result[0] = model.labels_
    print(result)
    result.to_csv('/Users/maeg/PycharmProjects/fisrtApp/result/affinity.csv')

    model = clu.linkage_tree(X, n_clusters=cluster)

    result = pd.DataFrame(X[0])
    result[0] = model.labels_
    print(result)
    result.to_csv('/Users/maeg/PycharmProjects/fisrtApp/result/linkage.csv')

    model = clu.MiniBatchKMeans( n_clusters=cluster).fit(X)

    result = pd.DataFrame(X[0])
    result[0] = model.labels_
    print(result)
    result.to_csv('/Users/maeg/PycharmProjects/fisrtApp/result/Minibatch.csv')