# coding=utf-8
import numpy as np

class DataSet(object):
    def __init__(self):
        self._ind_actual = 0

    def next_batch(self, dataSet, labels, batch_size):
        start = self._ind_actual
        self._ind_actual += batch_size
        if self._ind_actual > len(dataSet):
            dataSet, labels = random_shuffle_twoArrays(dataSet,labels)
            start = 0
            self._ind_actual = batch_size
        end = self._ind_actual
        return dataSet[start:end],labels[start:end]

def random_shuffle_twoArrays(array1, array2):
    aux = list(zip(array1, array2))
    np.random.shuffle(aux)
    a,b = zip(*aux)
    return a,b

def create_test_and_train_set(dataSet, labels):
    if dataSet.size != labels.size:
        return "Error! El tamaño no coincide"

    lenght = len(dataSet)
    train = int(lenght * 0.8)
    test = len(dataSet) - train

    data_train = np.zeros((train, (dataSet.size() / lenght)))
    labels_train = np.zeros((train, (labels.size() / lenght)))

    data_test = np.zeros((test, (dataSet.size() / lenght)))
    labels_test = np.zeros((test, (labels.size() / lenght)))

    j = 0
    for i in range(lenght):
        if i < train:
            data_train[i] = dataSet[i]
            labels_train[i] = labels[i]
        else:
            data_test[j] = dataSet[i]
            labels_test[j] = labels[i]
            j += 1

    return data_train, labels_train, data_test, labels_test




