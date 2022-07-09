import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# numpy euclidian distance
from numpy.linalg import norm
from math import *

class KNN():
    def __init__(self):
        self.k = 1
    
    def euclidian_distance(self, a, b):
        return sqrt(sum(pow(a-b, 2) for a, b in zip(a, b)))

    def fit(self, x_train, x_test, y_train, y_test, epochs):
        self.x_train, self.y_train = x_train, y_train

        k = []
        for i in range(epochs):
            error = []
            for test_index, test in enumerate(x_test):
                distances = []
                for train in self.x_train:
                    dist = self.euclidian_distance(test, train)
                    distances.append(dist)

                dists = pd.DataFrame(distances, columns = ['dist'], index = self.y_train.index)
                sortedDist = dists.sort_values(by=['dist'], axis=0)[:self.k]

                classes = []
                for index, c in enumerate(pd.unique(y_test)):
                    classes.append(0)
                    for j in sortedDist.index:
                        if self.y_train[j] == c:
                            classes[index] += 1

                pred = np.argmax(classes, axis=-1)
                error.append(0 if pred == y_test.iloc[[test_index]].values[0] else 1)

            k.append(np.mean(error))
            self.k += 1
        
        self.k = np.argmin(k, axis=-1) + 1
        plt.plot(k)
        print(f"best k = {self.k}")
