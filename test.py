#!/bin/env python3

import csv
import numpy as np
import random
import sys

from bayes import BayesClassifier
from naivebayes import NaiveBayesClassifier

def get_class(weight, height):
    bmi = weight / (height*0.0254)**2
    if bmi < 19:
        return 'Underweight'
    elif bmi > 25:
        return 'Overweight'
    else:
        return 'Normal'

def generate_dataset(n=1000):
    dataset = []
    for i in range(n):
        age = random.randint(20, 40)
        weight = random.randint(35, 100)
        height = random.randint(58, 77)
        _class = get_class(weight, height)
        dataset.append([age, weight, height, _class])
    return dataset

def load_csv(filename):
    data = []
    with open(filename, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            for i in range(len(row)-1):
                row[i] = float(row[i])
            data.append(row)
        return data
    raise RuntimeError('invalid csv file')


def main():
    try:
        n = int(sys.argv[1])
    except IndexError:
        n = 1000
    #dataset = load_csv('data.csv')
    dataset = generate_dataset(n)

    bayes = BayesClassifier()
    naivebayes = NaiveBayesClassifier()

    bayes.train(dataset)
    naivebayes.train(dataset)

    m_bayes = m_naivebayes = 0
    m = len(dataset)
    for row in dataset:
        x, y = row[:-1], row[-1]
        y_bayes = bayes.predict(x)
        y_naivebayes = naivebayes.predict(x)
        if y_bayes == y:
            m_bayes += 1
        if y_naivebayes == y:
            m_naivebayes += 1
    print("Bayes       : {:.2f}% accuracy".format(m_bayes*100/m))
    print("Naive Bayes : {:.2f}% accuracy".format(m_naivebayes*100/m))


if __name__ == '__main__':
    main()
