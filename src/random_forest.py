__author__ = 'bhavika'

"""
Kaggle exercise - https://www.kaggle.com/c/bioresponse
"""

from sklearn.ensemble import RandomForestClassifier
from numpy import genfromtxt, savetxt


def main():
    #create the training and test sets, skipping the header row with [1:]
    # f8 in dtype is 64 bit single precision floating point

    dataset = genfromtxt(open('data/biotrain.csv', 'r'), delimiter=',', dtype='f8')[1:]
    target = [x[0] for x in dataset]
    train = [x[1:] for x in dataset]
    test = genfromtxt(open('data/biotest.csv', 'r'), delimiter=',', dtype='f8')[1:]


    #create and train the random forest
    #multi-core CPUs can use: rf = RandomForestClassifier(n_estimators=100, n_jobs=2)

    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(train, target)
    predicted_probs = [[index+1, x[1]] for index, x in enumerate(rf.predict_proba(test))]
    savetxt('data/bio-output.csv', predicted_probs, delimiter=',', fmt='%d,%f',
            header='MoleculeID, PredictedProbability', comments='')

if __name__ == "__main__":
    main()
