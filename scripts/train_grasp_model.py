from sklearn.linear_model import LogisticRegression
import numpy as np

np.random.seed(0)
num_trials = 100

convs = [np.random.rand(9,) for i in np.arange(num_trials)]
# convs = np.reshape(convs, (1, 9))
outcomes = np.random.randint(2, size=num_trials)
convs = np.asarray(convs)
clf = LogisticRegression(solver='liblinear').fit(convs, outcomes)
# print clf.score(convs, outcomes)
# print clf.predict([np.random.rand(9,)])