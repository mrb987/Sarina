from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold, cross_val_score
import os
import numpy as np
from glob import glob
import cv2 as cv
import pickle
from random import shuffle
import warnings
warnings.simplefilter('ignore')

dir = os.path.join('d:/databank/images/WBC images')
def image_processing(dir):
    normal_dir = (os.path.join(dir,'Normal'))
    normal_data = glob(normal_dir + '/*jpg')
    illness_dir = (os.path.join(dir,'Illness'))
    illness_data = glob(illness_dir + '/*jpg')

    dataset = list()
    labels = [0,1] # 0 is normal, and 1 is illness
    for i in range(len(normal_data)):
        image1 = cv.imread(normal_data[i], cv.IMREAD_UNCHANGED) #IMREAD_GRAYSCALE IMREAD_COLOR 
        dataset.append([np.array(image1).flatten(), labels[0]])
    for i in range(len(illness_data)):
        image2 = cv.imread(illness_data[i], cv.IMREAD_UNCHANGED)
        dataset.append([np.array(image2).flatten(), labels[1]])

    output = open('d:/databank/data.pickle', 'wb')
    pickle.dump(dataset, output)
    output.close()

    input = open('d:/databank/data.pickle', 'rb')
    data = pickle.load(input)
    input.close()

    features, targets = list(), list()
    shuffle(data)
    for feature, lable in data:
        features.append(feature)
        targets.append(lable)
    return features, targets
features, targets = image_processing(dir)

def boosting(features, targets):
    scores = []
    final_models = []
    models = []
    models.append(KNeighborsClassifier())
    models.append(DecisionTreeClassifier())
    models.append(LogisticRegression())
    models.append(SVC())
    models.append(GaussianNB())
    models.append(RandomForestClassifier(n_estimators=25))
    kfold = KFold(n_splits=20, random_state=10, shuffle=True)
    for model in models:
        ens_model = AdaBoostClassifier(base_estimator=model, n_estimators=50)
        result = cross_val_score(ens_model, features, targets, cv=kfold)
        score = result.mean()
        scores.append(score)
    scores = [0 if np.isnan(i)==True else i for i in scores]
    best_model = models[np.argmax(scores)]
    best_score = scores[np.argmax(scores)]
    return best_score, AdaBoostClassifier(base_estimator=best_model, n_estimators=50)

score, model = boosting(features, targets)
print('The possible accuracy is {}%'.format(score * 100))