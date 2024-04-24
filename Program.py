from sklearn.ensemble import VotingClassifier, StackingClassifier, AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.metrics import accuracy_score
import os
import numpy as np
from glob import glob 
import cv2 as cv
import pickle
from random import shuffle
import warnings
warnings.simplefilter('ignore')

### Convert Pictures of Cancer to Arrays ###
dir = os.path.join('d:/Files/databank/images/WBC images')
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

    output = open('d:/Files/databank/data.pickle', 'wb')
    pickle.dump(dataset, output)
    output.close()

    input = open('d:/Files/databank/data.pickle', 'rb')
    data = pickle.load(input)
    input.close()

    features, targets = list(), list()
    shuffle(data)
    for feature, lable in data:
        features.append(feature)
        targets.append(lable)
    return features, targets
features, targets = image_processing(dir)

### Model Creation % Scaling ###
def model_stackng():
    def boosting(features, targets):
        scores = []
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

    def scaling(model, features, targets):
        kfold = KFold(n_splits=20, random_state=10, shuffle=True)
        Scaled_features = StandardScaler().fit_transform(features)
        result = cross_val_score(model, Scaled_features, targets, cv=kfold)
        scaled_score = result.mean()
        return scaled_score, StandardScaler()

    base_score, boosting_model = boosting(features, targets)
    print('The possible accuracy is {}%'.format(base_score * 100))
    scaled_score, scale_method = scaling(boosting_model, features, targets)
    print('The possible accuracy after apply scaling {}%'.format(scaled_score * 100))
    return scaled_score, scale_method, base_score, boosting_model
scaled_score, scale_method, base_score, boosting_model = model_stackng()


### Cancer Detection ###
def new_instance(features, targets):  
    dir_path = dir + '/New instance'
    list_instance = []
    for item in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, item)) == True:
            list_instance.append(item)
            
    illness_dir = (os.path.join(dir,'New instance'))
    illness_data = glob(illness_dir + '/*jpg')
    dataset, new_x = [], []
    for i in range(len(illness_data)):
        image = cv.imread(illness_data[i], cv.IMREAD_UNCHANGED)
        dataset.append([np.array(image).flatten()])
    output = open('d:/Files/databank/new_data.pickle', 'wb')
    pickle.dump(dataset, output)
    output.close()
    input = open('d:/Files/databank/new_data.pickle', 'rb')
    new_data = pickle.load(input)
    input.close()
    shuffle(new_data)
    if scaled_score > base_score:
        features = scale_method.fit_transform(features)
        for i in range(len(new_data)):
            new_x.append((list_instance[i], scale_method.fit_transform(new_data[i])))
    else:
        for i in range(len(new_data)):
            new_x.append((list_instance[i], new_data[i]))
    return new_x

def execute(new_x):
    xtrain, xtest, ytrain, ytest = train_test_split(features, targets, test_size=0.3,
                                                    random_state=40, shuffle=True)
    boosting_model.fit(xtrain, ytrain)
    ypred = boosting_model.predict(xtest)
    predict_accuracy = round(accuracy_score(ytest, ypred)*100, 2)
    for i in new_x:
        y_pred_true = boosting_model.predict(i[1])
        file_name = i[0]
        if y_pred_true == 0:
            print('{} was diagnosed as benign with an accuracy of {}%'.format(file_name, predict_accuracy))
        if y_pred_true == 1:
            print('{} was diagnosed as malignant with an accuracy of {}%'.format(file_name, predict_accuracy))
new_x = new_instance(features, targets)
execute(new_x)