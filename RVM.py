from sklearn import datasets
from skrvm import RVC
from scipy.special import expit
import numpy as np
import scipy.io as io

'''========================== Method to train RVM OneVsOne with NC2 classifiers======================================'''
def RVM(XEstimate,ClassLabels, XValidate, Parameters):
    clf =  RVC(alpha=Parameters.get('alpha'),beta=Parameters.get('beta'),n_iter=Parameters.get('n_iter'),verbose='true')
    clf.fit(XEstimate, ClassLabels)
    if np.shape(clf.classes_)[0] == 2:
        Yvalidate = clf.predict(XValidate)
    else:
        Yvalidate = predict_proba(clf,XValidate)
    EstParameters = get_params(clf)
    return Yvalidate,EstParameters

''' ========================== Method to Predict labels using RVM ==================================================='''
def TestMyClassifier(XTest,Parameters,EstParameters):
    if len(EstParameters) == 1:
        clf = EstParameters.get('clf')
    else:
        clf = EstParameters[0]
    if np.shape(clf.classes_)[0] == 2:
        Ytest = clf.predict_proba(XTest)
    else:
        Ytest = predict_proba(clf,XTest)
    return Ytest

'''============================ Predicts probability of labels for RVM ==============================================='''
def predict_proba(clf,XValidate):
    noOfClasses = np.shape(clf.classes_)[0]
    noOfClassifiers = (noOfClasses * (noOfClasses-1))/2
    print('no of Classifiers',noOfClassifiers)
    dataSize = np.shape(XValidate)[0]
    Yvalidate = np.zeros((dataSize, np.shape(clf.classes_)[0]))
    c = 0
    prob = clf.multi_.estimators_[c].predict_proba(XValidate)
    #Summing Fkm(X) where k!=m
    for i in range(0,noOfClasses):
        for j in range(i, noOfClasses):
            if (i < j):
                Yvalidate[:, i] =  Yvalidate[:, i]+ prob[:, 0]
                Yvalidate[:, j] =  Yvalidate[:, j]+ prob[:, 1]
                c = c + 1;
                if(c<noOfClassifiers):
                    prob = clf.multi_.estimators_[c].predict_proba(XValidate)
    #Calculating 1/G(summation(ykm))
    Yvalidate = Yvalidate / np.shape(clf.classes_)[0]
    #Calculating probability of XValidate  not belonging to any class
    prob_std = np.ndarray.std(Yvalidate, axis=1)[:, np.newaxis]
    sigmoid = 1 - expit(prob_std)
    Yvalidate = np.concatenate([Yvalidate, sigmoid], axis=1)
    Yvalidate = Yvalidate / np.repeat((sigmoid + 1), axis=1, repeats=np.shape(clf.classes_)[0] + 1)
    return Yvalidate

'''====================== Method to get the Parameters and the HyperParameters ====================================='''
def get_params(clf):
    if np.shape(clf.classes_)[0] == 2:
        Parameters = [{'phi': clf.phi,'relevance': clf.relevance_,'alpha': clf.alpha_,'beta': clf.beta_,'m': clf.m_,
                       'gamma': clf.gamma,'bias': clf.bias,'clf': clf}]
    else :
        Parameters = [clf]
        for c in clf.multi_.estimators_:
            Parameter = {'phi': c.phi,'relevance': c.relevance_,'alpha': c.alpha_,'beta': c.beta_,'m': c.m_,
                         'gamma': c.gamma,'bias': c.bias,'clf': c}
            Parameters.append(Parameter)
    return Parameters

'''==================================== Testing Part ============================================================== '''
# Hpyer Parameter Range : alpha 100,1,1e-6, beta 1.e-2 1.e-6 1.e-8,

#Default Parameters
''' kernel='rbf',degree=3,coef1=None,coef0=0.0,n_iter=500,tol=1e-3,alpha=1e-6,
    threshold_alpha=1e9,beta=1.e-6,beta_fixed=False,bias_used=True,verbose=False '''


Parameters = { 'alpha':1e-6,'beta':1.e-8,'n_iter':200 }
iris = datasets.load_iris()
train = io.loadmat("/Users/devanshusingh/PycharmProjects/ML/ML2/Proj2FeatVecsSet1.mat")['Proj2FeatVecsSet1']
# label shape is [25000, 5] as [num_sample, num_class]
label = io.loadmat("/Users/devanshusingh/PycharmProjects/ML/ML2/Proj2TargetOutputsSet1.mat")['Proj2TargetOutputsSet1']
label = np.argmax(label, axis=1)

Yvalidate,EstParameters = RVM(iris.data, iris.target, iris.data ,Parameters)
#print(EstParameters)
from sklearn.metrics import accuracy_score
print(np.argmax(Yvalidate ,axis=1))
print(iris.target)
print(accuracy_score(np.argmax(Yvalidate ,axis=1), iris.target))
#print(TestMyClassifier(iris.data,Parameters,EstParameters))
