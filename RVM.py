from sklearn import datasets
from skbayes.rvm_ard_models import RVC

def RVM(XEstimate,ClassLabels, XValidate, Parameters):
    clf = RVC(n_iter = Parameters.get('n_iter'),
    tol = Parameters.get('tol'),
    n_iter_solver = Parameters.get('n_iter_solver'),
    tol_solver = Parameters.get('tol_solver'),
    fit_intercept = Parameters.get('fit_intercept'),
    verbose = Parameters.get('verbose'),
    kernel = Parameters.get('kernel'),
    degree = Parameters.get('degree'),
    gamma = Parameters.get('gamma'),
    coef0 = Parameters.get('coef0'),
    kernel_params = Parameters.get('kernel_params') )
    clf.fit(XEstimate, ClassLabels)
    Yvalidate = clf.predict_proba(XValidate)
    EstParameters = { 'relevant_vectors':clf.relevant_vectors_ ,'coef':clf.coef_,'active':clf.active_,
                      'intercept':clf.intercept_,'mean':clf._x_mean, 'std':clf._x_std,'classes':clf.classes_,
                      'lambda':clf.lambda_,'sigma':clf.sigma_,'relevant':clf.relevant_}
    return Yvalidate,EstParameters


def TestMyClassifier(XTest,Parameters,EstParameters):
    clf = RVC(n_iter=Parameters.get('n_iter'),
              tol=Parameters.get('tol'),
              n_iter_solver=Parameters.get('n_iter_solver'),
              tol_solver=Parameters.get('tol_solver'),
              fit_intercept=Parameters.get('fit_intercept'),
              verbose=Parameters.get('verbose'),
              kernel=Parameters.get('kernel'),
              degree=Parameters.get('degree'),
              gamma=Parameters.get('gamma'),
              coef0=Parameters.get('coef0'),
              kernel_params=Parameters.get('kernel_params'))
    clf.relevant_vectors_ = EstParameters.get('relevant_vectors')
    clf.relevant_ = EstParameters.get('relevant')
    clf.active_ = EstParameters.get('active')
    clf.coef_ = EstParameters.get('coef')
    clf.intercept_ = EstParameters.get('intercept')
    clf._x_mean = EstParameters.get('mean')
    clf._x_std = EstParameters.get('std')
    clf.classes_ = EstParameters.get('classes')
    clf.lambda_ = EstParameters.get('lambda')
    clf.sigma_ = EstParameters.get('sigma')
    Ytest = clf.predict_proba(XTest)
    return Ytest


Parameters = { 'n_iter':100, 'tol':1e-4, 'n_iter_solver':15, 'tol_solver':1e-4,
                 'fit_intercept':True, 'verbose': False, 'kernel':'rbf', 'degree': 2,
                 'gamma': None, 'coef0':1, 'kernel_params':None }

iris = datasets.load_iris()

#Yvalidate,EstParameters =  RVM2(train_data,label_data,train_data,Parameters)
Yvalidate,EstParameters = RVM(iris.data, iris.target, iris.data[1:6] ,Parameters)



print(TestMyClassifier(iris.data[1:5],Parameters,EstParameters))