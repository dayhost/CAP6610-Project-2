from sklearn.metrics import confusion_matrix, accuracy_score

def MyConfusionMatrix(Y,ClassNames):
    print(confusion_matrix(ClassNames, Y))
    return confusion_matrix(ClassNames, Y), accuracy_score(ClassNames, Y)
