import numpy as np

def roc_curve(probabilities, labels):
    '''
    INPUT: numpy array, numpy array
    OUTPUT: list, list, list

    Take a numpy array of the predicted probabilities and a numpy array of the
    true labels.
    Return the True Positive Rates, False Positive Rates and Thresholds for the
    ROC curve.
    '''
    
    true_positives=[]
    false_positives=[]
    
    thold = []
    
    pos = len([x for x in labels if x == 1])
    
    neg = len([x for x in labels if x == 0])
    
    for t, l in sorted(zip(probabilities, labels)):
        
        
        thold.append(t)
        
        
        x = [(1, a) for b, a in sorted(zip(probabilities, labels)) if b >= t]
        truepos = np.sum([1 for b, a in x if b == 1 and a == 1]) / pos
        falsepos = np.sum([1 for b, a in x if b == 1 and a != 1]) / neg
        true_positives.append(truepos)
        false_positives.append(falsepos)
    return true_positives, false_positives, thold     
    
        
        
        