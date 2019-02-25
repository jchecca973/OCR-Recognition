import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
from skimage.measure import label, regionprops, moments, moments_central, moments_normalized, moments_hu
from skimage import io, exposure
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pickle
import train
import test

if __name__ == '__main__':

    test.test()
    pred = test.pred

    pkl_file = open('testing_ground_truth/test1_gt.pkl', 'rb')
    mydict = pickle.load(pkl_file)
    pkl_file.close()
    classes = mydict['classes']
    locations = mydict['locations']

    #print(classes)
    #print(pred)
    #print(locations)
    i = 0
    correct = 0
    for item in classes:
        item = item.lower()
        print "True character: " + item + " " + " Predicted character: " + pred[i]
        if item == pred[i]: correct += 1
        i += 1
    print
    print "Recognition Rate: " + str((float)(correct)/(float)(len(classes)))
