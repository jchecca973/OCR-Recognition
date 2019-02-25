import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
from skimage.measure import label, regionprops, moments, moments_central, moments_normalized, moments_hu
from skimage import io, exposure
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pickle
from scipy.signal import medfilt
from scipy import ndimage

YTrue = []

def train():
    letters = ['a', 'd', 'm', 'n', 'o', 'p', 'q', 'r', 'u', 'w']
    Features = []
    sum = 0
    sdevs = []
    means = []
    YPred = []
    for i in range(0,len(letters)):
        imgloc = "training/" + letters[i] + ".bmp"
        img = io.imread(imgloc)
        #img = medfilt(img)
        #print img.shape

        io.imshow(img)
        plt.title('Original Image')
        io.show()


        hist = exposure.histogram(img)
        plt.bar(hist[1], hist[0])
        plt.title('Histogram')
        plt.show()

        th = 210
        img_binary = (img < th).astype(np.double)


        if (i == 4):
            img_binary = ndimage.morphology.binary_dilation(img_binary)


        img_binary = medfilt(img_binary)

        io.imshow(img_binary)
        plt.title('Binary Image')
        io.show()
        #print(type(img_binary))


        img_label = label(img_binary, background=0)
        io.imshow(img_label)
        plt.title('Labeled Image')
        io.show()
        #print np.amax(img_label)

        regions = regionprops(img_label)
        io.imshow(img_binary)
        ax = plt.gca()
        plt.title('Bounding boxes')

        for props in regions:
            minr, minc, maxr, maxc = props.bbox
            #print str(maxr - minr) + "," +  str(maxc - minc)
            if (maxr - minr) < 8 or (maxc - minc) < 8 or (maxr - minr + maxc - minc) < 25 or (maxr - minr + maxc - minc) > 150: continue
            sum += 1
            ax.add_patch(Rectangle((minc, minr), maxc - minc, maxr - minr,
                                   fill=False, edgecolor='red', linewidth=1))
            roi = img_binary[minr:maxr, minc:maxc]
            m = moments(roi)
            cr = m[0, 1] / m[0, 0]
            cc = m[1, 0] / m[0, 0]
            mu = moments_central(roi, cr, cc)
            nu = moments_normalized(mu)
            hu = moments_hu(nu)
            Features.append(hu)
            YTrue.append(letters[i])

        #print(sum)
        io.show()

        #print Features[i]
        #print "*****"
        sdev = np.std(Features[i])
        mean = np.mean(Features[i])
        sdevs.append(sdev)
        means.append(mean)


    D = cdist(Features, Features)
    io.imshow(D)
    plt.title('Distance Matrix')
    io.show()
    #print(len(D))
    #print(D)

    D_index = np.argsort(D, axis=1)
    #print(D_index)

    for i in range(0, len(Features)):
        YPred.append(YTrue[D_index[i][1]])

    confM = confusion_matrix(YTrue, YPred)
    io.imshow(confM)
    plt.title('Confusion Matrix')
    io.show()


    #print(type(Features))
    sdev = np.std(sdevs)
    mean = np.mean(means)
    #print sdev
    #print mean
    #print(Features)
    #print (sum / 10)
    for i in range(0, len(letters)):
        Features[i] -= mean
        Features[i] /= sdev
    #print(len(Features))
    #print(YTrue)
    #print(YPred)
    return Features

