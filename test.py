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
import train

pred = []

def test():
    TrainingFeatures = train.train()
    Features = []
    sum = 0
    img = io.imread("testing/test1.bmp")
    #print img.shape

    io.imshow(img)
    plt.title('Original Image')
    io.show()

    hist = exposure.histogram(img)
    plt.bar(hist[1], hist[0])
    plt.title('Histogram')
    plt.show()

    th = 200
    img_binary = (img < th).astype(np.double)

    img_binary = medfilt(img_binary)
    #img_binary = ndimage.morphology.binary_closing(img_binary)

    io.imshow(img_binary)
    plt.title('Binary Image')
    io.show()
    #print(type(img_binary))

    img_label = label(img_binary, background=0)
    io.imshow(img_label)
    plt.title('Labeled Image')
    io.show()
    # print np.amax(img_label)

    regions = regionprops(img_label)
    io.imshow(img_binary)
    ax = plt.gca()
    plt.title('Bounding boxes')
    true = []
    for props in regions:
        minr, minc, maxr, maxc = props.bbox
        #print str(counter) + ": " + str(maxr - minr) + "," + str(maxc - minc)

        if (maxr - minr) < 9 or (maxc - minc) < 9 or (maxr - minr + maxc - minc) < 25: continue
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

    #print(counter)
    io.show()
    D = cdist(Features, TrainingFeatures)
    D_index = np.argsort(D, axis=1)
    #print(D_index)
    for i in range(0, len(D_index)):
        pred.append(train.YTrue[D_index[i][0]])

