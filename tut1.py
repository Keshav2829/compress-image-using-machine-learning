import cv2
import numpy as num

#  function to define random initial centroids
def KMeansInitialCentroid(X,K):
    randindex = num.random.permutation(num.shape(X)[0])
    Centroids = X[randindex[0:K],:]
    return Centroids

# function for assigning each pixel to closest centroid
def AssinIndex(X,Centroids):
    m = num.shape(X)[0]
    n = num.shape(Centroids)[0]
    index = num.zeros((m, 1))

    for i in range(m - 1):
        b = 100000
        for j in range(n - 1):
            v = num.matmul(X[i, :] - Centroids[j, :], num.transpose(X[i, :] - Centroids[j, :]))
            if v < b:
                b = v
                r = j
        index[i] = r
    index = num.array(index)
    return index

# function for calculating new centroids
def CalculateCentroid(X,index,n):
    X = num.array(X)
    NewCentroid = num.zeros((n, 3))
    for k in range(n - 1):
        b = num.argwhere(index == k)[:, 0]
        s = num.sum(X[b, :], axis=0)
        d = num.count_nonzero(index == k)
        NewCentroid[k] = num.divide(s, d)
    return NewCentroid

# k means algorithm function
def runKmeans(X,Centroids,max_itr,K):
    for i in range(max_itr):
        idx = AssinIndex(X,Centroids)
        Centroids = CalculateCentroid(X,idx,K)
    Centroids = num.array(Centroids)
    return Centroids



img = cv2.imread('photo.jpg', -1)
cv2.imshow('orignalimg', img)
# making img variable an object of numpy
img1= num.array(img)


# re-scalling the matrix so that each value is in between 0-1
imgdata = num.true_divide(img1,255)
imagesize = num.shape(imgdata)

# represent imagedata in shape of pixels x (R,G,B)
X = num.reshape(imgdata,(imagesize[0]*imagesize[1],3))
# for implementation of K - means algorithm, we are creating image that uses only 16 different colour combination
K = 25
max_itr = 12
# creating a function that returns 16 initial centroid for our algorithm

InitialCentroid = KMeansInitialCentroid(X,K)
Centroids = num.array(runKmeans(X,InitialCentroid,max_itr,K))
index = num.array(AssinIndex(X,Centroids)).astype(int)

# assign new pixel data for each pixel of image according to calculated Centroids
X_recovered = Centroids[index,:]
X_recovered = num.reshape(X_recovered,(imagesize[0],imagesize[1],3))


# show the modified image and saving it
cv2.imshow('modifiedimg',X_recovered)
cv2.waitKey(0)
cv2.destroyAllWindows()

X_recovered = num.multiply(X_recovered,255)
cv2.imwrite('modified_image.png',X_recovered)
