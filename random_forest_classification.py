import numpy as np
import mahotas
from cv2 import cv2
import os
import h5py
import glob
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from skimage.feature import hog
import time

tic = time.perf_counter()
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def unpickle_labels(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

cifar_data1 = unpickle("C:\\Users\\stipa\\Desktop\\random_forest\\dataset\\train\\data_batch_1")
cifar_data2 = unpickle("C:\\Users\\stipa\\Desktop\\random_forest\\dataset\\train\\data_batch_2")
cifar_data3 = unpickle("C:\\Users\\stipa\\Desktop\\random_forest\\dataset\\train\\data_batch_3")
cifar_data4 = unpickle("C:\\Users\\stipa\\Desktop\\random_forest\\dataset\\train\\data_batch_4")
cifar_data5 = unpickle("C:\\Users\\stipa\\Desktop\\random_forest\\dataset\\train\\data_batch_5")
unpickled_lables = unpickle_labels("C:\\Users\\stipa\\Desktop\\random_forest\\dataset\\train\\batches.meta")
test_data = unpickle("C:\\Users\\stipa\\Desktop\\random_forest\\dataset\\test\\test_batch")

train_labels = unpickled_lables['label_names']

array = []
array1 = cifar_data1[b'data']
array2 = cifar_data2[b'data']
array3 = cifar_data3[b'data']
array4 = cifar_data4[b'data']
array5 = cifar_data5[b'data']
for item in array1:
    array.append(item)
for item in array2:
    array.append(item)
for item in array3:
    array.append(item)
for item in array4:
    array.append(item)
for item in array5:
    array.append(item)
array = np.reshape(array, (50000, 3, 32, 32))

test_array = test_data[b'data']
test_array_final = []
for item in test_array:
    test_array_final.append(item)
test_array_final = np.reshape(test_array_final, (10000, 3, 32, 32))
'''print("TRAIN ARRAY SHAPE")
print(array[:10])
print("TEST ARRAY SHAPE")
print(test_array_final[:10])
'''
images = []
test_images = []
for item in array:
    images.append(np.transpose(item, (1,2,0)))

for item in images:
    item = cv2.cvtColor(item, cv2.COLOR_RGB2BGR)

for item in test_array_final:
    test_images.append(np.transpose(item, (1,2,0)))

for item in test_images:
    item = cv2.cvtColor(item, cv2.COLOR_RGB2BGR)

cifar_labels = []
cifar_labels1 = cifar_data1[b'labels']
cifar_labels2 = cifar_data2[b'labels']
cifar_labels3 = cifar_data3[b'labels']
cifar_labels4 = cifar_data4[b'labels']
cifar_labels5 = cifar_data5[b'labels']

for item in cifar_labels1:
    cifar_labels.append(item)
for item in cifar_labels2:
    cifar_labels.append(item)
for item in cifar_labels3:
    cifar_labels.append(item)
for item in cifar_labels4:
    cifar_labels.append(item)
for item in cifar_labels5:
    cifar_labels.append(item)

fixed_size = tuple((200,200))

h5_data = 'C:\\Users\\stipa\\Desktop\\random_forest\\output\\data.h5'
h5_labels = 'C:\\Users\\stipa\\Desktop\\random_forest\\output\\labels.h5'
bins = 8
scoring    = "accuracy"

#Feature descriptor: Hu Moments
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

#Feature descriptor: Haralick Texture
def fd_haralick_texture(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick

#Feature descriptor: Color Histogram
def fd_color_histogram(image, mask = None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([image], [0,1,2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

#Feature descriptor: Histogram of Gradients
def fd_hog(image):
    return hog(image, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), multichannel=True)
    

global_features = []
labels = []

for i in range(len(images)):
    curr_image = cv2.resize(images[i], fixed_size)

    fv_hu_moments = fd_hu_moments(curr_image)
    fv_haralick_texture = fd_haralick_texture(curr_image)
    #fv_color_histogram = fd_color_histogram(curr_image)
    fv_hog = fd_hog(curr_image)

    global_feature = np.hstack([fv_haralick_texture, fv_hu_moments, fv_hog])

    labels.append(train_labels[cifar_labels[i]])
    global_features.append(global_feature)
    print(i)
    
# get the overall feature vector size
#print("[STATUS] feature vector size {}".format(np.array(global_features).shape))

# get the overall training label size
#print("[STATUS] training Labels {}".format(np.array(labels).shape))

# encode the target labels
targetNames = np.unique(labels)
le          = LabelEncoder()
target      = le.fit_transform(labels)

#Now that we have the features we can scale them
scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_features = scaler.fit_transform(global_features)
#print("FEATURES BEFORE SCALING")
#print(global_features)
#print("SHAPE OF FEATURES")
#print(rescaled_features.shape)
#print(rescaled_features)

#print("[STATUS] target labels: {}".format(target))
#print("[STATUS] target labels shape: {}".format(target.shape))

num_trees = 100
test_size = 0.10
seed      = 9

#(trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(rescaled_features),
                                                                                          #np.array(target),
                                                                                          #test_size=test_size,
                                                                                          #random_state=seed)

#print(trainLabelsGlobal.shape)
#print(testLabelsGlobal.shape)
#print(trainDataGlobal.shape)
#print(testDataGlobal.shape)

#kfold = KFold(n_splits=10, random_state=seed, shuffle=True)
#cv_results = cross_val_score(RandomForestClassifier(n_estimators=num_trees, random_state=seed), trainDataGlobal, trainLabelsGlobal, cv=kfold, scoring=scoring)

#plt.boxplot(cv_results)
#plt.show()

toc = time.perf_counter()
print(f"Extracted features in {toc - tic:0.4f} seconds")
print("\n Training RFC...")
tik = time.perf_counter()
clf  = RandomForestClassifier(n_estimators=num_trees, random_state=seed)

#print("RESCALED FEATURES TYPE: " + str(rescaled_features.dtype))
#print(target)
clf.fit(rescaled_features, target)
tok = time.perf_counter()
print(f"Training finished in {tok-tik:0.4f} seconds")

#test_images_array = np.array(test_images)
#test_images_array = test_images_array.astype("float32")

for image in test_images:
    curr_image = cv2.resize(image, fixed_size)

    fv_hu_moments = fd_hu_moments(curr_image)
    fv_haralick_texture = fd_haralick_texture(curr_image)
    #fv_color_histogram = fd_color_histogram(curr_image)
    fv_hog = fd_hog(curr_image)

    global_feature = np.hstack([fv_haralick_texture, fv_hu_moments, fv_hog])
    #print("GLOBAL FEATURE SHAPE")
    #print(global_feature.shape)
    global_feature = global_feature.reshape(-1,1)
    #print("AFTER RESHAPE")
    #print(global_feature.shape)
    #print(global_feature[:10])

    # scale features in the range (0-1)
    #scaler_test = MinMaxScaler(feature_range=(0, 1))
    #rescaled_feature = scaler_test.fit_transform(global_feature.reshape(1,-1))
    #print(rescaled_feature)
    #print("SHAPE OF TEST FEATURE")
    #print(global_feature.shape)
    #print("AFTER RESHAPE")
    global_feature = global_feature.reshape(1,-1)
    #print(global_feature.shape)

    # predict label of test image
    #rescaled_feature = rescaled_feature.astype("float32")
    prediction = clf.predict(global_feature)[0]
    print(prediction)

    # show predicted label on image
    cv2.putText(curr_image, train_labels[prediction], (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 1)

    # display the output image
    plt.imshow(cv2.cvtColor(curr_image, cv2.COLOR_BGR2RGB))
    plt.show()
