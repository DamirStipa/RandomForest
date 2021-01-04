from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from cv2 import cv2

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def unpickle_labels(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

def label_names(index):
    return train_labels[index]

#start_time = time.time()

x_train = []
y_train = []
x_test = []
y_test = []

cifar_data1 = unpickle("C:\\Users\\stipa\\Desktop\\random_forest\\dataset\\train\\data_batch_1")
'''cifar_data2 = unpickle("C:\\Users\\stipa\\Desktop\\random_forest\\dataset\\train\\data_batch_2")
cifar_data3 = unpickle("C:\\Users\\stipa\\Desktop\\random_forest\\dataset\\train\\data_batch_3")
cifar_data4 = unpickle("C:\\Users\\stipa\\Desktop\\random_forest\\dataset\\train\\data_batch_4")
cifar_data5 = unpickle("C:\\Users\\stipa\\Desktop\\random_forest\\dataset\\train\\data_batch_5")'''
unpickled_lables = unpickle_labels("C:\\Users\\stipa\\Desktop\\random_forest\\dataset\\train\\batches.meta")
test_data = unpickle("C:\\Users\\stipa\\Desktop\\random_forest\\dataset\\test\\test_batch")

train_labels = unpickled_lables['label_names']

array1 = cifar_data1[b'data']
'''array2 = cifar_data2[b'data']
array3 = cifar_data3[b'data']
array4 = cifar_data4[b'data']
array5 = cifar_data5[b'data']'''
for item in array1:
    x_train.append(item)
'''for item in array2:
    x_train.append(item)
for item in array3:
    x_train.append(item)
for item in array4:
    x_train.append(item)
for item in array5:
    x_train.append(item)'''

X_train = np.array(x_train)
X_train = X_train.reshape(10000, 3072)
X_train = X_train.astype("float32")

x_test = test_data[b'data']

#formatting for CV output
x_test_cv = np.reshape(x_test, (10000, 3, 32, 32))
test_cv = []
for item in x_test_cv:
    test_cv.append(np.transpose(item, (1,2,0)))

for item in test_cv:
    item = cv2.cvtColor(item, cv2.COLOR_RGB2BGR)

X_test = np.array(x_test)
X_test = X_test.reshape(10000, 3072)
X_test = X_test.astype("float32")
test_cv = np.array(test_cv)

cifar_labels1 = cifar_data1[b'labels']
'''cifar_labels2 = cifar_data2[b'labels']
cifar_labels3 = cifar_data3[b'labels']
cifar_labels4 = cifar_data4[b'labels']
cifar_labels5 = cifar_data5[b'labels']'''

for item in cifar_labels1:
    y_train.append(item)
'''for item in cifar_labels2:
    y_train.append(item)
for item in cifar_labels3:
    y_train.append(item)
for item in cifar_labels4:
    y_train.append(item)
for item in cifar_labels5:
    y_train.append(item)'''

Y_train = np.array(y_train)

y_test = test_data[b'labels']
Y_test = np.array(y_test)
Y_train = Y_train.flatten()
Y_test = Y_test.flatten()

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

X_train /= 255.
X_test /= 255.

rfc_model = RandomForestClassifier(n_jobs=5, n_estimators=100)

#cv_results = cross_val_score(rfc_model, X_train, Y_train, cv=5, scoring='accuracy')

#plt.boxplot(cv_results)
#plt.show()

print("STATUS: Training RFC...")
rfc_model.fit(X_train[:10000], Y_train[:10000])
print("RFC SCore: " + str(rfc_model.score(X_train, Y_train)))

fixed_size = tuple((200,200))

for i in range(len(X_test)):
    curr_image = cv2.resize(test_cv[i], fixed_size)
    predicted = rfc_model.predict(X_test[i].reshape(1,-1))
    expected = Y_test[i]
    predicted = str(predicted).strip('[]')
    print("EXPECTED: " + str(expected))
    print("PREDICTED: " + str(predicted))
    # show predicted label on image
    cv2.putText(curr_image, str(label_names(int(predicted))), (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

    # display the output image
    plt.imshow(cv2.cvtColor(curr_image, cv2.COLOR_RGB2BGR))
    plt.show()

    #0-airplane    1-automobile    2-bird    3-cat    4-deer    5-dog   6-frog    7-horse   8-ship    9-truck
