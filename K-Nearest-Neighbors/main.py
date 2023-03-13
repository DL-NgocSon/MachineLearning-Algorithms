from sklearn import datasets
import numpy as np
import math
import operator

iris = datasets.load_iris()
iris_X = iris.data  #data (pental_length, pental_width, sepal_length, sepal_width)
iris_y = iris.target

randIndex = np.arange(iris_X.shape[0])
np.random.shuffle(randIndex)

iris_X = iris_X[randIndex]
iris_y = iris_y[randIndex]
#print(randIndex)
#print(iris_y)

X_train = iris_X[:100, :]
X_test = iris_X[100:, :]
y_train = iris_y[:100]
y_test = iris_y[100:]
# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)

def caculate_distance(p1, p2):
    distance_square = 0
    dimension = len(p1) #4 chiều
    for i in range(dimension):
        distance_square += (p1[i] - p2[i])*(p1[i] - p2[i])
    distance = math.sqrt(distance_square)
    return distance


# input là một danh sách. output là những điểm dữ liệu phổ biến nhất để convert điểm cần dự đoán theo nó. => trả ra nhãn của điểm cần dự đoán
def predict(training_X, label_y, point, k):
    neighbors_labels = get_k_neighbors(training_X, label_y, point, k)   #Cha lai k diem gan nhat
    return highest_votes(neighbors_labels)

# Tính khoảng cách từ điểm đó đến tất cả các điểm trong trainning
def get_k_neighbors(training_X, label_y, point, k):
    distances = []   # Danh sách khoảng cách.
    neighbors = []   #danh sách k điểm gần nhất dựa theo kc
    for i in range(len(training_X)):
        distance = caculate_distance(training_X[i], point)
        distances.append((distance, label_y[i]))
    # for i in distances:
    #     if()
    distances.sort(key=operator.itemgetter(0))    #sort by distance
    for i in range(k):
        neighbors.append(distances[i][1])
    return neighbors

# Kiem tra xem tan suat xuat hien nhieu nhat rồi lấy ra label của các điểm xuất hiện nhiều nhất.
def highest_votes(labels):
    labels_count = [0, 0, 0]
    for i in labels:
        labels_count[i] += 1
    
    max_count = max(labels_count)
    return labels_count.index(max_count)

    

def accuracy_(predicts, labels):   #label == groundTruth
    total = len(predicts)
    correct_count = 0
    for i in range(total):
        if predicts[i] == labels[i]:
            correct_count += 1
    
    accuracy = correct_count/total
    return accuracy

k = 5
y_predict = []
for p in X_test:
    label = predict(X_train, y_train, p, k)
    y_predict.append(label)


print(y_predict)
print(y_test)

accuracy = accuracy_(y_predict, y_test)
print(accuracy)