from math import sqrt

def caculate_distance(p1, p2):
    dimension = len(p1) #4 chiều
    distance_square = 0
    for i in range(dimension):
        distance_square += (p1[i] - p2[i])*(p1[i] - p2[i])
    distance = sqrt(distance_square)
    return distance

p1 = [1, 2]
p2 = [2, 4]

print(caculate_distance(p1, p2))