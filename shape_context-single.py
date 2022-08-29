# coding=utf-8
import cv2
import sys
import numpy as np
import copy
from munkres import Munkres,print_matrix
import math
#from matplotlib import pyplot as plt

def canny_points(edges, points_num=100):
    h, w = edges.shape
    print(h, w)

    count = 0
    edges_sample = np.zeros((h, w))
    points = []
    while count < points_num:
        axis_h = np.random.randint(h, size=1)
        axis_w = np.random.randint(w, size=1)
        # print(axis_h,axis_w)
        # print(edgesA[axis_h[0],axis_w[0]])
        if edges[axis_h[0], axis_w[0]] > 1:
            ax = [axis_h[0], axis_w[0]]
            edges[axis_h[0]-3:axis_h[0]+3, axis_w[0]-3:axis_w[0]+3] = 0
            edges_sample[axis_h[0], axis_w[0]] = 255
            points.append(ax)
            count = count + 1
            #print(count)

    #cv2.imshow('canny edges', edges_sample)
    #cv2.waitKey(1)
    return points
def shape_bins(points):
    N = len(points)
    bins_all = []
    ang_Block = 12
    dis_Block = 5
    for point_o in points[:]:
        distances = []
        angle = []
        for point in points[:]:
            distance = np.sqrt((point_o[0] - point[0]) ** 2 + (point_o[1] - point[1]) ** 2)
            if distance > 0.00001:
                distances.append(distance)
                angl = np.arcsin((point_o[0] - point[0]) / distance)
                if point_o[1] - point[1] < 0 and point_o[0] - point[0] > 0:
                    angl = angl + pi / 2
                if point_o[1] - point[1] < 0 and point_o[0] - point[0] < 0:
                    angl = angl - pi / 2
                if angl < 0:
                    angl = 2 * pi + angl
                angle.append(np.floor(6.0 * angl / pi))  # sin
                # print(distance,angl)
        mean_dist = np.mean(distances)
        distances = distances / mean_dist

        # print(angle)
        # print(mean_dist)
        # print(distances)
        block_lens = 1
        distances_log = np.log(distances / block_lens)

        for x in range(len(distances_log)):
            if distances_log[x] <= 0:
                distances_log[x] = 0
            elif distances_log[x] <= 1:
                distances_log[x] = 1
            elif distances_log[x] <= 2:
                distances_log[x] = 2
            elif distances_log[x] <= 3:
                distances_log[x] = 3
            elif distances_log[x] <= 4:
                distances_log[x] = 4

        bins = np.zeros((dis_Block, ang_Block))
        for x in range(len(distances_log)):
            bins[int(distances_log[x]), int(angle[x])] = bins[int(distances_log[x]), int(angle[x])] + 1

        # np.arcsin
        # print(bins)
        # plt.imsave('xx%d.jpg'%point_o[0],bins)
        # plt.show()
        bins = np.reshape(bins,[ang_Block*dis_Block])
        bins_all.append(bins)
    return bins_all


def find_farthest_point(points):
    points = np.float32(points)
    kdtree = cv2.flann.Index()
    params = dict(algorithm=1, trees=1)
    kdtree.build(points, params)
    indices, dists = kdtree.knnSearch(points, points.shape[0], params=-1)   #KDtree算法寻找最近点
    meandist = np.mean(dists,axis=1)
    maxindex = np.argmax(meandist)

    return np.array([int(points[maxindex][0]), int(points[maxindex][1])]), maxindex

def creatMSTMap(points):
    in_tree_array = [0]
    N = len(points)
    out_tree_array = [i for i in range(1, len(points))]
    map = np.zeros((N, N))
    points = np.float32(points)
    kdtree = cv2.flann.Index()
    params = dict(algorithm=1, trees=1)
    kdtree.build(points, params)
    indices, dists = kdtree.knnSearch(points, points.shape[0], params=-1)
    while len(out_tree_array)>0:
        min_distance = 100000000
        min_index1 = -1
        min_index2 = -1
        search_indices = indices[out_tree_array, :]
        search_dists = dists[out_tree_array, :]
        for i in range(len(search_indices)):
            for k in range(1, N):
                if search_indices[i][k] in in_tree_array and search_dists[i][k]<min_distance:
                    min_distance = search_dists[i][k]
                    min_index1 = i
                    min_index2 = search_indices[i][k]
                    continue
        map[min_index2][out_tree_array[min_index1]] = 1
        in_tree_array.append(out_tree_array[min_index1])
        del out_tree_array[min_index1]
    # print(np.sum(map, axis=0))
    return map
        # min_index = np.argmin(search_dist)
        #in_tree_array.append(min_out)
        #out_tree_array = out_tree_array[:min_out] + out_tree_array[min_out+1:]
        #map[min_in][min_out] = 1



def oriented_shape_bins(points, image=None):
    point_refrence, index_refrence = find_farthest_point(points)    # 查找refrence points
    points_out = points[:index_refrence] + points[index_refrence+1:]
    imageA_path = 'BM.png'
    # imageB_path = 'back_2.png'
    imageB_path = 'B.png'

    # read images A and B
    if image.all() != None:
        for point in points_out:
            point = [point[1], point[0]]
            point_refrence_temp = [point_refrence[1], point_refrence[0]]
            cv2.line(image, tuple(point), point_refrence_temp , (255, 0, 0),1)
        cv2.imshow("try", image)
        cv2.waitKey()
    # 从原点集中去掉refrence points
    # MST_map = creatMSTMap(points_out) # 生成一个MST map（按照原论文说法，不生成应该问题也不大）
    N = len(points_out)
    bins_all = []
    ang_Block = 12
    dis_Block = 5
    for point_o in points_out[:]:       # 制作oriented shape context bins
        distances = []
        angle = []
        point_o = point_o-point_refrence    # 得到方向
        for point in points_out[:]:     # 计算周围点的角度
            point = point-point_refrence
            distance = np.sqrt((point_o[0] - point[0]) ** 2 + (point_o[1] - point[1]) ** 2)
            if distance > 0.00001:
                distances.append(distance)
                angl = np.arcsin((point_o[0] - point[0]) / distance)
                angl = math.atan2(point)
                if point_o[1] - point[1] < 0 and point_o[0] - point[0] > 0:
                    angl = angl + pi / 2
                if point_o[1] - point[1] < 0 and point_o[0] - point[0] < 0:
                    angl = angl - pi / 2
                if angl < 0:
                    angl = 2 * pi + angl
                angle.append(np.floor(6.0 * angl / pi))  # sin(angle/(2pi/12))
                # print(distance,angl)
        mean_dist = np.mean(distances)
        distances = distances / mean_dist

        # print(angle)
        # print(mean_dist)
        # print(distances)
        block_lens = 1
        distances_log = np.log(distances / block_lens)

        for x in range(len(distances_log)):
            if distances_log[x] <= 0:
                distances_log[x] = 0
            elif distances_log[x] <= 1:
                distances_log[x] = 1
            elif distances_log[x] <= 2:
                distances_log[x] = 2
            elif distances_log[x] <= 3:
                distances_log[x] = 3
            elif distances_log[x] <= 4:
                distances_log[x] = 4

        bins = np.zeros((dis_Block, ang_Block))
        for x in range(len(distances_log)):
            bins[int(distances_log[x]), int(angle[x])] = bins[int(distances_log[x]), int(angle[x])] + 1

        # np.arcsin
        # print(bins)
        # plt.imsave('xx%d.jpg'%point_o[0],bins)
        # plt.show()
        bins = np.reshape(bins,[ang_Block*dis_Block])
        bins_all.append(bins)
        edges_all = points_out-point_refrence

    return bins_all, edges_all


def cost_matrix(bins_A, bins_B, frameedge_A, frameedge_B):
    row = 0
    col = 0
    N_A = len(bins_A)
    N_B = len(bins_B)
    miu = 1
    cost1 = np.zeros((N_A, N_B))
    cost2 = np.zeros((N_A, N_B))
    for i in range(N_A):
        col = 0
        for k in range(N_B):
            # print(bin_A+bin_B)
            cost1[row, col] = 0.5 * np.sum(((bins_A[i] - bins_B[k]) ** 2) / (bins_A[i] + bins_B[k] + 0.00000001))
            cost2[row, col] = miu*((np.abs(frameedge_A[i][0])+np.abs(frameedge_A[i][1]))-\
                              (np.abs(frameedge_B[k][0])+np.abs(frameedge_B[k][1])))**2/((np.abs(frameedge_A[i][0])+np.abs(frameedge_A[i][1]))+\
                              (np.abs(frameedge_B[k][0])+np.abs(frameedge_B[k][1])))
            col = col + 1
        row = row + 1

    cost = cost1+cost2
        # cv2.imshow('xxx2',cost1/255.0)
        # cv2.waitKey()
    return cost1


if __name__ =="__main__":
    pi = 3.1415926535
    sample_num = 100
    imageA_path = 'BM.png'
    #imageB_path = 'back_2.png'
    imageB_path = 'B.png'

    # read images A and B
    imageA = cv2.imread(imageA_path)
    imageB = cv2.imread(imageB_path)
    imageA = cv2.cvtColor(imageA,cv2.COLOR_BGR2GRAY)
    imageB = cv2.cvtColor(imageB,cv2.COLOR_BGR2GRAY)

    # canny
    minVal_canny = 100
    maxVal_canny = 200
    edgesA = cv2.Canny(imageA,minVal_canny,maxVal_canny)
    edgesB = cv2.Canny(imageB,minVal_canny,maxVal_canny)
    #cv2.imshow('edges',edgesA)
    #cv2.waitKey()

    # Randomly select some points
    pointsA = canny_points(edgesA,sample_num)
    pointsB = canny_points(edgesB,sample_num)

    # Calculate shape context
    # rotation invariance is not considered yet
    bins_A, frameedgesA = oriented_shape_bins(pointsA, imageA)
    bins_B, frameedgesB = oriented_shape_bins(pointsB, imageB)

    # bins_A = shape_bins(pointsA)
    # bins_B = shape_bins(pointsB)

    bins_A = np.array(bins_A)
    bins_B = np.array(bins_B)
    frameedgesA = np.array(frameedgesA)
    frameedgesB = np.array(frameedgesB)
    # Calculate the cost matrix between two bins
    cost = cost_matrix(bins_A, bins_B, frameedgesA, frameedgesB)
    cost_list = cost.tolist()
    #cost = [[50,61,23,98],[57,24,54,19],[78,73,7,46],[6,86,1,88]]
    m = Munkres()   # Munkres Algorithm
    indexes = m.compute(cost_list)   #
    #print_matrix('Lowest cost through this matrix:', cost)
    total = 0
    for row, column in indexes:
        value = cost_list[row][column]
        print(value)
        total += value
        #print('(%d, %d) -> %d' % (row, column, value))
    print('total cost: %d' % total)

    if total < 750:
        print('Same shape!')
    else:
        print('Not the Same shape')

