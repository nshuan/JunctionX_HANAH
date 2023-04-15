import cv2
import numpy as np
from datetime import datetime
import os
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

from scipy.spatial import ConvexHull
from time import perf_counter_ns as clock

THRESHOLD = 0.6
NUM_ITERS = 100

BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)

def getPoints(image1, image2, point_map, inliers=None, max_points=1000):
    rows1, cols1 = image1.shape
    rows2, cols2 = image2.shape

    matchImage = np.zeros((max(rows1, rows2), cols1 + cols2, 3), dtype='uint8')
    matchImage[:rows1, :cols1, :] = np.dstack([image1] * 3)
    matchImage[:rows2, cols1:cols1 + cols2, :] = np.dstack([image2] * 3)

    small_point_map = [point_map[i] for i in np.random.choice(len(point_map), max_points)]
    ret_points = [[], []]
    for x1, y1, x2, y2 in small_point_map:
        point1 = [int(x1), int(y1)]
        point2 = [int(x2), int(y2)]
        color = BLUE if inliers is None else (
            GREEN if (x1, y1, x2, y2) in inliers else RED)
        if color != RED:
            cv2.circle(matchImage, point1, 5, BLUE, 1)
            cv2.circle(matchImage, point2, 5, BLUE, 1)
            cv2.line(matchImage, point1, point2, color, 1)
            ret_points[0].append(point1)
            ret_points[1].append(point2)

    ret_points[0] = np.array(ret_points[0])
    ret_points[1] = np.array(ret_points[1])
    points = ConvexHull(ret_points[0]).vertices
    x_hull = np.append(ret_points[0][points,0], ret_points[0][points,0][0])
    y_hull = np.append(ret_points[0][points,1], ret_points[0][points,1][0])
    points = ConvexHull(ret_points[1]).vertices
    x_hull1 = np.append(ret_points[1][points,0], ret_points[1][points,0][0])
    y_hull1 = np.append(ret_points[1][points,1], ret_points[1][points,1][0])
    return list(zip(x_hull, y_hull)), list(zip(x_hull1, y_hull1))


def computeHomography(pairs):
    A = []
    for x1, y1, x2, y2 in pairs:
        A.append([x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2])
        A.append([0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1, -y2])
    A = np.array(A)

    # Singular Value Decomposition (SVD)
    U, S, V = np.linalg.svd(A)

    H = np.reshape(V[-1], (3, 3))

    # Normalization
    H = (1 / H.item(8)) * H
    return H


def dist(pair, H):
    # points in homogeneous coordinates
    p1 = np.array([pair[0], pair[1], 1])
    p2 = np.array([pair[2], pair[3], 1])

    p2_estimate = np.dot(H, np.transpose(p1))
    p2_estimate = (1 / p2_estimate[2]) * p2_estimate

    return np.linalg.norm(np.transpose(p2) - p2_estimate)


def RANSAC(point_map, threshold=THRESHOLD, verbose=True):
    if verbose:
        print(f'Running RANSAC with {len(point_map)} points...')
    bestInliers = set()
    homography = None
    for i in range(NUM_ITERS):
        # randomly choose 4 points from the matrix to compute the homography
        pairs = [point_map[i] for i in np.random.choice(len(point_map), 4)]

        H = computeHomography(pairs)
        inliers = {(c[0], c[1], c[2], c[3])
                   for c in point_map if dist(c, H) < 500}

        if verbose:
            print(f'\x1b[2K\r└──> iteration {i + 1}/{NUM_ITERS} ' +
                  f'\t{len(inliers)} inlier' + ('s ' if len(inliers) != 1 else ' ') +
                  f'\tbest: {len(bestInliers)}', end='')

        if len(inliers) > len(bestInliers):
            bestInliers = inliers
            homography = H
            if len(bestInliers) > (len(point_map) * threshold):
                break

    if verbose:
        print(f'\nNum matches: {len(point_map)}')
        print(f'Num inliers: {len(bestInliers)}')
        print(f'Min inliers: {len(point_map) * threshold}')

    return homography, bestInliers


def createPointMap(image1, image2, verbose=True):
    if verbose:
        print('Finding keypoints and descriptors for both images...')
    sift = cv2.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(image1, None)
    kp2, desc2 = sift.detectAndCompute(image2, None)
    if verbose:
        print('Determining matches...')
    matches = cv2.BFMatcher(cv2.NORM_L2, True).match(desc1, desc2)

    point_map = np.array([
        [kp1[match.queryIdx].pt[0],
         kp1[match.queryIdx].pt[1],
         kp2[match.trainIdx].pt[0],
         kp2[match.trainIdx].pt[1]] for match in matches
    ])
    points0, points1 = getPoints(image1, image2, point_map)
    return point_map, points0, points1


def _subOverlap(image1, image2, verbose=False):
    point_map = None
    if verbose:
        print('Creating point map...')
    point_map, points0, points1 = createPointMap(image1, image2, verbose=verbose)

    homography, inliers = RANSAC(point_map, verbose=verbose)
    return points0, points1

def overlap(img1, img2, downscale=1):
    start = clock()
    img1 = cv2.resize(img1, (img1.shape[0]//downscale, img1.shape[1]//downscale))
    img2 = cv2.resize(img2, (img2.shape[0]//downscale, img2.shape[1]//downscale))
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    points = _subOverlap(gray1, gray2)
    points0 = np.array(points[0], dtype=np.int32)
    points1 = np.array(points[1], dtype=np.int32)
    img_1 = cv2.polylines(img1, [points0], isClosed=True, color=(0,255,0), thickness=4)
    img_2 = cv2.polylines(img2, [points1], isClosed=True, color=(0,255,0), thickness=4)
    finish = clock()
    return img_1, img_2, (finish - start)/(1e9)

if __name__ == '__main__':
    vid1 = cv2.VideoCapture("D:/video_data/dynamic_cam/192_168_5_101.mp4")
    vid2 = cv2.VideoCapture("D:/video_data/dynamic_cam/192_168_5_103.mp4")

    while True:
        ret, i1 = vid1.read()
        if ret:
            ret2, i2 = vid2.read()
            if ret2:
                img1, img2, t = overlap(i1, i2, 2)
                img1 = cv2.resize(img1, (640, 480))
                img2 = cv2.resize(img2, (640, 480))
                print(f"Time: {t}s")
                cv2.imshow("Camera 1", img2)
                cv2.imshow("Camera 0", img1)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            