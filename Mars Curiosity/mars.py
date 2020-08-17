
import cv2
import numpy as np
import sys

from os import listdir
from os.path import isfile, join


cv2.namedWindow("IMAGE STITCHING",cv2.WINDOW_NORMAL)

MIN_MATCH_COUNT = 10

def stitchImages(img1, img2):
    gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

    # Initiate SIFT detector
    orb = cv2.xfeatures2d.SIFT_create()

    # Find keypoints and descriptor using SIFT
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(np.asarray(des1,np.float32),np.asarray(des2,np.float32),k=2)

    # bf = cv2.BFMatcher()
    # matches = bf.knnMatch(des1,des2, k=2)

    good = []
    for m,n in matches:
        if m.distance < 0.5*n.distance:
            good.append(m)

    # matches = np.asarray(good)

    # points1 = np.zeros((len(matches), 2), dtype=np.float32)
    # points2 = np.zeros((len(matches), 2), dtype=np.float32)
    #
    # for i, match in enumerate(matches):
    #     points1[i, :] = keypoints1[match.queryIdx].pt
    #     points2[i, :] = keypoints2[match.trainIdx].pt

    if len(good)>=MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        h,w,c = img2.shape

        dst = cv2.warpPerspective(img1, M, (img1.shape[1]+ img2.shape[1], img2.shape[0]))
        dst[0:img2.shape[0], 0:img2.shape[1]] = img2
        return(dst,M)

    else:
        print("[Not enough matches are found]")
        matchesMask = None

#
if __name__ == "__main__":
    onlyfiles = [f for f in listdir("Samples") if isfile(join("Samples", f))]
    onlyfiles.sort()
    print(onlyfiles)


    onlyfiles=onlyfiles[::-1]
    img1 = cv2.imread("Samples/"+onlyfiles[0])
    print("[ Stitching Started .... ]")
    for (i,j) in enumerate(onlyfiles[1:]):
        print(f"[ Stitching Status : Image No. {i+1}  ...]")
        img2 = cv2.imread("Samples/"+j)
        img1 ,homography = stitchImages(img1,img2)
    # stitchImages(img1,img2)
    print("[ Stitching Done !]")
    cv2.imwrite(f"Result/Mars_Curiosity.png",img1)
    # co["currentOutput"]=str(int(co["currentOutput"])+1)
    # if sys.argv[3]=="1":
    #     with open('config.json', 'w') as f:
    #         json.dump(co, f)
    cv2.imshow("IMAGE STITCHING",img1)
    cv2.waitKey(0)
