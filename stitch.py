import cv2
import numpy as np
import sys
import json

cv2.namedWindow("IMAGE STITCHING",cv2.WINDOW_NORMAL)

f = open('config.json')
co = json.load(f)


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


if __name__ == "__main__":
    img1 = cv2.imread("Tests/"+sys.argv[1])
    img2 = cv2.imread("Tests/"+sys.argv[2])

    print("[ Stitching Started .... ]")

    st_img ,homography = stitchImages(img1,img2)
    # stitchImages(img1,img2)
    cv2.imwrite(f"Result/result_stitch_{co['currentOutput']}.jpg",st_img)
    co["currentOutput"]=str(int(co["currentOutput"])+1)
    if sys.argv[3]=="1":
        with open('config.json', 'w') as f:
            json.dump(co, f)
    cv2.imshow("IMAGE STITCHING",st_img)
    cv2.waitKey(0)

f.close()
