# Images-Stitching-Using-OpenCV

### Step By Step Procedure :
1. Convert Image's color mode from BGR to grayscale.
2. Keypoints and descriptors are computed using the SIFT object. [ SIFT (Scale-Invariant Feature Transform) ]
3. Keypoints between two images are matched by identifying their nearest neighbours using FLANN (stands for Fast Library for Approximate Nearest Neighbors ).
4. Ratio test is applied to detect the good keypoints.
5. To found locations of some parts of an object in another cluttered image using <b>cv2.findHomography()</b>. If we pass the set of points from both
the images, it will find the perpective transformation of that object. Then we can use <b>cv2.perspectiveTransform()</b> to
find the object. It needs atleast four correct points to find the transformation. <b>[ Algorithm uses RANSAC or LEAST_MEDIAN ]</b>

## Results

#### M.A.R.S. C.U.R.I.O.S.I.T.Y. R.O.V.E.R. panaroma view


#### Input Images :
* Image 1
* Image 2

![GitHub Logo](https://github.com/ashish1sasmal/Images-Stitching-Using-OpenCV/blob/master/Tests/b2.png)
![GitHub Logo](https://github.com/ashish1sasmal/Images-Stitching-Using-OpenCV/blob/master/Tests/b1.png)

#### Output Image [Stitched Output ] : 

![GitHub Logo](https://github.com/ashish1sasmal/Images-Stitching-Using-OpenCV/blob/master/Result/result_stitch_1.jpg)
