import numpy as np
import cv2
import imutils
from imutils import paths

cv2.namedWindow("Output",cv2.WINDOW_NORMAL)

print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images("Tests/C")))
images = []
# loop over the image paths, load each one, and add them to our
# images to stitch list
for imagePath in imagePaths:
	image = cv2.imread(imagePath)
	images.append(image)


print("[INFO] stitching images...")
stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
(status, stitched) = stitcher.stitch(images)

if status == 0:
	# write the output stitched image to disk
	cv2.imwrite("Result/opencv_stitcher_result_C.jpg", stitched)
	# display the output stitched image to our screen
	cv2.imshow("Output", stitched)
	cv2.waitKey(0)
# otherwise the stitching failed, likely due to not enough keypoints)
# being detected
else:
	print("[INFO] image stitching failed ({})".format(status))
