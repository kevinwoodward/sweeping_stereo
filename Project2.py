import numpy as np
import math
import matplotlib.pyplot as plt
import os
import cv2
import glob


# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('./calpics/*.JPG')

print("Now calibrating camera for radial distorion, this may take a few minutes...")
for num,fname in enumerate(images,1):
    print("{0}%...".format(num*10))
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners)
        cv2.drawChessboardCorners(img, (9,6), corners,ret)
print("100%!")
print("")


ret, kmtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
print("Intrinsic matrix K:")
print(kmtx)
print("Radial distortion coeff (k1, k2, k3)=({0},{1},{2})".format(dist[0][0],dist[0][1],dist[0][4]))
img = cv2.imread('parallax1.JPG')
h,  w = img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(kmtx,dist,(w,h),1,(w,h))



mean_error = 0
for i in xrange(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], kmtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error**2

print "Mean squared error for quality of calibration: ", mean_error/len(objpoints)


pic1 = cv2.imread("parallax1.JPG")
pic2 = cv2.imread("parallax2.JPG")

# undistort
dst = cv2.undistort(pic1, kmtx, dist, None, newcameramtx)


# crop the image
x,y,w,h = roi
pic1result = dst[y:y+h, x:x+w]
#cv2.imwrite('pic1result.png', pic1result)


# undistort
dst = cv2.undistort(pic2, kmtx, dist, None, newcameramtx)


# crop the image
x,y,w,h = roi
pic2result = dst[y:y+h, x:x+w]
#cv2.imwrite('pic2result.png', pic2result)

gray1 = cv2.cvtColor(pic1result,cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(pic2result,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(gray1,None)
kp2, des2 = sift.detectAndCompute(gray2,None)

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)

good = []
pts1 = []
pts2 = []

print("Feature detecting, this may take a minute...")

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

print("Done!")
pts1all = np.float32(list(pts1))
pts2all = np.float32(list(pts2))
F, mask = cv2.findFundamentalMat(pts1all,pts2all,cv2.FM_RANSAC)
# We select only inlier points
pts1 = pts1all[mask.ravel()==1]
pts2 = pts2all[mask.ravel()==1]



pic1points = gray1.copy()
pic1points = cv2.drawKeypoints(gray1,[],pic1points, color=(0,255,0))
#plt.imshow(pic1points)
##cv2.imwrite("pic1points.JPG", pic1points)

pic2points = gray2.copy()
pic2points = cv2.drawKeypoints(gray2,[],pic2points, color=(0,255,0))
#plt.imshow(pic2points)
##cv2.imwrite("pic2points.JPG", pic2points)



def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2


# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
lines1 = lines1.reshape(-1,3)
img5,img6 = drawlines(gray1,gray2,lines1,pts1,pts2)

# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(gray2,gray1,lines2,pts2,pts1)

plt.subplot(121),plt.imshow(img5)
plt.subplot(122),plt.imshow(img3)
plt.title("Epipolar lines Pic1,Pic2")
plt.show()

#print("Saving epipolar lines drawn on images...")
#cv2.imwrite("epipoles1.JPG", img5)
#print("epipoles1.JPG saved!")
#cv2.imwrite("epipoles2.JPG", img3)
#print("epipoles2.JPG saved!")


E, _ = cv2.findEssentialMat(pts1, pts2, newcameramtx)

R1, R2, t = cv2.decomposeEssentialMat(E)


posebase = np.matmul(newcameramtx, np.append(np.identity(3), np.zeros_like(t), axis=1))
pose1pos = np.matmul(newcameramtx, np.append(R1, t, axis=1))
pose2pos = np.matmul(newcameramtx, np.append(R2, t, axis=1))
pose1neg = np.matmul(newcameramtx, np.append(R1, -t, axis=1))
pose2neg = np.matmul(newcameramtx, np.append(R2, -t, axis=1))

print("Rotation matrix R:")
print(R2)
print("Translation vector t:")
print(-t)

pts1float = pts1.astype(np.float32)
pts1formatted = pts1float[:, np.newaxis, :]
pts1undist = cv2.undistortPoints(pts1formatted, newcameramtx, dist, P=newcameramtx)
pts1undist = pts1.astype(np.float32)

pts2float = pts2.astype(np.float32)
pts2formatted = pts2float[:, np.newaxis, :]
pts2undist = cv2.undistortPoints(pts2formatted, newcameramtx, dist, P=newcameramtx)
pts2undist = pts2.astype(np.float32)

pb = posebase.copy()
p = pose2neg.copy()
p1 = pts1undist.copy()
p2 = pts2undist.copy()

ptsHomo = cv2.triangulatePoints(np.squeeze(pb), np.squeeze(p), np.squeeze(p1).transpose(), np.squeeze(p2).transpose())

ptsEuclid = []
for i in range(0, ptsHomo[0].size):
    euclidPt = np.array([ptsHomo[0][i]/ptsHomo[3][i], ptsHomo[1][i]/ptsHomo[3][i], ptsHomo[2][i]/ptsHomo[3][i]])
    ptsEuclid.append(euclidPt)

#Cleaning points
ptsEuclid = [pt for pt in ptsEuclid if pt[2] > 0 and pt[2] < 50]

ptsSensor = []
for i in range(0, len(ptsEuclid)):
    vec = ptsEuclid[i].reshape(3, 1)
    homovec = np.matmul(newcameramtx, vec)
    homovec = np.squeeze(homovec)
    ptsSensor.append((int(round(homovec[0]/homovec[2])), int(round(homovec[1]/homovec[2]))))


z_list = [z for x,y,z in ptsEuclid if z > 0]

projImage = pic1points.copy()
print("Projecting triangulated points to image 1...")
for point, projection, threedeepoint in zip(pts1, ptsSensor, ptsEuclid):
    cv2.circle(projImage, tuple(point), 6, (0, 0, 255), -1)
    if threedeepoint[2] < 0:
        continue
    cv2.circle(projImage, projection, 10, (0, 255, 0), -1)
plt.imshow(cv2.cvtColor(projImage, cv2.COLOR_BGR2RGB))
plt.title("Projected points")
plt.show()
print("done!")
#print("projected.JPG saved!")
#cv2.imwrite("projected.JPG", projImage)




z_list.sort()
z_max, z_min = max(z_list), min(z_list)
print("Distance d_min=" + str(z_min))
print("Distance d_max=" + str(z_max))
newcameramtx_inv = np.linalg.inv(newcameramtx)
n_vector = np.array([[0.], [0.], [-1.]])
n_vector_trans = n_vector.transpose()
Homography_vec = []
print("Calculating sweeping plane homographies...")
for d in np.linspace(z_min, z_max, 20):
    rn_by_d = np.matmul(-t, n_vector_trans)/d
    H = R2 - rn_by_d
    KH = np.matmul(newcameramtx, H)
    KHM = np.matmul(KH, newcameramtx_inv)
    Homography_vec.append(KHM)
print("done!")
warped_img_list = []

for Homography,d in zip(Homography_vec, np.linspace(z_min, z_max, 20)):
    im = cv2.warpPerspective(pic2points.copy(), Homography, (pic2points.shape[1], pic2points.shape[0]))
    plt.imshow(im)
    plt.title("Warped image with distance d="+ str(d))
    plt.show()
    warped_img_list.append((im, d))


originalpic = pic1points.copy().astype(np.int16)
processed_list = []
t_mag = np.linalg.norm(t)
combined_differences = np.ones_like(originalpic) * z_min

for warped,d in warped_img_list:
    differences = np.zeros_like(originalpic).astype(np.int16)
    warped = warped.astype(np.int16)
    abs_diff = np.abs(originalpic - warped)
    abs_diff_filtered = cv2.boxFilter(abs_diff, 3, (25, 25))
    processed_list.append((abs_diff_filtered, d))

d_list = []
for d in np.linspace(z_max, z_min, 20):
    d_list.append(d)

print("Generating sweeping plane depthmap...")
for x, col in enumerate(combined_differences):
    if (x % 100 == 0):
        print("{0}%...".format(int(float(x)/(2267) * 100)))
    for y, row in enumerate(combined_differences[x]):
        sad_list = np.array([image[x][y] for image, _ in processed_list])
        min_d_index = np.argmin(sad_list, axis=0)[0]
        combined_differences[x][y] = d_list[min_d_index]
print("100%!")

scaled_heatmap = combined_differences.astype(np.float32) * 255 / z_max
plt.imshow(scaled_heatmap)
plt.title("Final sweeping plane depthmap, may look bad in matplotlib! Open file depthmap.JPG to view.")
plt.show()
#print("Saving heatmap to heatmap.JPG")
cv2.imwrite("dephtmap.JPG", scaled_heatmap.astype(np.float32))
#plt.imshow(combined_differences.astype(np.uint8))
#plt.show()


# originalpic_half = 0.5 * pic1points.copy().astype(np.float32)
# for index, (warped, d) in enumerate(warped_img_list):
#     warped_half = 0.5 * warped
#     #cv2.imwrite("overlayed"+str(index)+".JPG", originalpic_half + warped_half)
