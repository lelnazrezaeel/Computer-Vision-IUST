from importlib.resources import path
import cv2
import os
import numpy

#part1 (reading img1.png by cv2.imread syntax)
#cv2.imread (path, state)
path = r'E:\University\Term7\FCV\HW2\Images\img1.png'
img1 = cv2.imread(path)
cv2.imshow('img1', img1)
cv2.waitKey(0)

#part2 (find corners by cv2.findChesboardCorners synatx)
#findChessboardCorners (InputArray image, Size patternSize,OutputArray corners, int flags)
reterval, corners= cv2.findChessboardCorners(img1, (24,17), None)
print(reterval)
print(corners, '\n\n')
    
#part3 (increase the accuracy of found corners by cv2.cornerSubPix)
#cornerSubPix (InputArray image, InputOutputArray corners, Size winSize, Size zeroZone, TermCriteria criteria)	
#drawChessboardCorners (InputOutputArray image, Size patternSize, InputArray corners,bool patternWasFound)	
gray = cv2.cvtColor(img1 ,cv2.COLOR_BGR2GRAY)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
imgSubPix = cv2.drawChessboardCorners(img1, (24,17), corners2, reterval)
cv2.imshow('imgSubPix', imgSubPix)
cv2.waitKey(0)
pathAnswers = r'E:\University\Term7\FCV\HW2\Images'
cv2.imwrite(os.path.join(pathAnswers, '4.3_Answer.png'), imgSubPix)

#part4 (determine camera calibration parameters by cv2.calibrateCamera)
#calibrateCamera (InputArrayOfArrays objectPoints, InputArrayOfArrays imagePoints, Size imageSize,
#                 InputOutputArray cameraMatrix, InputOutputArray distCoeffs,
#                 OutputArrayOfArrays rvecs, OutputArrayOfArrays tvecs, int flags = 0, TermCriteria)	
objectPoints = []
imagePoints = []
imagePoints.append(corners2)
#numpy.zeros(shape, dtype=float, order='C', *, like=None)
objectP = numpy.zeros((24*17,3), numpy.float32)
#print(objectP, '\n\n')
objectP[:,:2] = numpy.mgrid[0:24,0:17].T.reshape(-1,2)
objectPoints.append(objectP)
ret, camMat, distortion, rotV, transV = cv2.calibrateCamera(objectPoints, imagePoints, img1.shape[1::-1], None, None)
print('reterval:\n', ret , '\n\ncamera matrix:\n' , camMat , '\n\ndistortion coefficience:\n' , distortion 
      , '\n\nrotation vectors:\n' , rotV , '\n\ntranslation vectors:\n' , transV, '\n\n')

#part5 (determine k1, k2, k3, p1, p2)
print('distortion coefficience:')
print('k1:', distortion[0][0], '\nk2:', distortion[0][1], '\np1:', distortion[0][2],
      '\np2:', distortion[0][3], '\nk3:', distortion[0][4], '\n\n')

#part6 (Undisortion image5 by distortion parameters)
img5 = cv2.imread(os.path.join(pathAnswers, 'img5.png'))
cv2.imshow('img5', img5)
cv2.waitKey(0)
#getOptimalNewCameraMatrix (cameraMatrix, distCoeffs, imageSize, alpha[, newImgSize[, centerPrincipalPoint]]) 
camMat2, roi = cv2.getOptimalNewCameraMatrix(camMat, distortion, img5.shape[1::-1], 0, img5.shape[1::-1])
#undistort(src, cameraMatrix, distCoeffs[, dst[, newCameraMatrix]]) 
undisortion = cv2.undistort(img5, camMat, distortion, None, camMat2)
cv2.imshow('Undisorted image', undisortion)
cv2.waitKey(0)
cv2.imwrite(os.path.join(pathAnswers, '4.6_Answer.png'), undisortion)

#part7 (determin calibration parameters with images[1-4])
img1 = cv2.imread(os.path.join(pathAnswers, 'img1.png'))
img2 = cv2.imread(os.path.join(pathAnswers, 'img2.png'))
img3 = cv2.imread(os.path.join(pathAnswers, 'img3.png'))
img4 = cv2.imread(os.path.join(pathAnswers, 'img4.png'))
img5 = cv2.imread(os.path.join(pathAnswers, 'img5.png'))
images = [img1, img3, img4]
objPoints = []
imgPoints = []
objectP2 = numpy.zeros((17*24,3), numpy.float32)
objectP2[:,:2] = numpy.mgrid[0:17,0:24].T.reshape(-1,2)
for image in images:
    reterval, corners = cv2.findChessboardCorners(image, (24,17))
    if reterval == True:
        objPoints.append(objectP)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgPoints.append(corners2)
reterval, corners = cv2.findChessboardCorners(img2, (17,24))
if reterval == True:
    objPoints.append(objectP2)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
    imgPoints.append(corners2)
ret2, camMat3, distortion2, rotV2, transV2 = cv2.calibrateCamera(objPoints, imgPoints, img1.shape[1::-1], None, None)
camMat4, roi = cv2.getOptimalNewCameraMatrix(camMat3, distortion2, img1.shape[1::-1], 0, img1.shape[1::-1])
undisortion2 = cv2.undistort(img5, camMat3, distortion2, None, camMat4)
cv2.imshow('Undisorted image', undisortion2)
cv2.waitKey(0)
cv2.imwrite(os.path.join(pathAnswers, '4.7_Answer.png'), undisortion2)