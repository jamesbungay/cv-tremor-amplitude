# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import cv2


CAMERA_FOCAL_LENGTH = 2.87  # mm
CAMERA_FOCAL_LENGTH_STD = 32  # mm

OBJECT_DEPTH = 40  # cm

VIDEO_WIDTH = 1080
VIDEO_FILEPATH = 'data/IMG_1224.mov'
START_FRAME = 540
END_FRAME = 9999


# -----------------------------------------------------------------------------
# V I D E O  P R O C E S S I N G

def setUpBlobDetector():
    params = cv2.SimpleBlobDetector_Params()

    params.minThreshold = 0
    params.maxThreshold = 255
    # params.thresholdStep = 10

    params.filterByArea = True
    params.minArea = 2500
    params.maxArea = 25000

    params.filterByCircularity = False
    params.filterByColor = False
    params.filterByConvexity = False
    params.filterByInertia = False

    detector = cv2.SimpleBlobDetector_create(params)
    return detector


def findPendulumMidpoint(frame, detector):

    # Pre-process image ready for blob detection:
    cv2.imwrite('data/frame.jpg', frame)
    grayscaled = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grayscaled, (5, 5), 0)
    (__, thresholded) = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)

    keypoints = detector.detect(thresholded)
    
    imgWithKeypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite('data/keypoints.jpg', imgWithKeypoints)
    
    if len(keypoints) != 1:
        return -1
    
    # Return the midpoint of the keypoint:
    for kp in keypoints:
        return round(kp.pt[0])



def processVideo():
    minLeft = VIDEO_WIDTH
    maxRight = 0
    path = []
    pathFrameNumbers = []
    failedFrames = 0
    
    capture = cv2.VideoCapture(VIDEO_FILEPATH)
    frameN = 0
    detector = setUpBlobDetector()
    while (True):
        success, frame = capture.read()
        if success:
            frameN += 1
            if START_FRAME <= frameN <= END_FRAME:
                mp = findPendulumMidpoint(frame, detector)
                if mp == -1:
                    failedFrames += 1
                else:
                    path.append(mp)
                    pathFrameNumbers.append(frameN)
                    if mp < minLeft:
                        minLeft = mp
                    if mp > maxRight:
                        maxRight = mp
        else:
            break
    capture.release()
    
    pixelDistance = maxRight - minLeft
    return pixelDistance


# -----------------------------------------------------------------------------
# C A M E R A  P A R A M E T E R  C A L C U L A T I O N

# TODO derive these constants, rather than hardcoding

CAMERA_SENSOR_WIDTH = 2.328
CAMERA_SENSOR_HEIGHT = 3.104


def calcPixelSize(imageResolution, sensorWidth, lensFocalLength, depth):
    """
    Calculate the width, in mm, of a pixel in an image at a given depth.

    Parameters
    ----------
    imageResolution: The width of an image in pixels.
    sensorWidth: The width of the camera sensor in mm.
    lensFocalLength: Focal length of the lens, in mm.
    depth: The depth at which the width of a pixel is to be calculated, in cm.
    """

    # TODO: adjust for error. See phone photos 3/12. take depth error as input
    # and output a tuple, of width and error.

    viewWidth = depth * 10 * sensorWidth / lensFocalLength
    return viewWidth / imageResolution


def calcCropSensorWidth(sensorWidth, nativeAspectRatio, mediaAspectRatio):
    """
    Calculate effective/utilised width of camera sensor when image/video is
    recorded at non-native aspect ratio.
    """

    cropRatio = (nativeAspectRatio[0] / nativeAspectRatio[1]
                 ) / (mediaAspectRatio[0] / mediaAspectRatio[1])
    return sensorWidth * cropRatio


# -----------------------------------------------------------------------------
# M A I N  F U N C T I O N

def main():
    
    
    
    videoSensorWidth = calcCropSensorWidth(CAMERA_SENSOR_WIDTH,
                                           (4, 3), (16, 9))
    pixelSize = calcPixelSize(VIDEO_WIDTH, videoSensorWidth,
                              CAMERA_FOCAL_LENGTH, OBJECT_DEPTH)

    # Note that amplitude here is peak-to-trough distance, as that is the
    # standard for tremor amplitude measurement:
    amplitudePixelDistance = processVideo()
    amplitude = amplitudePixelDistance * pixelSize / 10

    depthError = "N"  # TODO implement

    print("Pendulum amplitude (peak-to-trough) = %f +/- %s cm"
          % (amplitude, depthError))


main()
