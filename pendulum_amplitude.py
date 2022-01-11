# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import cv2


CAMERA_FOCAL_LENGTH = 2.87  # mm, focal length of camera lens
CAMERA_FOCAL_LENGTH_STD = 32  # mm, 35mm equiv. focal length of camera lens
CAMERA_NATIVE_ASPECT = (4, 3)  # Native aspect ratio of camera sensor
CAMERA_VIDEO_ASPECT = (16, 9)  # Aspect ratio of video recorded by camera

VIDEO_FILEPATH = 'data/pendulum_80_30.mov'
VIDEO_WIDTH = 1080  # resolution, pixels
VIDEO_FRAMERATE = 60  # frames per second

START_FRAME = 600  # Frame of video to start analysis at
END_FRAME = 9999  # Frame of video to end analysis at

MEASURED_OBJECT_DEPTH = 80  # cm, value from TrueDepth sensor

# The following are used in the plot title only:
REAL_OBJECT_DEPTH = VIDEO_FILEPATH.split('_')[1]  # cm
REAL_AMPLITUDE = VIDEO_FILEPATH.split('_')[2].split('.')[0]  # cm


# -----------------------------------------------------------------------------
# P A T H  P L O T T I N G


def plotPath(pathTime, path, minLeft, maxRight, pixelSize):
    path = list(map(lambda x: (x - minLeft - ((maxRight - minLeft) / 2))
                    * pixelSize / 10, path))
    pathTime = list(map(lambda x: float((x - START_FRAME) / VIDEO_FRAMERATE),
                        pathTime))

    xPoints = np.array(pathTime)
    yPoints = np.array(path)

    plt.plot(xPoints, yPoints)

    plt.xlabel('Time (seconds)')
    plt.ylabel('Pendulum Amplitude (cm)')
    plt.title('Measured Pendulum Amplitude Over Time with a\n' +
              '%scm Initial Amplitude at a Depth of %scm'
              % (REAL_AMPLITUDE, REAL_OBJECT_DEPTH))

    plt.show()


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


def findPendulumBobMidpoint(frame, detector, firstFrame):

    # Pre-process image ready for blob detection:
    grayscaled = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grayscaled, (5, 5), 0)
    (__, thresholded) = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)

    keypoints = detector.detect(thresholded)

    if firstFrame:
        imgKP = cv2.drawKeypoints(frame, keypoints, np.array([]), (0, 0, 255),
                                  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imwrite('data/firstFrame.jpg', imgKP)
        plt.imshow(cv2.cvtColor(imgKP, cv2.COLOR_BGR2RGB))
        plt.show()
        input("Check first frame, and press enter to continue...")
        print("Continuing...")

    if len(keypoints) != 1:
        return -1

    # Return the midpoint of the keypoint:
    for kp in keypoints:
        return round(kp.pt[0])


def computePendulumPath():
    minLeft = VIDEO_WIDTH
    maxRight = 0
    path = []
    pathFrameNumbers = []
    failedFrames = 0

    print("Computing pendulum path for %s ..." % VIDEO_FILEPATH)

    capture = cv2.VideoCapture(VIDEO_FILEPATH)
    frameN = 0
    detector = setUpBlobDetector()
    while (True):
        success, frame = capture.read()
        if success:
            frameN += 1
            if START_FRAME <= frameN <= END_FRAME:
                mp = findPendulumBobMidpoint(frame, detector,
                                             frameN == START_FRAME)
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

    return path, pathFrameNumbers, minLeft, maxRight, failedFrames


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
    imageResolution : The width of an image in pixels.
    sensorWidth : The width of the camera sensor in mm.
    lensFocalLength : Focal length of the lens, in mm.
    depth : The depth at which the width of a pixel is to be calculated, in cm.

    Returns
    -------
    pixelSize : Size of a pixel at the given depth, in mm.
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
    path, pathTime, minLeft, maxRight, failedFrames = computePendulumPath()

    videoSensorWidth = calcCropSensorWidth(CAMERA_SENSOR_WIDTH,
                                           CAMERA_NATIVE_ASPECT,
                                           CAMERA_VIDEO_ASPECT)
    pixelSize = calcPixelSize(VIDEO_WIDTH, videoSensorWidth,
                              CAMERA_FOCAL_LENGTH, MEASURED_OBJECT_DEPTH)

    plotPath(pathTime, path, minLeft, maxRight, pixelSize)

    # Note that amplitude here is peak-to-trough distance, as that is the
    # standard for tremor amplitude measurement:
    amplitudePixelDistance = maxRight - minLeft
    amplitude = amplitudePixelDistance * pixelSize / 10

    depthError = "N"  # TODO implement

    print("Pendulum amplitude (peak-to-trough) = %.1f +/- %s cm"
          % (amplitude, depthError))
    print("Frames where pendulum bob detection failed: %d" % failedFrames)


main()
