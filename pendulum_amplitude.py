# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import cv2
import math


CAMERA_FOCAL_LENGTH = 2.87  # mm, focal length of camera lens
CAMERA_FOCAL_LENGTH_STD = 32  # mm, 35mm equiv. focal length of camera lens
CAMERA_NATIVE_ASPECT = (3, 4)  # Native aspect ratio of camera sensor
CAMERA_VIDEO_ASPECT = (9, 16)  # Aspect ratio of video recorded by camera

VIDEO_FILEPATH = 'data/phase2/pendulum_40_8.mov'
VIDEO_WIDTH = 1080  # resolution, pixels
VIDEO_FRAMERATE = 60  # frames per second

START_FRAME = 660  # Frame of video to start analysis at
END_FRAME = 9999  # Frame of video to end analysis at

MEASURED_OBJECT_DEPTH = int(VIDEO_FILEPATH.split('_')[1])  # cm, value from TrueDepth sensor

# The following are used in the plot title only:
REAL_OBJECT_DEPTH = VIDEO_FILEPATH.split('_')[1]  # cm
REAL_AMPLITUDE = VIDEO_FILEPATH.split('_')[2].split('.')[0]  # cm


# -----------------------------------------------------------------------------
# P A T H  P L O T T I N G

def plotPath(pathTime, path, minLeft, maxRight, pixelSize):
    """
    Plot the path of oscillation of a pendulum bob over time.
    """

    path = list(map(lambda x: (x - minLeft - ((maxRight - minLeft) / 2))
                    * pixelSize, path))
    pathTime = list(map(lambda x: float((x - START_FRAME) / VIDEO_FRAMERATE),
                        pathTime))

    xPoints = np.array(pathTime)
    yPoints = np.array(path)

    plt.plot(xPoints, yPoints)

    plt.xlabel('Time (seconds)')
    plt.ylabel('Pendulum Amplitude (cm)')
    plt.title('Measured Pendulum Amplitude Over Time with a %scm\n'
              % REAL_AMPLITUDE +
              'Initial Displacement, Recorded at a Depth of %scm'
              % REAL_OBJECT_DEPTH)

    plt.show()


# -----------------------------------------------------------------------------
# V I D E O  P R O C E S S I N G

def setUpBlobDetector():
    """
    Configure parameters for a cv2 blob detector, and returns the detector.
    """

    params = cv2.SimpleBlobDetector_Params()

    params.minThreshold = 0
    params.maxThreshold = 255

    params.filterByArea = True
    params.minArea = 1500
    params.maxArea = 25000

    params.filterByCircularity = False
    params.filterByColor = False
    params.filterByConvexity = False
    params.filterByInertia = False

    detector = cv2.SimpleBlobDetector_create(params)
    return detector


def findPendulumBobMidpoint(frame, detector, firstFrame):
    """
    Pre-process and apply blob detection to a frame of video to find the
    mid-point x-coordinate of the pendulum bob.
    """

    # Pre-process image ready for blob detection:
    grayscaled = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grayscaled, (5, 5), 0)
    (__, thresholded) = cv2.threshold(blurred, 75, 255, cv2.THRESH_BINARY)

    keypoints = detector.detect(thresholded)

    imgKP = cv2.drawKeypoints(frame, keypoints, np.array([]), (0, 0, 255),
                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    if firstFrame:
        cv2.imwrite('data/firstFrame.jpg', imgKP)
        plt.imshow(cv2.cvtColor(imgKP, cv2.COLOR_BGR2RGB))
        plt.show()
        input('Check first frame, and press enter to continue...')
        print('Continuing...')

    if len(keypoints) != 1:
        return -1, None

    # Return the midpoint of the keypoint:
    for kp in keypoints:
        return round(kp.pt[0]), imgKP


def computePendulumPath():
    """
    Process an input video to compute the path of oscillation of the
    pendulum bob.
    """

    minLeft = VIDEO_WIDTH
    maxRight = 0
    path = []
    pathFrameNumbers = []
    failedFrames = 0

    minLeftFrame = None
    maxRightFrame = None

    print('Computing pendulum path for %s ...' % VIDEO_FILEPATH)

    capture = cv2.VideoCapture(VIDEO_FILEPATH)
    frameN = 0
    detector = setUpBlobDetector()
    while (True):
        success, frame = capture.read()
        if success:
            frameN += 1
            if START_FRAME <= frameN <= END_FRAME:
                mp, frameKP = findPendulumBobMidpoint(frame, detector,
                                                      frameN == START_FRAME)
                if mp == -1:
                    failedFrames += 1
                else:
                    path.append(mp)
                    pathFrameNumbers.append(frameN)
                    if mp < minLeft:
                        minLeft = mp
                        minLeftFrame = frameKP
                    if mp > maxRight:
                        maxRight = mp
                        maxRightFrame = frameKP
        else:
            break
    cv2.imwrite('data/leftMostFrame.jpg', minLeftFrame)
    cv2.imwrite('data/rightMostFrame.jpg', maxRightFrame)
    capture.release()

    return path, pathFrameNumbers, minLeft, maxRight, failedFrames


# -----------------------------------------------------------------------------
# C A M E R A  P A R A M E T E R  C A L C U L A T I O N

def getDepthError(depth):
    """
    Return the absolute error in a depth value from a TrueDepth sensor, which
    varies depending on the depth value. These are OBSERVED values from my
    experimentation in phase 2 data collection.

    Parameters
    ----------
    depth : number
        The depth value reported by the TrueDepth sensor, in cm

    Returns
    -------
    error : float
        Error n, in the format n +/- cm
    """

    if depth <= 45:
        return 0.115470054
    elif 45 < depth <= 55:
        return 0.081649658
    elif 55 < depth <= 65:
        return 0.141421356
    elif 65 < depth <= 75:
        return 0.203442594
    elif 75 < depth <= 85:
        return 0.324893145
    elif 85 < depth <= 95:
        return 0.37155828
    else:
        return 0.37859389


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
    pixelSize : Size of a pixel at the given depth, in cm.
    pixelSizeErr : The error bounds of pixelSize, in +/- cm.
    """

    viewWidth = depth * sensorWidth / lensFocalLength
    viewWidthErr = (((getDepthError(depth) + depth) * sensorWidth /
                    lensFocalLength) - viewWidth)
    return viewWidth / imageResolution, viewWidthErr / imageResolution


def calcCropSensorWidth(sensorWidth, nativeAspectRatio, mediaAspectRatio):
    """
    Calculate effective/utilised width of camera sensor when image/video is
    recorded at non-native aspect ratio.
    """

    cropRatio = (nativeAspectRatio[1] / nativeAspectRatio[0]
                 ) / (mediaAspectRatio[1] / mediaAspectRatio[0])
    return sensorWidth * cropRatio


def calcSensorSize(focalLength, focalLength35mmEquiv, sensorAspectRatio):
    """
    Calculate the height and width of a camera sensor from its focal length,
    35mm equivalent focal length and aspect ratio.
    """

    cropFactor = focalLength35mmEquiv / focalLength
    # n.b. 43.27mm is the diagonal size of a full-frame 35mm sensor:
    sensorDiagonalLength = 43.27 / cropFactor
    angle = math.atan(sensorAspectRatio[1] / sensorAspectRatio[0])
    sensorWidth = math.cos(angle) * sensorDiagonalLength
    sensorHeight = math.sin(angle) * sensorDiagonalLength
    return sensorWidth, sensorHeight


# -----------------------------------------------------------------------------
# M A I N  F U N C T I O N

def main():
    path, pathTime, minLeft, maxRight, failedFrames = computePendulumPath()

    cameraSensorWidth, __ = calcSensorSize(CAMERA_FOCAL_LENGTH,
                                           CAMERA_FOCAL_LENGTH_STD,
                                           CAMERA_NATIVE_ASPECT)
    videoSensorWidth = calcCropSensorWidth(cameraSensorWidth,
                                           CAMERA_NATIVE_ASPECT,
                                           CAMERA_VIDEO_ASPECT)
    pixelSize, pixelSizeError = calcPixelSize(VIDEO_WIDTH, videoSensorWidth,
                                              CAMERA_FOCAL_LENGTH,
                                              MEASURED_OBJECT_DEPTH)

    plotPath(pathTime, path, minLeft, maxRight, pixelSize)

    # Note that amplitude here is peak-to-trough distance, as that is the
    # standard for tremor amplitude measurement:
    amplitudePixelDistance = maxRight - minLeft
    amplitude = amplitudePixelDistance * pixelSize

    # There are two measurable sources of error:
    #   1. From the depth measurement from the TrueDepth sensor, which
    #      translates into error in the value of pixelSize.
    #   2. From each pixel representing discrete areas of continuous space,
    #      i.e. the pendulum's midpoint may not be at the very centre of a
    #      pixel, but somewhere between either side.
    amplitudeError = amplitudePixelDistance * pixelSizeError
    amplitudeError += (0.5 * pixelSize) * 2  # Bad code, to emphasise meaning

    print('------------------------------------------------------------------')
    print('Pendulum amplitude (peak-to-trough) = %.1f +/- %.2f cm'
          % (amplitude, amplitudeError))
    print('------------------------------------------------------------------')
    print('N.B. The above error bounds do not account for human error from ' +
          'releasing the pendulum at a different displacement to that which ' +
          'was intended. Consult the saved left-most and right-most frames ' +
          'in the data folder to analyse this human error.')
    print('Frames where pendulum bob detection failed: %d' % failedFrames)


main()
