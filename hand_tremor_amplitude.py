# -*- coding: utf-8 -*-

from enum import Enum
import sys
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math

import mediapipe as mp
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


class Tremor(Enum):
    RESTING = 1
    POSTURAL = 2


# -----------------------------------------------------------------------------
# C O N S T A N T S  A N D  G L O B A L S

# Camera parameters:
CAMERA_FOCAL_LENGTH = 2.87  # mm, focal length of camera lens
CAMERA_FOCAL_LENGTH_STD = 32  # mm, 35mm equiv. focal length of camera lens
CAMERA_NATIVE_ASPECT = (3, 4)  # Native aspect ratio of camera sensor
CAMERA_VIDEO_ASPECT = (9, 16)  # Aspect ratio of video recorded by camera

# Video file:
VIDEO_FILEPATH = 'data/phase3/resting_o_100_10.MOV'
VIDEO_WIDTH = 1080  # resolution, pixels
VIDEO_FRAMERATE = 60  # frames per second
START_FRAME = 1  # Frame of video to start tremor measurement at
END_FRAME = 15 * VIDEO_FRAMERATE  # Frame of video to end tremor measurement at

# Depth measurement value from TrueDepth sensor, in cm:
HAND_DEPTH = int(VIDEO_FILEPATH.split('_')[2])

# Tremor type to measure:
tremorType = None

# Hand landmarks to track for tremor measurement:
chosenLandmarks = None
chosenLandmarksText = None

# Values for plot title:
TARGET_AMPLITUDE = VIDEO_FILEPATH.split('_')[3].split('.')[0]  # cm


# -----------------------------------------------------------------------------
# P A T H  P L O T T I N G

def plotPath(pathTime, path, minLeft, maxRight, pixelSize):
    """
    Plot the path of oscillation of hand tremor over time.
    """

    path = list(map(lambda x: (x - minLeft - ((maxRight - minLeft) / 2))
                    * pixelSize, path))
    pathTime = list(map(lambda x: float((x - START_FRAME) / VIDEO_FRAMERATE),
                        pathTime))

    xPoints = np.array(pathTime)
    yPoints = np.array(path)

    plt.plot(xPoints, yPoints)

    plt.xlabel('Time (seconds)')
    plt.ylabel('Tremor Amplitude (cm)')
    plt.title('Measured Tremor with a Target Amplitude\n' +
              'of %scm, Recorded at a Depth of %scm'
              % (TARGET_AMPLITUDE, HAND_DEPTH))

    plt.show()


# -----------------------------------------------------------------------------
# H A N D  T R A C K I N G  /  V I D E O  P R O C E S S I N G

def selectLandmarks():
    """
    Set finger landmarks to track, to use in calculating tremor amplitude.
    """

    global chosenLandmarks, chosenLandmarksText

    if tremorType == Tremor.RESTING:
        print('Track finger MCP joint (first knuckle), PIP joint, DIP joint or'
              + ' finger tip (nail)?')
        joint = input('Type in MCP, PIP, DIP, or TIP: ').upper()
        if joint == 'MCP':
            chosenLandmarks = [mp_hands.HandLandmark.RING_FINGER_MCP,
                               mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
                               mp_hands.HandLandmark.INDEX_FINGER_MCP]
            chosenLandmarksText = 'Index, middle and ring finger MCP joints'
        elif joint == 'PIP':
            chosenLandmarks = [mp_hands.HandLandmark.RING_FINGER_PIP,
                               mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
                               mp_hands.HandLandmark.INDEX_FINGER_PIP]
            chosenLandmarksText = 'Index, middle and ring finger PIP joints'
        elif joint == 'DIP':
            chosenLandmarks = [mp_hands.HandLandmark.RING_FINGER_DIP,
                               mp_hands.HandLandmark.MIDDLE_FINGER_DIP,
                               mp_hands.HandLandmark.INDEX_FINGER_DIP]
            chosenLandmarksText = 'Index, middle and ring finger DIP joints'
        else:
            chosenLandmarks = [mp_hands.HandLandmark.RING_FINGER_TIP,
                               mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                               mp_hands.HandLandmark.INDEX_FINGER_TIP]
            chosenLandmarksText = 'Index, middle and ring finger tips'

    else:
        chosenLandmarks = []
        chosenLandmarksText = 'NOT YET IMPLEMENTED'


def computeTremorPath():
    """
    Process an input video to compute the path of the movement of a subject
    hand. This path is used to calculate tremor amplitude:
    """

    minLeft = [VIDEO_WIDTH] * 3
    maxRight = [0] * 3
    path = []
    pathFrameNumbers = []
    failedFrames = 0

    minLeftFrame = None
    maxRightFrame = None

    print('------------------------------------------------------------------')
    print('Computing tremor amplitude for %s...' % VIDEO_FILEPATH)
    print('------------------------------------------------------------------')

    # Configure mediapipe hand detector:
    detector = mp_hands.Hands(
        static_image_mode=False,  # False -> treat images as video sequence, to track hands between images. Allows faster and more accurate tracking.
        max_num_hands=1,  # Expect maximum of 1 hand in frame
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1  # Model accuracy, 1 == more accurate but slower
    )

    capture = cv2.VideoCapture(VIDEO_FILEPATH)
    frameN = 0

    while (True):
        readSuccess, frame = capture.read()

        # i.e. if a frame was read from the video:
        if readSuccess:
            frameN += 1

            # Only process frame if within chosen time range:
            if START_FRAME <= frameN <= END_FRAME:

                # Apply hand detection:
                frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detectionResults = detector.process(frameRGB)

                # Skip processing of frame if hand was not detected:
                if not detectionResults.multi_hand_landmarks:
                    failedFrames += 1
                    continue

                # Get landmarks for (first) detected hand in image (note only
                # one hand can be detected but it still must be selected):
                landmarks = detectionResults.multi_hand_landmarks[0]

                # Draw hand landmarks onto frame:
                frameWithLandmarks = frame.copy()
                mp_drawing.draw_landmarks(
                    frameWithLandmarks,
                    landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # cv2.imwrite('tempImg.jpg', frameWithLandmarks)

                # Array for storing x coordinates of finger pip joints:
                fingerLandmarkX = [None] * 3

                # Get x position (in terms of pixels) of the finger landmarks
                # which were chosen above:
                # (n.b. landmark coordinates are normalised to be between 0
                #  and 1, so must be multiplied by video width)
                for i in range(0, 3):
                    fingerLandmarkX[i] = round(landmarks.landmark[
                        chosenLandmarks[i]].x * VIDEO_WIDTH)

                    if fingerLandmarkX[i] < minLeft[i]:
                        minLeft[i] = fingerLandmarkX[i]
                        if i == 1:
                            minLeftFrame = frameWithLandmarks
                    if fingerLandmarkX[i] > maxRight[i]:
                        maxRight[i] = fingerLandmarkX[i]
                        if i == 1:
                            maxRightFrame = frameWithLandmarks

                # TODO: record path for all 3 fingers.
                path.append(fingerLandmarkX[1])
                pathFrameNumbers.append(frameN)

                print('Frame ' + str(frameN) + '/' + str(END_FRAME)
                      + ' | landmark x positions: '
                      + str(minLeft[0]) + ' <= ' + str(fingerLandmarkX[0]) + ' <= ' + str(maxRight[0]) + ', '
                      + str(minLeft[1]) + ' <= ' + str(fingerLandmarkX[1]) + ' <= ' + str(maxRight[1]) + ', '
                      + str(minLeft[2]) + ' <= ' + str(fingerLandmarkX[2]) + ' <= ' + str(maxRight[2]),
                      end='\r')

        # Catch video read errors, including reaching end of the video:
        else:
            print('\n', end='')
            break

    # Left and right most frames can be manually viewed to check for erroneous
    # landmark placement which may affect measurement:
    cv2.imwrite('data/leftMostFrame.jpg', minLeftFrame)
    cv2.imwrite('data/rightMostFrame.jpg', maxRightFrame)

    capture.release()

    return path, pathFrameNumbers, minLeft[1], maxRight[1], failedFrames


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
# M A I N

def printConfig():
    print('------------------------------------------------------------------')
    print('Tremor type to measure          : ' + tremorType.name)
    print('Hand landmarks to track         : ' + chosenLandmarksText)
    print('Video file path                 : ' + VIDEO_FILEPATH)
    print('Video resolution and fps        : ' + 'nnnnxnnnn, nnfps')
    print('Camera focal length             : ' + str(CAMERA_FOCAL_LENGTH) + 'mm')
    print('Camera 35mm equiv. focal length : ' + str(CAMERA_FOCAL_LENGTH_STD) + 'mm')
    print('Camera aspect ratio             : ' + str(CAMERA_NATIVE_ASPECT[0])
          + ':' + str(CAMERA_NATIVE_ASPECT[1]))
    print('Video aspect ratio              : ' + str(CAMERA_VIDEO_ASPECT[0])
          + ':' + str(CAMERA_VIDEO_ASPECT[1]))
    print('Hand depth measurement          : ' + str(HAND_DEPTH) + 'cm')
    
    if input('Check the above values; ok to continue? (y/n): ') == 'y':
        return
    else:
        sys.exit()
    # TODO: print run mode (resting or postural), camera specs from constants, etc. before running.


def main():
    print('------------------------------------------------------------------')
    print('T R E M O R  A M P L I T U D E  M E A S U R E M E N T')
    print('------------------------------------------------------------------')

    global tremorType
    tremorType = Tremor.RESTING

    selectLandmarks()

    printConfig()

    # Compute the path of movement of the hand in the input video:
    path, pathTime, minLeft, maxRight, failedFrames = computeTremorPath()

    # Calculate camera sensor width from more readily-available camera
    # specifications:
    cameraSensorWidth, __ = calcSensorSize(CAMERA_FOCAL_LENGTH,
                                           CAMERA_FOCAL_LENGTH_STD,
                                           CAMERA_NATIVE_ASPECT)

    # Calculate the width of the sensor which is utilised for video recording
    # (as video is generally a different aspect ratio to sensor aspect ratio):
    videoSensorWidth = calcCropSensorWidth(cameraSensorWidth,
                                           CAMERA_NATIVE_ASPECT,
                                           CAMERA_VIDEO_ASPECT)

    # Thus, calculate the size of each pixel in the video at a given depth:
    pixelSize, pixelSizeError = calcPixelSize(VIDEO_WIDTH,
                                              videoSensorWidth,
                                              CAMERA_FOCAL_LENGTH,
                                              HAND_DEPTH)

    plotPath(pathTime, path, minLeft, maxRight, pixelSize)

    # Note that amplitude here is peak-to-trough distance, as that is the
    # standard for tremor amplitude measurement:
    amplitudePixelDistance = maxRight - minLeft
    amplitude = amplitudePixelDistance * pixelSize

    # There are two measurable sources of error:
    #   1. From the depth measurement from the TrueDepth sensor, which
    #      translates into error in the value of pixelSize.
    #   2. From each pixel representing discrete areas of continuous space,
    #      i.e. the hand landmark may not be at the very centre of a
    #      pixel, but somewhere between either side.
    amplitudeError = amplitudePixelDistance * pixelSizeError
    amplitudeError += (0.5 * pixelSize) * 2  # Bad code, to emphasise meaning

    print('------------------------------------------------------------------')
    print('Tremor amplitude (peak-to-trough) = %.1f +/- %.2f cm'
          % (amplitude, amplitudeError))
    # TODO: print error breakdown (pixel error, depth error, hand detection error).
    print('------------------------------------------------------------------')
    print('N.B. The above error bounds do not account for error from hand ' +
          'detection. Consult the saved left-most and right-most frames ' +
          'in the data folder to analyse this error by checking the hand ' +
          'landmark placement for discrepencies.')
    print('Frames where hand detection failed: %d' % failedFrames)


main()
