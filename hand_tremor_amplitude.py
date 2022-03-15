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
    RESTING = 0
    POSTURAL = 1


# -----------------------------------------------------------------------------
# C O N S T A N T S   A N D   g l o b a l   V a r i a b l e s

# Camera parameters:
CAMERA_FOCAL_LENGTH = None  # mm, focal length of camera lens
CAMERA_FOCAL_LENGTH_STD = None  # mm, 35mm equiv. focal length of camera lens
CAMERA_NATIVE_ASPECT = None  # Native aspect ratio of camera sensor
CAMERA_VIDEO_ASPECT = None  # Aspect ratio of video recorded by camera

# Video file:
VIDEO_FILEPATH = None
videoWidth = -1  # resolution, pixels
videoFramerate = -1  # frames per second

# Frame numbers to start and end tremor measurement / hand tracking at:
START_FRAME = None
END_FRAME = None

# Depth measurement value from TrueDepth sensor, in cm:
HAND_DEPTH = None

# Display hand tracking in GUI, or use console only? (GUI -> much slower):
GUI_HAND_TRACKING = None

# Run from command line parameters without user prompting, or not:
AUTO_MODE = None

# Tremor type to measure:
tremorType = None

# Hand landmarks to track for tremor measurement:
chosenLandmarks = None
chosenLandmarksText = None

# OpenCV video capture for tremor video:
capture = None


# -----------------------------------------------------------------------------
# P A T H   P L O T T I N G

def plotPath(pathTime, path, pixelSize, amplitude):
    """
    Plot the path of oscillation of hand tremor over time.
    """

    minLeft = min(path)
    maxRight = max(path)

    path = list(map(lambda x: (x - minLeft - ((maxRight - minLeft) / 2))
                    * pixelSize, path))
    pathTime = list(map(lambda x: float((x - START_FRAME) / videoFramerate),
                        pathTime))

    xPoints = np.array(pathTime)
    yPoints = np.array(path)

    plt.plot(xPoints, yPoints)

    plt.axhline(y=max(path), color='dimgrey', linestyle='dotted')
    plt.axhline(y=min(path), color='dimgrey', linestyle='dotted')
    plt.axhline(y=amplitude/2, color='dimgrey', linestyle='dashed')
    plt.axhline(y=-(amplitude/2), color='dimgrey', linestyle='dashed')

    plt.xlabel('Time (seconds)')
    plt.ylabel('Tremor Amplitude (cm)')
    plt.title('Waveform of a Tremor with an Amplitude of\n' +
              '%.2fcm, Recorded at a Depth of %scm'
              % (amplitude, HAND_DEPTH))

    if plt.gcf().canvas.manager is not None:
        plt.gcf().canvas.manager.set_window_title('Tremor Waveform')

    plt.show()


# -----------------------------------------------------------------------------
# T R E M O R   A M P L I T U D E   C A L C U L A T I O N

def calcErrorFromHandTracking(amplitude):
    """
    Calculate the error due to inaccuracy in hand tracking by calculating the
    RMSE of measured tremor amplitude between the three chosen hand landmarks.

    Note that this relies on the subject's fingers staying the same distance
    apart from each other throughout the video. Works best with MCP (first
    knuckle) landmarks, as they cannot be moved apart like fingers can.

    Returns the error, in +/- cm.
    """

    avg = sum(amplitude) / 3

    rmse = math.sqrt((((amplitude[0] - avg) ** 2) + ((amplitude[1] - avg) ** 2)
                      + ((amplitude[2] - avg) ** 2)) / 3)

    # Divide by two to get +/- value, rather than total error value:
    return rmse / 2


def calcPixelDistAmplitudeFromPath(path):
    """
    Calculate tremor amplitude (in terms of pixels) from a tremor path.

    The path waveform is differentiated to find the peaks and troughs of the
    path. The differences between neighbouring peaks and troughs are calculated
    to obtain a collection of amplitude samples. The outliers of these are
    filtered, then the maximum is returned.
    """

    # Differentiate tremor path to find an estimate of the gradient (it is
    # only an estimate as the path is sampled, not continuous):
    pathDerivative = []
    for i in range(0, len(path) - 1):
        # https://math.stackexchange.com/questions/304069/use-a-set-of-data-points-from-a-graph-to-find-a-derivative
        gradientAtPoint = path[i+1] - path[i]
        pathDerivative.append(gradientAtPoint)

    # Find peaks and troughs of tremor path (they are at the point where
    # gradient changes from positive to negative or vice versa):
    peaksAndTroughs = []
    positiveGrad = pathDerivative[0] >= 0
    for i in range(1, len(pathDerivative)):
        # If gradient has switched from positive to negative or v.v.:
        if positiveGrad != (pathDerivative[i] >= 0):
            positiveGrad = not positiveGrad
            peaksAndTroughs.append(path[i])

    # Calculate the differences between neighbouring peaks and troughs, i.e.
    # the pixel distance of each 'oscillation' of tremor:
    amplitudeSamples = []
    for i in range(0, len(peaksAndTroughs) - 1):
        diff = abs(peaksAndTroughs[i] - peaksAndTroughs[i+1])
        amplitudeSamples.append(diff)

    # Calculate standard deviation and median of amplitude samples:
    stdDv = np.std(amplitudeSamples)
    median = np.median(amplitudeSamples)

    # Filter outliers of amplitude samples by +/- two standard deviations
    # (only filter high outliers as I don't care about low values):
    filteredAmplitudeSamples = filter(lambda ampl: ampl <= median
                                      + (2 * stdDv), amplitudeSamples)

    # Finally, amplitude = the maximum of the differences between tremor
    # path peaks and troughs:
    amplitude = max(filteredAmplitudeSamples)
    return amplitude


# -----------------------------------------------------------------------------
# H A N D   T R A C K I N G   &   V I D E O   P R O C E S S I N G

def selectLandmarks():
    """
    Set finger landmarks to track, to use in calculating tremor amplitude.
    """

    global chosenLandmarks, chosenLandmarksText

    if tremorType == Tremor.RESTING:
        print('Track MCP joints (first knuckles), PIP joints, DIP joints or'
              + ' tips of fingers?')
        inp = input('Type in MCP, PIP, DIP, or TIP: ').upper()[0]
        if inp == 'M':
            chosenLandmarks = [mp_hands.HandLandmark.RING_FINGER_MCP,
                               mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
                               mp_hands.HandLandmark.INDEX_FINGER_MCP]
            chosenLandmarksText = 'Index, middle and ring finger MCP joints'
        elif inp == 'P':
            chosenLandmarks = [mp_hands.HandLandmark.RING_FINGER_PIP,
                               mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
                               mp_hands.HandLandmark.INDEX_FINGER_PIP]
            chosenLandmarksText = 'Index, middle and ring finger PIP joints'
        elif inp == 'D':
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
        print('Track MCP joint, IP joint and tip of thumb, or track MCP, PIP' +
              ' and DIP joints of index finger?')
        inp = input('Type in thumb or index: ').upper()[0]
        if inp == 'T':
            chosenLandmarks = [mp_hands.HandLandmark.THUMB_MCP,
                               mp_hands.HandLandmark.THUMB_IP,
                               mp_hands.HandLandmark.THUMB_TIP]
            chosenLandmarksText = 'Thumb tip, IP joint and MCP joint'
        else:
            chosenLandmarks = [mp_hands.HandLandmark.INDEX_FINGER_MCP,
                               mp_hands.HandLandmark.INDEX_FINGER_PIP,
                               mp_hands.HandLandmark.INDEX_FINGER_DIP]
            chosenLandmarksText = 'Index finger MCP, PIP and DIP joints'


def computeTremorPath():
    """
    Process an input video to compute the path of the movement of a subject
    hand. This path is used to calculate tremor amplitude:
    """

    global capture

    minLeft = [videoWidth] * 3
    maxRight = [0] * 3
    path = [[], [], []]
    pathFrameNumbers = []
    failedFrames = 0

    minLeftFrame = None
    maxRightFrame = None

    print('-' * 80)
    print('Computing tremor amplitude for %s...' % VIDEO_FILEPATH)

    # Configure mediapipe hand detector:
    detector = mp_hands.Hands(
        static_image_mode=False,  # False -> treat images as video sequence, to track hands between images. Allows faster and more accurate tracking.
        max_num_hands=1,  # Expect maximum of 1 hand in frame
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1  # Model accuracy, 1 == more accurate but slower
    )

    frameN = 0

    while (True):
        readSuccess, frame = capture.read()

        # i.e. if a frame was read from the video:
        if readSuccess:
            frameN += 1

            # Only process frame if within chosen time range:
            if START_FRAME <= frameN <= END_FRAME:

                # Rotate postural tremor videos by 90 degrees (to allow tremor
                # to be measured in the x direction):
                if tremorType == Tremor.POSTURAL:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

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

                # Show hand landmarks in a GUI window:
                if GUI_HAND_TRACKING:
                    cv2.namedWindow('Hand Tracking')
                    cv2.imshow('Hand Tracking', frameWithLandmarks)
                    cv2.waitKey(1)

                # Array for storing x coordinates of finger pip joints:
                fingerLandmarkX = [None] * 3

                for i in range(0, 3):
                    # Get x position (in pixels) of the finger landmarks:
                    # (n.b. landmark coordinates are normalised to be between 0
                    #  and 1, so must be multiplied by video width)
                    fingerLandmarkX[i] = round(landmarks.landmark[
                        chosenLandmarks[i]].x * videoWidth)

                    # For each landmark, if it is the furthest left or right
                    # of any frame so far, save the x coordinate and frame:
                    if fingerLandmarkX[i] < minLeft[i]:
                        minLeft[i] = fingerLandmarkX[i]
                        if i == 1:
                            minLeftFrame = frameWithLandmarks
                    if fingerLandmarkX[i] > maxRight[i]:
                        maxRight[i] = fingerLandmarkX[i]
                        if i == 1:
                            maxRightFrame = frameWithLandmarks

                    path[i].append(fingerLandmarkX[i])

                pathFrameNumbers.append(frameN)

                print('Processing frame ' + str(frameN) + '/' + str(END_FRAME)
                      + ' | landmark x positions: '
                      + str(minLeft[0]) + ' <= ' + str(fingerLandmarkX[0])
                      + ' <= ' + str(maxRight[0]) + ', '
                      + str(minLeft[1]) + ' <= ' + str(fingerLandmarkX[1])
                      + ' <= ' + str(maxRight[1]) + ', '
                      + str(minLeft[2]) + ' <= ' + str(fingerLandmarkX[2])
                      + ' <= ' + str(maxRight[2]),
                      end='\r')

        # Catch video read errors, including reaching end of the video:
        else:
            print('\n', end='')
            if GUI_HAND_TRACKING:
                cv2.destroyAllWindows()
            break

    # Left and right most frames can be manually viewed to check for erroneous
    # landmark placement which may affect measurement:
    cv2.imwrite('data/leftMostFrame.jpg', minLeftFrame)
    cv2.imwrite('data/rightMostFrame.jpg', maxRightFrame)

    capture.release()

    return path, pathFrameNumbers, failedFrames


# -----------------------------------------------------------------------------
# C A M E R A   P A R A M E T E R   C A L C U L A T I O N

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
# S E T U P   F U N C T I O N S

def loadConstantsFromConfigFile():
    """
    Load a configuration file and set the constants from the values in the
    file.
    """

    global CAMERA_FOCAL_LENGTH, CAMERA_FOCAL_LENGTH_STD
    global CAMERA_NATIVE_ASPECT, CAMERA_VIDEO_ASPECT
    global VIDEO_FILEPATH
    global START_FRAME, END_FRAME
    global HAND_DEPTH
    global GUI_HAND_TRACKING
    global AUTO_MODE
    
    # n.b. see global definitions at the top of the file to see their meaning.

    CAMERA_FOCAL_LENGTH = 2.87  # mm, focal length of camera lens
    CAMERA_FOCAL_LENGTH_STD = 32  # mm, 35mm equiv. focal length of camera lens
    CAMERA_NATIVE_ASPECT = (3, 4)  # Native aspect ratio of camera sensor
    CAMERA_VIDEO_ASPECT = (9, 16)  # Aspect ratio of video recorded by camera

    VIDEO_FILEPATH = 'data/phase3/postural_o_50_5.MOV'

    START_FRAME = 1
    END_FRAME = 900

    HAND_DEPTH = int(VIDEO_FILEPATH.split('_')[2])

    GUI_HAND_TRACKING = False

    AUTO_MODE = False


def selectTremorType():
    """
    Set whether the input video which is to be processed depicts resting or
    postural tremor.
    """

    global tremorType

    inp = input('Measure resting or postural tremor? (r/p): ').lower()[0]
    if inp == 'r':
        tremorType = Tremor.RESTING
    else:
        tremorType = Tremor.POSTURAL


def openCaptureAndGetVideoInfo():
    global capture, videoWidth, videoFramerate

    capture = cv2.VideoCapture(VIDEO_FILEPATH)

    if tremorType == Tremor.RESTING:
        videoWidth = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        videoHeight = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    else:
        # Swap width and height, since postural videos are rotated 90 degrees:
        videoWidth = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        videoHeight = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))

    videoFramerate = round(capture.get(cv2.CAP_PROP_FPS))

    # Check that the aspect ratio in the config file matches the aspect ratio
    # of the loaded video file:
    aspectRatio = videoWidth / videoHeight
    if aspectRatio != CAMERA_VIDEO_ASPECT[0] / CAMERA_VIDEO_ASPECT[1]:
        print('WARNING: Camera video aspect ratio may be wrong. This can' +
              ' cause incorrect amplitude measurement. Consult the config' +
              ' file to check this value. Alternatively this can be caused' +
              ' by incorrectly selecting rest or postural tremor mode.')


def printConfig():
    print('-' * 80)
    print('Tremor type to measure          : ' + tremorType.name)
    print('Hand landmarks to track         : ' + chosenLandmarksText)
    print('Video file path                 : ' + VIDEO_FILEPATH)
    resFpsStr = (str(videoWidth) + 'x' + str(int(videoWidth /
                 CAMERA_VIDEO_ASPECT[0] * CAMERA_VIDEO_ASPECT[1])) + ', '
                 + str(videoFramerate) + 'fps')
    print('Video resolution and frame rate : ' + resFpsStr)
    print('Camera focal length             : ' + str(CAMERA_FOCAL_LENGTH)
          + 'mm')
    print('Camera 35mm equiv. focal length : ' + str(CAMERA_FOCAL_LENGTH_STD)
          + 'mm')
    print('Camera aspect ratio             : ' + str(CAMERA_NATIVE_ASPECT[0])
          + ':' + str(CAMERA_NATIVE_ASPECT[1]))
    print('Video aspect ratio              : ' + str(CAMERA_VIDEO_ASPECT[0])
          + ':' + str(CAMERA_VIDEO_ASPECT[1]))
    print('Hand depth measurement          : ' + str(HAND_DEPTH) + 'cm')

    if not AUTO_MODE:
        inp = input('Check above values; ok to continue? (y/n): ').lower()[0]
        if inp != 'y':
            sys.exit()


# -----------------------------------------------------------------------------
# M A I N

def main():
    print('-' * 80)
    print('T R E M O R   A M P L I T U D E   M E A S U R E M E N T')
    print('-' * 80)

    # Load constants from config file:
    loadConstantsFromConfigFile()

    # Select whether the input video depicts resting or postural tremor:
    if not AUTO_MODE:
        selectTremorType()

    # Open video capture and get video info:
    openCaptureAndGetVideoInfo()

    # Choose hand landmarks to track for tremor measurement:
    if not AUTO_MODE:
        selectLandmarks()

    # Print the configuration before starting video analysis:
    printConfig()

    # Compute the path of movement of the hand in the input video:
    path, pathTime, failedFrames = computeTremorPath()

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

    # Thus, calculate the real-world size, in cm, of each pixel in the video at
    # a given depth:
    pixelSize, pixelSizeError = calcPixelSize(videoWidth,
                                              videoSensorWidth,
                                              CAMERA_FOCAL_LENGTH,
                                              HAND_DEPTH)

    # Note that amplitude here is peak-to-trough distance, as that is the
    # standard for tremor amplitude measurement:
    amplitudePixelDistance = [None] * 3
    amplitude = [None] * 3
    for i in range(0, 3):
        # Calculate amplitude in pixels:
        amplitudePixelDistance[i] = calcPixelDistAmplitudeFromPath(path[i])
        # Convert amplitude from pixels to cm:
        amplitude[i] = amplitudePixelDistance[i] * pixelSize

    # Take the average of the tremor amplitudes from the three landmarks to
    # obtain a final value for tremor amplitude:
    finalAmplitude = sum(amplitude) / len(amplitude)

    # Calculate error in the amplitute due to the error in depth measurement
    # from the TrueDepth sensor, which translates into error in the value of
    # pixelSize:
    amplitudeError = finalAmplitude * pixelSizeError

    # Calcluate error due to each pixel representing a discrete area of
    # continuous space; i.e. occurs since a hand landmark may not be at the
    # very centre of a pixel, but somewhere between either side:
    pixelSizeError = (0.5 * pixelSize) * 2  # Bad code, to emphasise meaning

    # Calculate error due to inaccuracy in hand tracking:
    trackingError = calcErrorFromHandTracking(amplitude)

    totalError = amplitudeError + pixelSizeError + trackingError

    # Print results to console:
    print('-' * 80)
    print('Tremor amplitude = %.2f +/- %.2f cm' % (finalAmplitude, totalError))
    print('-' * 80)
    print('Error breakdown:')
    print('  1. Error due to depth sensor inaccuracy  : +/- %.2f cm'
          % amplitudeError)
    print('  2. Error due to pixel size discretion    : +/- %.2f cm'
          % pixelSizeError)
    print('  3. Error due to hand tracking inaccuracy : +/- %.2f cm'
          % trackingError)
    print('-' * 80)
    print('n.b. Consult the saved left-most and right-most frames in the' +
          ' data folder to\nascertain the amplitude measurement, by checking' +
          ' the hand landmark placement\nfor discrepencies.')
    print('-' * 80)
    print('Frames where hand detection failed: %d' % failedFrames)
    print('-' * 80)

    # Plot the path of the hand tremor over time:
    plotPath(pathTime, path[1], pixelSize, finalAmplitude)


main()
