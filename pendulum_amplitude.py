# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import cv2


CAMERA_FOCAL_LENGTH = 2.87  # mm
CAMERA_FOCAL_LENGTH_STD = 32  # mm

OBJECT_DEPTH = 40  # cm

VIDEO_WIDTH = 1080
VIDEO_FILEPATH = './IMG_1224.mov'
START_FRAME = 570
END_FRAME = 9999


# -----------------------------------------------------------------------------
# V I D E O  P R O C E S S I N G

def handleFrame(frame):
    cv2.imwrite('1.jpg', frame)
    grayscaled = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grayscaled, (5, 5), 0)
    (T, thresholded) = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)
    cv2.imwrite('2.jpg', thresholded)
    
    # TODO detect blob and find midpoint. Return this value


def processVideo():
    minLeft = 0
    maxRight = VIDEO_WIDTH
    
    capture = cv2.VideoCapture(VIDEO_FILEPATH)
    frameN = 0
    while (True):
        success, frame = capture.read()
        if success:
            frameN += 1
            if START_FRAME <= frameN <= END_FRAME:
                handleFrame(frame)
                break
        else:
            break
    capture.release()
    print(frameN)
    
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
