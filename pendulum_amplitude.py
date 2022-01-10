# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt


CAMERA_FOCAL_LENGTH = 2.87
CAMERA_FOCAL_LENGTH_STD = 32

OBJECT_DEPTH = 40


# -----------------------------------------------------------------------------
# C A M E R A  R E L A T E D  F U N C T I O N S

CAMERA_SENSOR_WIDTH = 2.328
CAMERA_SENSOR_HEIGHT = 3.104

VIDEO_WIDTH = 1080

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
    amplitudePixelDistance = 1080  # TODO: implement. currently just outputs width of the view.
    amplitude = amplitudePixelDistance * pixelSize

    print("Pendulum amplitude (peak-to-trough) = %fcm" % amplitude)


main()
