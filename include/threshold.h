/*
  Charul - 002535330
  Spring 2026
  CS 5330 Computer Vision
  Project 3: Real-time 2-D Object Recognition

  function prototypes for thresholding and preprocessing.
*/

#ifndef THRESHOLD_H
#define THRESHOLD_H

#include <opencv2/opencv.hpp>

/*
  preprocesses a BGR frame for thresholding
  steps:
    1. blur the image slightly to make regions more uniform
    2. darken strongly saturated pixels so colored objects move further from the white background (which is unsaturated)
    3. convert to grayscale

  arguments:
    src  - input BGR image (CV_8UC3)
    gray - output grayscale image (CV_8UC1), ready for thresholding
  returns 0 on success, -1 if src is empty
*/
int preprocess(const cv::Mat &src, cv::Mat &gray);

/*
  computes a dynamic threshold using the ISODATA / K-means algorithm with K=2.
  samples a fraction of pixels at random, then iteratively splits them into two clusters by intensity and
  returns the midpoint of the two cluster means as the threshold value.

  arguments:
    gray         - input grayscale image (CV_8UC1)
    sampleFrac   - fraction of pixels to sample (default 1/16 = 0.0625)
    maxIter      - max k-means iterations (default 20)
  returns the threshold value in [0, 255], or -1 on error
*/
int isodataThreshold(const cv::Mat &gray, float sampleFrac = 0.0625f, int maxIter = 20);

/*
  applies a binary threshold to a grayscale image.
  convention: object (dark in input) becomes white (255) in output;
  background (light in input) becomes black (0). this matches the standard
  for downstream morphological and connected-components processing.

  arguments:
    gray   - input grayscale image (CV_8UC1)
    thresh - threshold value in [0, 255]
    dst    - output binary image (CV_8UC1)
  returns 0 on success, -1 if gray is empty
*/
int applyThreshold(const cv::Mat &gray, int thresh, cv::Mat &dst);

#endif