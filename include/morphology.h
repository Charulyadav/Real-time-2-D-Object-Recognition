/*
  Charul - 002535330
  Spring 2026
  CS 5330 Computer Vision
  Project 3: Real-time Object Recognition

  from-scratch morphological filtering: erosion, dilation, opening, closing.
  all functions operate on binary images (CV_8UC1, values 0 or 255).
*/

#ifndef MORPHOLOGY_H
#define MORPHOLOGY_H

#include <opencv2/opencv.hpp>

/*
  dilation with a 3x3 square structuring element.
  an output pixel is white (255) if ANY pixel in its 3x3 neighborhood
  is white in the input. grows white regions; fills small gaps/holes.

  arguments:
    src - input binary image (CV_8UC1, values 0 or 255)
    dst - output dilated binary image (CV_8UC1)
  returns 0 on success, -1 if src is empty.
*/
int dilate3x3(const cv::Mat &src, cv::Mat &dst);

/*
  erosion with a 3x3 square structuring element.
  an output pixel is white (255) only if ALL pixels in its 3x3
  neighborhood are white in the input. Shrinks white regions; removes
  small speckle.

  arguments:
    src - input binary image (CV_8UC1, values 0 or 255)
    dst - output eroded binary image (CV_8UC1)
  returns 0 on success, -1 if src is empty.
*/
int erode3x3(const cv::Mat &src, cv::Mat &dst);

/*
  cleans up a binary image: `openIter` rounds of opening (erode then
  dilate) to remove background speckle, followed by `closeIter` rounds
  of closing (dilate then erode) to fill internal holes.

  arguments:
    src       - input binary image (CV_8UC1)
    dst       - output cleaned binary image (CV_8UC1)
    openIter  - number of opening iterations (default 1)
    closeIter - number of closing iterations (default 2)
  returns 0 on success, -1 if src is empty.
*/
int cleanup(const cv::Mat &src, cv::Mat &dst, int openIter = 1, int closeIter = 2);

#endif