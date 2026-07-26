/*
  Charul - 002535330
  Spring 2026
  CS 5330 Computer Vision
  Project 3: Real-time Object Recognition

  composites all views into a single dashboard image for one-window display
*/

#ifndef DASHBOARD_H
#define DASHBOARD_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

/*
  assembles a single dashboard image from the four pipeline views plus a status strip.
  all inputs are resized to a common tile size and tiled 2x2, with a status bar across the bottom.

  arguments:
    original, thresh, regions, features - the four views (any size)
    statusLines - lines of status text to print along the bottom
    dst         - output composite image (CV_8UC3)
  returns 0 on success.
*/
int buildDashboard(const cv::Mat &original, const cv::Mat &thresh, const cv::Mat &regions, const cv::Mat &features,
                   const std::vector<std::string> &statusLines, cv::Mat &dst);

#endif