/*
  Charul - 002535330
  Spring 2026
  CS 5330 Computer Vision
  Project 3: Real-time Object Recognition

  CNN embedding utilities (adapted from Prof. Bruce Maxwell's utilities.cpp).
  uses OpenCV's DNN module to run a pre-trained ResNet18
*/

#ifndef EMBEDDING_H
#define EMBEDDING_H

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

/*
  computes a ResNet18 embedding for a (rotated, cropped) object ROI.
  ]adapted from Prof. Maxwell's getEmbedding

  arguments:
    src       - the ROI image (8UC3)
    embedding - output embedding vector (1 x 512 CV_32F)
    net       - loaded ResNet18 DNN
    debug     - 1 to show/print debug info, 0 otherwise
  returns 0 on success.
*/
int getEmbedding(cv::Mat &src, cv::Mat &embedding, cv::dnn::Net &net, int debug);

/*
  rotates and crops the object ROI from the frame so its primary axis points right.
  adapted from Prof. Maxwell's prepEmbeddingImage

  arguments described in the implementation.
*/
void prepEmbeddingImage(cv::Mat &frame, cv::Mat &embimage, int cx, int cy, float theta, float minE1, float maxE1,
                        float minE2, float maxE2, int debug);

#endif