/*
  Charul - 002535330
  Spring 2026
  CS 5330 Computer Vision
  Project 3: Real-time Object Recognition

  CNN embedding utilities.
  adapted from Prof. Bruce Maxwell's utilities.cpp (CS 5330).
  the getEmbedding and prepEmbeddingImage functions are his code; only the include directives were changed
  to match this project's structure
*/
#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdio>
#include <cstring>
#include "opencv2/opencv.hpp"
#include "opencv2/dnn.hpp"
#include "embedding.h"

int getEmbedding(cv::Mat &src, cv::Mat &embedding, cv::dnn::Net &net, int debug)
{
    const int ORNet_size = 224;
    cv::Mat blob;
    cv::Mat resized;

    cv::resize(src, resized, cv::Size(ORNet_size, ORNet_size));

    cv::dnn::blobFromImage(resized,
                           blob,
                           (1.0 / 255.0) * (1 / 0.226),
                           cv::Size(ORNet_size, ORNet_size),
                           cv::Scalar(124, 116, 104),
                           true, // swapRB
                           false,
                           CV_32F);

    net.setInput(blob);
    embedding = net.forward("onnx_node!resnetv22_flatten0_reshape0");

    if (debug)
    {
        std::cout << embedding << std::endl;
    }
    return 0;
}

void prepEmbeddingImage(cv::Mat &frame, cv::Mat &embimage, int cx, int cy,
                        float theta, float minE1, float maxE1,
                        float minE2, float maxE2, int debug)
{
    cv::Mat rotatedImage;
    cv::Mat M;

    M = cv::getRotationMatrix2D(cv::Point2f(cx, cy), -theta * 180 / M_PI, 1.0);
    int largest = frame.cols > frame.rows ? frame.cols : frame.rows;
    largest = (int)(1.414 * largest);
    cv::warpAffine(frame, rotatedImage, M, cv::Size(largest, largest));

    if (debug)
        cv::imshow("rotated", rotatedImage);

    int left = cx + (int)minE1;
    int top = cy - (int)maxE2;
    int width = (int)maxE1 - (int)minE1;
    int height = (int)maxE2 - (int)minE2;

    if (left < 0)
    {
        width += left;
        left = 0;
    }
    if (top < 0)
    {
        height += top;
        top = 0;
    }
    if (left + width >= rotatedImage.cols)
        width = (rotatedImage.cols - 1) - left;
    if (top + height >= rotatedImage.rows)
        height = (rotatedImage.rows - 1) - top;

    if (debug)
        printf("ROI box: %d %d %d %d\n", left, top, width, height);

    cv::Rect objroi(left, top, width, height);
    cv::Mat extractedImage(rotatedImage, objroi);

    if (debug)
        cv::imshow("extracted", extractedImage);

    extractedImage.copyTo(embimage);
    return;
}