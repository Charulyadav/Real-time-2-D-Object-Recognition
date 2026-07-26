/*
  Charul - 002535330
  Spring 2026
  CS 5330 Computer Vision
  Project 3: Real-time 2-D Object Recognition

  Thresholding and preprocessing implementations.
*/

#include "threshold.h"
#include <random>
#include <vector>

int preprocess(const cv::Mat &src, cv::Mat &gray)
{
    if (src.empty())
        return -1;

    // blur to make regions more uniform
    // using OpenCV's GaussianBlur here
    cv::Mat blurred;
    cv::GaussianBlur(src, blurred, cv::Size(5, 5), 0);

    // darken strongly saturated pixels
    // convert to HSV, then for each pixel multiply its value (brightness) by (1 - S/255 * 0.7)
    // highly saturated colors (red shirts, green markers, etc.) get pushed darker so they cluster with the object
    // side of the threshold instead of landing near the white background
    cv::Mat hsv;
    cv::cvtColor(blurred, hsv, cv::COLOR_BGR2HSV);

    for (int r = 0; r < hsv.rows; r++)
    {
        cv::Vec3b *row = hsv.ptr<cv::Vec3b>(r);
        for (int c = 0; c < hsv.cols; c++)
        {
            uchar s = row[c][1];                      // saturation
            float scale = 1.0f - (s / 255.0f) * 0.7f; // darken proportionally
            row[c][2] = cv::saturate_cast<uchar>(row[c][2] * scale);
        }
    }

    // convert back to BGR, then grayscale
    cv::Mat bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    return 0;
}

int isodataThreshold(const cv::Mat &gray, float sampleFrac, int maxIter)
{
    if (gray.empty())
        return -1;

    // randomly sample pixel values from the image
    int total = gray.rows * gray.cols;
    int sampleCount = (int)(total * sampleFrac);
    if (sampleCount < 100)
        sampleCount = 100; // keep a minimum sample size

    std::mt19937 rng(42); // fixed seed for stability across frames
    std::uniform_int_distribution<int> distR(0, gray.rows - 1);
    std::uniform_int_distribution<int> distC(0, gray.cols - 1);

    std::vector<uchar> samples;
    samples.reserve(sampleCount);
    for (int i = 0; i < sampleCount; i++)
    {
        int r = distR(rng);
        int c = distC(rng);
        samples.push_back(gray.at<uchar>(r, c));
    }

    // initialize two means
    // a safe initialization: min sample and max sample. The two clusters will pull apart toward the object and background means
    float meanA = 255.0f, meanB = 0.0f;
    for (uchar v : samples)
    {
        if (v < meanA)
            meanA = v;
        if (v > meanB)
            meanB = v;
    }

    // k-means iteration with k=2
    // at each step:
    //   - assign every sample to the closer of the two means
    //   - recompute each mean as the average of its assignments
    // stop when the means stop changing (or maxIter is hit)
    for (int iter = 0; iter < maxIter; iter++)
    {
        float sumA = 0, sumB = 0;
        int countA = 0, countB = 0;

        for (uchar v : samples)
        {
            if (std::abs((float)v - meanA) < std::abs((float)v - meanB))
            {
                sumA += v;
                countA++;
            }
            else
            {
                sumB += v;
                countB++;
            }
        }

        float newMeanA = (countA > 0) ? sumA / countA : meanA;
        float newMeanB = (countB > 0) ? sumB / countB : meanB;

        // check for convergence
        if (std::abs(newMeanA - meanA) < 0.5f &&
            std::abs(newMeanB - meanB) < 0.5f)
        {
            meanA = newMeanA;
            meanB = newMeanB;
            break;
        }
        meanA = newMeanA;
        meanB = newMeanB;
    }

    // threshold is the midpoint between the two means
    int thresh = (int)((meanA + meanB) / 2.0f);
    if (thresh < 0)
        thresh = 0;
    if (thresh > 255)
        thresh = 255;
    return thresh;
}

int applyThreshold(const cv::Mat &gray, int thresh, cv::Mat &dst)
{
    if (gray.empty())
        return -1;

    // convention: object (dark, < thresh) becomes 255 (white)
    // background (light, >= thresh) becomes 0 (black)
    // this is what morphological ops and connectedComponents expect
    dst = cv::Mat::zeros(gray.size(), CV_8UC1);
    for (int r = 0; r < gray.rows; r++)
    {
        const uchar *srcRow = gray.ptr<uchar>(r);
        uchar *dstRow = dst.ptr<uchar>(r);
        for (int c = 0; c < gray.cols; c++)
        {
            dstRow[c] = (srcRow[c] < thresh) ? 255 : 0;
        }
    }
    return 0;
}