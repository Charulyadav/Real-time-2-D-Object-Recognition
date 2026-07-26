/*
  Charul - 002535330
  Spring 2026
  CS 5330 Computer Vision
  Project 3: Real-time Object Recognition

  from-scratch morphological filtering implementations
*/

#include "morphology.h"

int dilate3x3(const cv::Mat &src, cv::Mat &dst)
{
    if (src.empty())
        return -1;

    dst = cv::Mat::zeros(src.size(), CV_8UC1);

    for (int r = 0; r < src.rows; r++)
    {
        uchar *dstRow = dst.ptr<uchar>(r);
        for (int c = 0; c < src.cols; c++)
        {
            // look at the 3x3 neighborhood. If ANY neighbor is white, this output pixel is white.
            uchar maxVal = 0;
            for (int dr = -1; dr <= 1; dr++)
            {
                for (int dc = -1; dc <= 1; dc++)
                {
                    int nr = r + dr;
                    int nc = c + dc;
                    // skip out-of-bounds neighbors (treat as background)
                    if (nr < 0 || nr >= src.rows || nc < 0 || nc >= src.cols)
                        continue;
                    if (src.at<uchar>(nr, nc) > maxVal)
                    {
                        maxVal = src.at<uchar>(nr, nc);
                    }
                }
            }
            dstRow[c] = maxVal;
        }
    }
    return 0;
}

int erode3x3(const cv::Mat &src, cv::Mat &dst)
{
    if (src.empty())
        return -1;

    dst = cv::Mat::zeros(src.size(), CV_8UC1);

    for (int r = 0; r < src.rows; r++)
    {
        uchar *dstRow = dst.ptr<uchar>(r);
        for (int c = 0; c < src.cols; c++)
        {
            // look at the 3x3 neighborhood. output is white only if ALL neighbors are white. start assuming white, flip to black
            //  if any neighbor is black.
            uchar minVal = 255;
            for (int dr = -1; dr <= 1; dr++)
            {
                for (int dc = -1; dc <= 1; dc++)
                {
                    int nr = r + dr;
                    int nc = c + dc;
                    // out-of-bounds counts as background (black), so an object pixel at the image edge erodes away
                    if (nr < 0 || nr >= src.rows || nc < 0 || nc >= src.cols)
                    {
                        minVal = 0;
                        continue;
                    }
                    if (src.at<uchar>(nr, nc) < minVal)
                    {
                        minVal = src.at<uchar>(nr, nc);
                    }
                }
            }
            dstRow[c] = minVal;
        }
    }
    return 0;
}

int cleanup(const cv::Mat &src, cv::Mat &dst, int openIter, int closeIter)
{
    if (src.empty())
        return -1;

    cv::Mat temp = src.clone();
    cv::Mat buffer;

    // opening (erode then dilate), repeated openIter times.
    //  removes small background speckle without shrinking the object much.
    for (int i = 0; i < openIter; i++)
    {
        erode3x3(temp, buffer);
        dilate3x3(buffer, temp);
    }

    // closing (dilate then erode), repeated closeIter times.
    // fills internal holes and gaps (e.g. the scissor blade light reflection breakage).
    for (int i = 0; i < closeIter; i++)
    {
        dilate3x3(temp, buffer);
        erode3x3(buffer, temp);
    }

    dst = temp;
    return 0;
}