/*
  Charul - 002535330
  Spring 2026
  CS 5330 Computer Vision
  Project 3: Real-time Object Recognition

  single-window dashboard compositor
*/

#include "dashboard.h"

// helper: ensure an image is 3-channel BGR (threshold is 1-channel)
static cv::Mat toBGR(const cv::Mat &src)
{
    if (src.channels() == 1)
    {
        cv::Mat out;
        cv::cvtColor(src, out, cv::COLOR_GRAY2BGR);
        return out;
    }
    return src;
}

// helper: draw a small caption label in the top-left of a tile
static void caption(cv::Mat &tile, const std::string &text)
{
    cv::rectangle(tile, cv::Rect(0, 0, 150, 24), cv::Scalar(0, 0, 0), -1);
    cv::putText(tile, text, cv::Point(6, 17),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
}

int buildDashboard(const cv::Mat &original, const cv::Mat &thresh, const cv::Mat &regions, const cv::Mat &features,
                   const std::vector<std::string> &statusLines, cv::Mat &dst)
{
    const int tileW = 480, tileH = 360; // each tile size
    const int statusH = 90;             // bottom status strip height

    // resize each view to the tile size and ensure 3 channels
    cv::Mat t0, t1, t2, t3;
    cv::resize(toBGR(original), t0, cv::Size(tileW, tileH));
    cv::resize(toBGR(thresh), t1, cv::Size(tileW, tileH));
    cv::resize(toBGR(regions), t2, cv::Size(tileW, tileH));
    cv::resize(toBGR(features), t3, cv::Size(tileW, tileH));

    caption(t0, "Original");
    caption(t1, "Threshold");
    caption(t2, "Regions");
    caption(t3, "Features");

    // composite canvas: 2 wide x 2 tall, plus a status strip below
    int W = tileW * 2;
    int H = tileH * 2 + statusH;
    dst = cv::Mat(H, W, CV_8UC3, cv::Scalar(30, 30, 35));

    t0.copyTo(dst(cv::Rect(0, 0, tileW, tileH)));
    t1.copyTo(dst(cv::Rect(tileW, 0, tileW, tileH)));
    t2.copyTo(dst(cv::Rect(0, tileH, tileW, tileH)));
    t3.copyTo(dst(cv::Rect(tileW, tileH, tileW, tileH)));

    // status strip text
    int y = tileH * 2 + 25;
    for (const auto &line : statusLines)
    {
        cv::putText(dst, line, cv::Point(12, y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(220, 220, 220), 1);
        y += 24;
    }

    return 0;
}