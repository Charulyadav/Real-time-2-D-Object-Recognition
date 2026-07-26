/*
  Charul - 002535330
  Spring 2026
  CS 5330 Computer Vision
  Project 3: Real-time Object Recognition

  connected components analysis and region visualization
*/

#include "regions.h"
#include <algorithm>
#include <cmath>

int findRegions(const cv::Mat &binary, cv::Mat &labels, std::vector<RegionInfo> &regions, int minArea, int topN, bool ignoreBorder)
{
    if (binary.empty())
        return -1;

    regions.clear();

    // OpenCV's connectedComponentsWithStats does the heavy lifting.
    // it returns:
    //   labels   - CV_32S map, each pixel labeled 0 (background) to N
    //   stats    - per-label stats (area, bounding box, etc.)
    //   centroids - per-label (x, y) center of mass
    cv::Mat stats, centroids;
    int numLabels = cv::connectedComponentsWithStats(
        binary, labels, stats, centroids, 8, CV_32S);

    // label 0 is always the background, so we start from label 1
    for (int i = 1; i < numLabels; i++)
    {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);

        // skip regions that are too small
        if (area < minArea)
            continue;

        int x = stats.at<int>(i, cv::CC_STAT_LEFT);
        int y = stats.at<int>(i, cv::CC_STAT_TOP);
        int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);

        // optionally skip regions touching the image border.
        // these are usually objects that are partially out of frame, or the background bleeding in - bad candidates for recognition.
        if (ignoreBorder)
        {
            if (x <= 0 || y <= 0 ||
                (x + w) >= binary.cols || (y + h) >= binary.rows)
            {
                continue;
            }
        }

        RegionInfo info;
        info.label = i;
        info.area = area;
        info.centroid = cv::Point2d(centroids.at<double>(i, 0),
                                    centroids.at<double>(i, 1));
        info.bbox = cv::Rect(x, y, w, h);
        regions.push_back(info);
    }

    // sort regions largest-first so "top N" means the N biggest
    std::sort(regions.begin(), regions.end(),
              [](const RegionInfo &a, const RegionInfo &b)
              {
                  return a.area > b.area;
              });

    // keep only the top N
    if ((int)regions.size() > topN)
    {
        regions.resize(topN);
    }

    return 0;
}

int colorizeRegions(const cv::Mat &labels, const std::vector<RegionInfo> &regions, cv::Mat &dst)
{
    dst = cv::Mat::zeros(labels.size(), CV_8UC3);

    // assign a distinct color to each kept region. a small fixed palette keeps colors stable and readable rather than fully random.
    static const cv::Vec3b palette[] = {
        cv::Vec3b(0, 0, 255),   // red
        cv::Vec3b(0, 255, 0),   // green
        cv::Vec3b(255, 0, 0),   // blue
        cv::Vec3b(0, 255, 255), // yellow
        cv::Vec3b(255, 0, 255), // magenta
        cv::Vec3b(255, 255, 0), // cyan
    };
    const int paletteSize = 6;

    // build a quick lookup: label ID -> palette color
    //  (only the kept regions get colored; everything else stays black)
    for (size_t k = 0; k < regions.size(); k++)
    {
        int targetLabel = regions[k].label;
        cv::Vec3b color = palette[k % paletteSize];

        for (int r = 0; r < labels.rows; r++)
        {
            const int *labelRow = labels.ptr<int>(r);
            cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(r);
            for (int c = 0; c < labels.cols; c++)
            {
                if (labelRow[c] == targetLabel)
                {
                    dstRow[c] = color;
                }
            }
        }
    }

    return 0;
}

int computeFeatures(const cv::Mat &binary, const cv::Mat &labels, RegionInfo &region, RegionFeatures &feats)
{
    if (binary.empty() || labels.empty())
        return -1;

    // build a single-region mask (255 where this region is, else 0).
    //  cv::moments works on this mask so it only "sees" this one region.
    cv::Mat mask = cv::Mat::zeros(labels.size(), CV_8UC1);
    for (int r = 0; r < labels.rows; r++)
    {
        const int *labelRow = labels.ptr<int>(r);
        uchar *maskRow = mask.ptr<uchar>(r);
        for (int c = 0; c < labels.cols; c++)
        {
            if (labelRow[c] == region.label)
                maskRow[c] = 255;
        }
    }

    // compute moments of the region.
    // cv::moments returns spatial moments (m00, m10, ...), central moments (mu20, mu02, mu11, ... - measured relative to the centroid, so they're
    // translation invariant) and normalized central moments (nu.. - also scale invariant).
    cv::Moments m = cv::moments(mask, true);

    // axis of least central moment.
    // theta = 0.5 * atan2(2*mu11, mu20 - mu02)
    double theta = 0.5 * std::atan2(2.0 * m.mu11, m.mu20 - m.mu02);
    region.theta = theta;

    // oriented bounding box.
    // rotate every region pixel by -theta around the centroid, track the min/max extents in the rotated frame, then rotate the box corners back.
    double cosT = std::cos(theta);
    double sinT = std::sin(theta);
    double cx = region.centroid.x;
    double cy = region.centroid.y;

    double minU = 1e9, maxU = -1e9, minV = 1e9, maxV = -1e9;
    for (int r = 0; r < labels.rows; r++)
    {
        const int *labelRow = labels.ptr<int>(r);
        for (int c = 0; c < labels.cols; c++)
        {
            if (labelRow[c] != region.label)
                continue;
            // translate to centroid, then rotate by -theta
            double dx = c - cx;
            double dy = r - cy;
            double u = dx * cosT + dy * sinT;  // along the primary axis
            double v = -dx * sinT + dy * cosT; // perpendicular
            if (u < minU)
                minU = u;
            if (u > maxU)
                maxU = u;
            if (v < minV)
                minV = v;
            if (v > maxV)
                maxV = v;
        }
    }
    region.minE1 = minU;
    region.maxE1 = maxU;
    region.minE2 = minV;
    region.maxE2 = maxV;

    // the four corners in rotated (u,v) space, rotated back to image space
    region.orientedBox.clear();
    double corners[4][2] = {
        {minU, minV}, {maxU, minV}, {maxU, maxV}, {minU, maxV}};
    for (int i = 0; i < 4; i++)
    {
        double u = corners[i][0];
        double v = corners[i][1];
        // rotate (u,v) back by +theta and translate to image coords
        double x = cx + u * cosT - v * sinT;
        double y = cy + u * sinT + v * cosT;
        region.orientedBox.push_back(cv::Point2f((float)x, (float)y));
    }

    // features
    double boxW = maxU - minU; // length along primary axis
    double boxH = maxV - minV; // length along perpendicular axis
    double boxArea = boxW * boxH;

    // percent filled: how much of the oriented box the region actually occupies
    feats.percentFilled = (boxArea > 0) ? (m.m00 / boxArea) : 0.0;

    // aspect ratio: shorter side / longer side, so it's always in (0, 1] and doesn't depend on which axis we call "width"
    double longer = std::max(boxW, boxH);
    double shorter = std::min(boxW, boxH);
    feats.aspectRatio = (longer > 0) ? (shorter / longer) : 0.0;

    // first Hu moment: a rotation/scale/translation invariant shape descriptor
    double hu[7];
    cv::HuMoments(m, hu);
    feats.huMoment1 = hu[0];

    return 0;
}

int drawRegionFeatures(cv::Mat &img, const RegionInfo &region)
{
    if (region.orientedBox.size() != 4)
        return 0;

    // draw the oriented bounding box (green lines between the 4 corners)
    for (int i = 0; i < 4; i++)
    {
        cv::line(img, region.orientedBox[i], region.orientedBox[(i + 1) % 4],
                 cv::Scalar(0, 255, 0), 2);
    }

    // draw the axis of least central moment (red line through the centroid)
    double len = 75.0; // half-length of the drawn axis in pixels
    double cx = region.centroid.x;
    double cy = region.centroid.y;
    cv::Point2d p1(cx - len * std::cos(region.theta),
                   cy - len * std::sin(region.theta));
    cv::Point2d p2(cx + len * std::cos(region.theta),
                   cy + len * std::sin(region.theta));
    cv::line(img, p1, p2, cv::Scalar(0, 0, 255), 2);

    // mark the centroid with a small filled circle
    cv::circle(img, region.centroid, 4, cv::Scalar(255, 0, 0), -1);

    return 0;
}

int featuresToVector(const RegionFeatures &feats, std::vector<float> &vec)
{
    vec.clear();
    vec.push_back((float)feats.percentFilled);
    vec.push_back((float)feats.aspectRatio);
    vec.push_back((float)feats.huMoment1);
    return 0;
}