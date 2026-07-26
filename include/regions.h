/*
  Charul - 002535330
  Spring 2026
  CS 5330 Computer Vision
  Project 3: Real-time Object Recognition

  connected components analysis and region visualization
*/

#ifndef REGIONS_H
#define REGIONS_H

#include <opencv2/opencv.hpp>
#include <vector>

/*
  holds the key statistics for one detected region
*/
struct RegionInfo
{
  int label;                            // the region's ID in the label map
  int area;                             // number of pixels in the region
  cv::Point2d centroid;                 // (x, y) center of mass
  cv::Rect bbox;                        // axis-aligned bounding box
  double theta;                         // axis of least central moment (radians)
  std::vector<cv::Point2f> orientedBox; // 4 corners of the oriented bbox
  // projection extents along the primary (E1) and secondary (E2) axes,
  // needed for the embedding ROI extraction in Task 9
  double minE1, maxE1, minE2, maxE2;
};

/*
  holds the computed shape features for one region.
  all features are designed to be translation, scale and rotation invariant.
*/
struct RegionFeatures
{
  double percentFilled; // area/oriented bounding box area
  double aspectRatio;   // oriented bbox: shorter side / longer side
  double huMoment1;     // first Hu moment (rotation-invariant shape descriptor)
};

/*
  runs connected components analysis on a cleaned binary image and returns
  the top-N largest regions, ignoring regions smaller than minArea and regions that touch the image border

  arguments:
    binary        - input cleaned binary image (CV_8UC1, object = 255)
    labels        - output label map (CV_32S), one integer ID per pixel
    regions       - output list of kept regions, sorted largest-first
    minArea       - ignore regions with fewer pixels than this (default 500)
    topN          - keep at most this many regions (default 3)
    ignoreBorder  - if true, discard regions touching the image edge (default true)
  returns 0 on success, -1 if binary is empty
*/
int findRegions(const cv::Mat &binary, cv::Mat &labels, std::vector<RegionInfo> &regions,
                int minArea = 500, int topN = 3, bool ignoreBorder = true);

/*
  builds a color visualization of the kept regions. each region in the 'regions' list is drawn in a distinct color; everything else is black

  arguments:
    labels  - label map from findRegions (CV_32S)
    regions - the kept regions from findRegions
    dst     - output color image (CV_8UC3)
  returns 0 on success
*/
int colorizeRegions(const cv::Mat &labels, const std::vector<RegionInfo> &regions, cv::Mat &dst);

/*
  computes shape features for a single region given the binary region mask.
  calculates the axis of least central moment, the oriented bounding box (stored back into the RegionInfo) and a set of invariant features.

  arugments:
    binary  - the cleaned binary image (CV_8UC1)
    labels  - label map from findRegions (CV_32S)
    region  - the region to analyze; its theta and orientedBox are filled in
    feats   - output computed features
  returns 0 on success, -1 on error.
*/
int computeFeatures(const cv::Mat &binary, const cv::Mat &labels, RegionInfo &region, RegionFeatures &feats);

/*
  draws the oriented bounding box and the axis of least central moment for a region onto a color image, for visualization.

  arguments:
    img    - color image to draw on (CV_8UC3), modified in place
    region - region with theta and orientedBox already computed
  returns 0 on success.
*/
int drawRegionFeatures(cv::Mat &img, const RegionInfo &region);

/*
  packs a RegionFeatures struct into a flat float vector for storage.
  order: [percentFilled, aspectRatio, huMoment1]

  arguments:
    feats - the computed features
    vec   - output flat vector
  returns 0 on success.
*/
int featuresToVector(const RegionFeatures &feats, std::vector<float> &vec);

#endif