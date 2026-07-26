/*
  Charul - 002535330
  Spring 2026
  CS 5330 Computer Vision
  Project 3: Real-time 2-D Object Recognition

  captures live webcam video, applies thresholding, and displays results in two windows side by side

  controls:
    q - quit
    s - save current frames to capture_<N>_*.jpg
    t - toggle threshold view
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include "threshold.h"
#include "morphology.h"
#include "regions.h"
#include "csv_util.h"
#include <cstring>
#include "classifier.h"
#include "embedding.h"
#include <utility> // for std::pair
#include <fstream>
#include "dashboard.h"
#include <cstdio>

int main(int argc, char *argv[])
{
    bool showCleaned = true;
    bool showRegions = true;
    std::vector<RegionFeatures> allFeats;

    cv::VideoCapture capdev(1);
    if (!capdev.isOpened())
    {
        std::cout << "Unable to open video device\n";
        return -1;
    }

    // one window
    cv::namedWindow("Object Recognition", cv::WINDOW_AUTOSIZE);

    // trackbars on the single window
    int thresholdOffset = 50; // center=no offset
    int minAreaSlider = 500;
    int topNSlider = 3;
    cv::createTrackbar("Thresh", "Object Recognition", &thresholdOffset, 100, nullptr);
    cv::createTrackbar("MinArea", "Object Recognition", &minAreaSlider, 5000, nullptr);
    cv::createTrackbar("TopN", "Object Recognition", &topNSlider, 5, nullptr);

    // load DBs
    ObjectDB db;
    bool dbLoaded = (loadDatabase("object_db.csv", db) == 0);
    ObjectDB embDb;
    bool embDbLoaded = false;
    cv::dnn::Net net = cv::dnn::readNet("resnet18-v2-7.onnx");

    bool classifyMode = false;     // hand-feature classify (cheap, every frame OK)
    std::string lastCnnLabel = ""; // cached CNN result (only updates on demand)

    cv::Mat frame, gray, binary, cleaned, labels, regionDisplay;
    std::vector<RegionInfo> regions;
    int saveCounter = 0;

    std::cout << "Controls:\n"
              << "  q quit | s save | n train(hand) | p classify(hand) toggle\n"
              << "  N train(CNN) | P classify(CNN) once | l reload | x clear\n";

    for (;;)
    {
        capdev >> frame;
        if (frame.empty())
            break;

        // PROCESSING
        preprocess(frame, gray);
        int thresh = isodataThreshold(gray) + (thresholdOffset - 50);
        applyThreshold(gray, thresh, binary);
        cleanup(binary, cleaned, 1, 2);

        int minArea = (minAreaSlider < 50) ? 50 : minAreaSlider;
        int topN = (topNSlider < 1) ? 1 : topNSlider;
        findRegions(cleaned, labels, regions, minArea, topN, true);
        colorizeRegions(labels, regions, regionDisplay);

        // features + hand-feature classification drawn on a copy
        cv::Mat featureDisplay = frame.clone();
        for (size_t i = 0; i < regions.size(); i++)
        {
            RegionFeatures f;
            computeFeatures(cleaned, labels, regions[i], f);
            drawRegionFeatures(featureDisplay, regions[i]);

            if (classifyMode && dbLoaded)
            {
                std::vector<float> v;
                featuresToVector(f, v);
                float d;
                std::string label = classify(db, v, d);
                cv::putText(featureDisplay, label,
                            cv::Point(regions[i].bbox.x, regions[i].bbox.y - 8),
                            cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 0, 255), 3);
            }
            // show cached CNN label if we have one
            if (!lastCnnLabel.empty())
            {
                cv::putText(featureDisplay, lastCnnLabel + " (CNN)",
                            cv::Point(regions[i].bbox.x, regions[i].bbox.y - 40),
                            cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 0, 255), 3);
            }
        }

        // DISPLAY
        std::vector<std::string> status = {
            "Mode: " + std::string(classifyMode ? "CLASSIFY(hand)" : "idle") + "   Regions: " + std::to_string(regions.size()) + "   DB: " + std::to_string(db.labels.size()),
            "n=train  p=classify  N=trainCNN  P=classifyCNN  s=save  q=quit"};
        cv::Mat dashboard;
        buildDashboard(frame, cleaned, regionDisplay, featureDisplay, status, dashboard);
        cv::imshow("Object Recognition", dashboard);

        // INPUT
        int key = cv::waitKey(10);
        if (key == 'q')
            break;
        else if (key == 'p')
        {
            classifyMode = !classifyMode;
            lastCnnLabel = "";
            if (classifyMode)
            {
                dbLoaded = (loadDatabase("object_db.csv", db) == 0); // reload fresh
            }
            std::cout << "Classify mode: " << (classifyMode ? "ON" : "OFF")
                      << " | DB loaded: " << (dbLoaded ? "yes" : "no")
                      << " | DB size: " << db.labels.size() << "\n";
        }
        else if (key == 'f')
        {
            if (regions.empty())
            {
                std::cout << "No region detected.\n";
            }
            else
            {
                std::cout << "Feature vectors:\n";
                for (size_t i = 0; i < regions.size(); i++)
                {
                    RegionFeatures f;
                    computeFeatures(cleaned, labels, regions[i], f);
                    std::cout << "  Region " << i
                              << ": percentFilled=" << f.percentFilled
                              << "  aspectRatio=" << f.aspectRatio
                              << "  huMoment1=" << f.huMoment1 << "\n";
                }
            }
        }
        else if (key == 'n')
        {
            if (!regions.empty())
            {
                RegionFeatures f;
                computeFeatures(cleaned, labels, regions[0], f);
                std::vector<float> v;
                featuresToVector(f, v);
                std::cout << "Label: ";
                std::string lbl;
                std::cin >> lbl;
                char dbf[] = "object_db.csv";
                char *lp = const_cast<char *>(lbl.c_str());
                append_image_data_csv(dbf, lp, v, 0);
                dbLoaded = (loadDatabase("object_db.csv", db) == 0);
            }
        }
        else if (key == 'e')
        {
            if (regions.empty())
            {
                std::cout << "No object detected.\n";
            }
            else if (!dbLoaded)
            {
                std::cout << "No database loaded.\n";
            }
            else
            {
                RegionFeatures f;
                computeFeatures(cleaned, labels, regions[0], f);
                std::vector<float> v;
                featuresToVector(f, v);
                float dist;
                std::string predicted = classify(db, v, dist);

                std::cout << "Predicted: " << predicted << ". Enter TRUE label: ";
                std::string truth;
                std::cin >> truth;

                FILE *fp = fopen("eval_results.csv", "a");
                if (fp)
                {
                    fprintf(fp, "%s,%s\n", truth.c_str(), predicted.c_str());
                    fclose(fp);
                }
                std::cout << (truth == predicted ? "CORRECT" : "WRONG")
                          << " (true=" << truth << ", pred=" << predicted << ")\n";
            }
        }
        else if (key == 'N')
        { // train CNN - on demand only
            if (!regions.empty() && !net.empty())
            {
                RegionInfo &r = regions[0];
                RegionFeatures f;
                computeFeatures(cleaned, labels, r, f);
                cv::Mat roi, emb;
                prepEmbeddingImage(frame, roi, (int)r.centroid.x, (int)r.centroid.y,
                                   (float)r.theta, (float)r.minE1, (float)r.maxE1,
                                   (float)r.minE2, (float)r.maxE2, 0);
                getEmbedding(roi, emb, net, 0);
                std::vector<float> v;
                for (int j = 0; j < emb.cols; j++)
                    v.push_back(emb.at<float>(0, j));
                std::cout << "CNN Label: ";
                std::string lbl;
                std::cin >> lbl;
                char dbf[] = "embedding_db.csv";
                char *lp = const_cast<char *>(lbl.c_str());
                append_image_data_csv(dbf, lp, v, 0);
            }
        }
        else if (key == 'P')
        { // classify CNN - runs ONCE, caches result
            embDbLoaded = (loadDatabase("embedding_db.csv", embDb) == 0);
            if (!regions.empty() && !net.empty() && embDbLoaded)
            {
                RegionInfo &r = regions[0];
                RegionFeatures f;
                computeFeatures(cleaned, labels, r, f);
                cv::Mat roi, emb;
                prepEmbeddingImage(frame, roi, (int)r.centroid.x, (int)r.centroid.y,
                                   (float)r.theta, (float)r.minE1, (float)r.maxE1,
                                   (float)r.minE2, (float)r.maxE2, 0);
                getEmbedding(roi, emb, net, 0);
                std::vector<float> v;
                for (int j = 0; j < emb.cols; j++)
                    v.push_back(emb.at<float>(0, j));
                float d;
                lastCnnLabel = classify(embDb, v, d);
                std::cout << "CNN says: " << lastCnnLabel << " (dist " << d << ")\n";
            }
        }
        else if (key == 'l')
        {
            dbLoaded = (loadDatabase("object_db.csv", db) == 0);
        }
        else if (key == 's')
        {
            cv::imwrite("capture_" + std::to_string(saveCounter) + ".jpg", dashboard);
            std::cout << "Saved dashboard " << saveCounter++ << "\n";
        }
    }

    cv::destroyAllWindows();
    return 0;
}