/*
  Charul - 002535330
  Spring 2026
  CS 5330 Computer Vision
  Project 3: Real-time Object Recognition

  nearest-neighbor classifier using scaled Euclidean distance
*/

#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <vector>
#include <string>

/*
  holds the loaded object database and the per-feature standard deviations needed for scaled Euclidean distance
*/
struct ObjectDB
{
    std::vector<std::string> labels;          // label for each entry
    std::vector<std::vector<float>> features; // feature vector for each entry
    std::vector<float> stdevs;                // per-feature standard deviation
};

/*
  loads the object database from a CSV file and computes the standard deviation of each feature across all entries (used to scale distances)

  arguments:
    filename - path to the object database CSV
    db       - output loaded database
  returns 0 on success, -1 if the file can't be read or is empty
*/
int loadDatabase(const char *filename, ObjectDB &db);

/*
  classifies an unknown feature vector against the database using nearest-neighbor with scaled Euclidean distance:
    d = sqrt( sum_i ( (a_i - b_i) / stdev_i )^2 )

  arguments:
    db          - the loaded object database
    unknown     - the unknown object's feature vector
    bestDist    - output: distance to the nearest neighbor
  returns the label of the nearest neighbor or "unknown" if the DB is empty
*/
std::string classify(const ObjectDB &db, const std::vector<float> &unknown, float &bestDist);

#endif