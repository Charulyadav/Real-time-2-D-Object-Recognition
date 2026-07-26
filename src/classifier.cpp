/*
  Charul - 002535330
  Spring 2026
  CS 5330 Computer Vision
  Project 3: Real-time Object Recognition

  nearest-neighbor classifier implementation
*/

#include "classifier.h"
#include "csv_util.h"
#include <cmath>
#include <cstring>
#include <limits>

int loadDatabase(const char *filename, ObjectDB &db)
{
    db.labels.clear();
    db.features.clear();
    db.stdevs.clear();

    std::vector<char *> names;
    std::vector<std::vector<float>> data;
    char *fname = const_cast<char *>(filename);
    if (read_image_data_csv(fname, names, data, 0) != 0)
    {
        return -1;
    }
    if (data.empty())
        return -1;

    // copy into our struct
    for (size_t i = 0; i < names.size(); i++)
    {
        db.labels.push_back(std::string(names[i]));
        db.features.push_back(data[i]);
        delete[] names[i]; // csv_util allocated these; free them
    }

    // compute per-feature standard deviation across all entries
    int numFeatures = (int)db.features[0].size();
    db.stdevs.assign(numFeatures, 0.0f);

    // mean of each feature
    std::vector<double> mean(numFeatures, 0.0);
    for (const auto &f : db.features)
    {
        for (int j = 0; j < numFeatures; j++)
            mean[j] += f[j];
    }
    for (int j = 0; j < numFeatures; j++)
        mean[j] /= db.features.size();

    // variance then standard deviation
    std::vector<double> var(numFeatures, 0.0);
    for (const auto &f : db.features)
    {
        for (int j = 0; j < numFeatures; j++)
        {
            double diff = f[j] - mean[j];
            var[j] += diff * diff;
        }
    }
    for (int j = 0; j < numFeatures; j++)
    {
        var[j] /= db.features.size();
        db.stdevs[j] = (float)std::sqrt(var[j]);
        // guard against zero stdev (a feature that never varies) to avoid division by zero
        // set to 1 so it contributes its raw difference
        if (db.stdevs[j] < 1e-6f)
            db.stdevs[j] = 1.0f;
    }

    return 0;
}

std::string classify(const ObjectDB &db, const std::vector<float> &unknown,
                     float &bestDist)
{
    bestDist = std::numeric_limits<float>::max();
    if (db.features.empty())
        return "unknown";

    std::string bestLabel = "unknown";

    // compare against every database entry, keep the closest
    for (size_t i = 0; i < db.features.size(); i++)
    {
        const std::vector<float> &dbFeat = db.features[i];
        if (dbFeat.size() != unknown.size())
            continue;

        // scaled euclidean distance
        double sum = 0.0;
        for (size_t j = 0; j < unknown.size(); j++)
        {
            double diff = (unknown[j] - dbFeat[j]) / db.stdevs[j];
            sum += diff * diff;
        }
        float dist = (float)std::sqrt(sum);

        if (dist < bestDist)
        {
            bestDist = dist;
            bestLabel = db.labels[i];
        }
    }

    return bestLabel;
}