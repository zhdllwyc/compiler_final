#ifndef STATS_H
#define STATS_H

#include <chrono>
#include <vector>
#include <string>
#include <map>
#include <fstream>
#include <iostream>

using namespace std;
using namespace std::chrono;

// Contains a helper class to make gathering statistics easy and standardized.

class Stats {

private:
    void add_point() {
        points.push_back(high_resolution_clock::now());        
    }
    int stopped = 0;

public:
    vector<time_point<high_resolution_clock>> points;
    vector<string> names;
    map<string, float> scalar_stats;
    map<string, vector<float>*> vector_stats;

    void checkpoint(string name) {
        if (!points.size()) start();
        names.push_back(name);
        add_point();
    }

    void start() {
        if (!points.size()) add_point();
    }

    void stop() {
        //avoid double-stopping
        if (stopped) return;
        stopped = 1;
        checkpoint("total");
    }

    // add a scalar statistic (i.e # of iterations to convergence)
    void add_stat(string name, float value) {
        scalar_stats[name] = value;
    }

    // add a datapoint to a list of datapoints
    // This is used for keeping track of a value that changes each iteration of a graph algorithm (i.e worklist size)
    void add_datapoint(string name, float value) {
        if (vector_stats.find(name) == vector_stats.end()) {
            vector<float>* list = new vector<float>();
            vector_stats[name] = list;
        }
        vector_stats[name]->push_back(value);
    }

    void json_dump(string fname) {
        // If not already stopped, stop
        stop();
        for (int i = 1; i < points.size(); i++) {
            scalar_stats[names[i-1]] = duration_cast<microseconds>(points[i] - points[i-1]).count() / 1000000.;
        }
        scalar_stats["total"] = duration_cast<microseconds>(points[points.size() - 1] - points[0]).count() / 1000000.;
        
        ofstream f(fname);
        if (!f.is_open()) {
            cout << "Cannot open file " << fname << " for writing." << endl;
            return;
        }

        int first = 1;
        f << "{";
        for (auto it : scalar_stats) {
            if (!first) f << ",";            
            first = 0;

            f << "\"" << it.first << "\": " << it.second;
        }

        for (auto it : vector_stats) {
            if (!first) f << ",";            
            first = 0;

            f << "\"" << it.first << "\": [";
            for (int j = 0; j < it.second->size(); j++) {
                if (j) f << ",";
                f << it.second->at(j);
            }
            f << "]";
        }
        
        f << "}";
        f.close();
        cout << "Stats written to " << fname << "." << endl;
    }
};

#endif
