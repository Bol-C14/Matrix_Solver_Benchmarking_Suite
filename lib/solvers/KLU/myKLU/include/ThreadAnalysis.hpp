/**
 *
 *
 */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <iomanip>
#include <limits>
#include <algorithm>

namespace ThreadAnlysis
{

    struct ThreadData
    {
        int loopIndex;
        int threadIndex;
        double beginTime;
        double endTime;
    };

    bool writeThreadData(std::vector<ThreadData> &threadData, std::string fileName)
    {
        std::ofstream file;
        file.open(fileName);
        if (!file.is_open())
        {
            std::cerr << "Error opening file: " << fileName << std::endl;
            return false;
        }
        file << "loopIndex,threadIndex,beginTime,endTime" << std::endl;
        for (auto &data : threadData)
        {
            file << data.loopIndex << "," << data.threadIndex << "," << data.beginTime << "," << data.endTime << std::endl;
        }
        file.close();
        return true;
    }

}