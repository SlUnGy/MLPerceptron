#include "seqOCR.h"
#include "parOCR.h"
#include "loadData.h"
#include "tdata.h"

#include <iostream>

void parseCommandlineParameters(int argc, char* argv[], TrainingData &pTData)
{
    if(argc > 1)
    {
        std::vector<std::string> cmdParams(argc);
        for(int i=1; i<argc; ++i)
        {
            cmdParams[i] = std::string(argv[i]);
        }
        for(const std::string &par : cmdParams)
        {
            if(par == "-parallel" || par == "-p")
            {
                pTData.setType(parallel);
            }
            else if(par == "-sequential" || par == "-s")
            {
                pTData.setType(sequential);
            }
            else if( par.find_first_not_of(' ') != std::string::npos )//ignore whitespace
            {
                std::cerr << "didn't recognize the option: " << par << std::endl;
            }
        }
    }
}

int main(int argc, char* argv[])
{
    std::vector<float> *trainingData            = nullptr;
    std::vector<float> *trainingClassifications = nullptr;
    std::vector<float> *testingData             = nullptr;
    std::vector<int>   *testingClassifications  = nullptr;

    int inputWidth  = 0;
    int outputWidth = 0;

    TrainingData tdata;

    parseCommandlineParameters(argc, argv, tdata);

    std::cout << "loading and initialising images." << std::endl;
    if(loadImageData(&trainingData, &trainingClassifications, &testingData, &testingClassifications, inputWidth, outputWidth))
    {
        int retCode = 0;
        if(tdata.getType() == parallel)
        {
            retCode = parallelOCR(trainingData, trainingClassifications, testingData, testingClassifications, inputWidth, outputWidth);
        }
        else if (tdata.getType() == sequential)
        {
            retCode = sequentialOCR(trainingData, trainingClassifications, testingData, testingClassifications, inputWidth, outputWidth);
        }
        else
        {
            retCode = -1;
            std::cerr << "training type not recognized. Have you set a command line parameter(-s or -p)?" << std::endl;
        }

        trainingData->clear();
        trainingClassifications->clear();
        testingData->clear();
        testingClassifications->clear();

        return retCode;
    }
    else
    {
        return 3;
    }
}
