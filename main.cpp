#include "seqOCR.h"
#include "parOCR.h"
#include "loadData.h"

#include <iostream>

enum TrainingType
{
    invalid,
    parallel,
    sequential
};

void parseCommandlineParameters(int argc, char* argv[], TrainingType &pType)
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
                    pType = parallel;
            }
            else if(par == "-sequential" || par == "-s")
            {
                    pType = sequential;
            }
            else
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

    TrainingType type = invalid;

    parseCommandlineParameters(argc, argv, type);

    std::cout << "loading and initialising images." << std::endl;
    if(loadImageData(&trainingData, &trainingClassifications, &testingData, &testingClassifications, inputWidth, outputWidth))
    {
        int retCode = 0;
        if(type == parallel)
        {
            retCode = parallelOCR(trainingData, trainingClassifications, testingData, testingClassifications, inputWidth, outputWidth);
        }
        else if (type == sequential)
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
