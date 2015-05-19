#include "seqOCR.h"
#include "parOCR.h"
#include "loadData.h"

#include <iostream>

enum TrainingType
{
    parallel,
    sequential
};

int main(int argc, char* argv[])
{
    std::vector<float> *trainingData            = nullptr;
    std::vector<float> *trainingClassifications = nullptr;
    std::vector<float> *testingData             = nullptr;
    std::vector<int>   *testingClassifications  = nullptr;

    int inputWidth  = 0;
    int outputWidth = 0;

    std::cout << "loading and initialising images." << std::endl;
    if(loadImageData(&trainingData, &trainingClassifications, &testingData, &testingClassifications, inputWidth, outputWidth))
    {
        int retCode = 0;
        retCode = parallelOCR(trainingData, trainingClassifications, testingData, testingClassifications, inputWidth, outputWidth);

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
