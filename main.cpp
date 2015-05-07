#include "nlp.h"
#include "olp.h"
#include "oclp.h"
#include "loadData.h"

#include <iostream>

float calcMeanSquaredError(const int pSize,const float* pTarget, const float* pResult)
{
    float error = 0;
    for(int i=0; i<pSize; ++i)
    {
        error += (pResult[i]-pTarget[i])*(pResult[i]-pTarget[i]);
    }
    return error/pSize;
}

int findHighestIndex(const float* pResults, const int pSize)
{
    int highestIndex = 0;
    float highestValue = 0.0f;
    for(int i=0;i<pSize;++i)
    {
        if(pResults[i]>highestValue)
        {
            highestIndex = i;
            highestValue = pResults[i];
        }
    }
    return highestIndex;
}

float calcCorrect(const float * pClassifications, const std::vector<float> *pTargets, const unsigned int pSize)
{
    if( pTargets != nullptr )
    {
        return calcMeanSquaredError(pSize, pTargets->data(), pClassifications);
    }
    return -1.0f;
}

int OCLTest() {
    OpenCLPerceptron oclp(0.25f,28*28,300,10);

    std::vector<float> *trainingImages          = nullptr;
    std::vector<float> *trainingClassifications = nullptr;
    std::vector<float> *testingImages           = nullptr;
    std::vector<float> *testingClassifications  = nullptr;

    float outputBuffer[10];

    std::cout << "initialising opencl and images." << std::endl;
    if(oclp.initOpenCL() && loadData(&trainingImages,&trainingClassifications,&testingImages,&testingClassifications))
    {
        std::cout << "initialising opencl buffers." << std::endl;
        if(oclp.initTraining(trainingImages,trainingClassifications, testingImages))
        {
            std::cout << "training." << std::endl;
            unsigned int epoch  = 0;
            float correctness   = 1.0f;
            const float target  = 0.05f;
            while(correctness>target)
            {
                ++epoch;
                oclp.trainAll();
                oclp.testAll(outputBuffer);
                correctness = calcCorrect(outputBuffer, testingClassifications,1);

                std::cout << "[";
                for(int i=0; i<9; ++i)
                {
                    std::cout << outputBuffer[i] << ",";
                }
                std::cout << outputBuffer[9] << "] - c:" << correctness << " - e:" << epoch << std::endl;
            }
            return 0;
        }
        else
        {
            return 1;
        }
    }
    else
    {
        return 2;
    }
}

int main()
{
    return OCLTest();
}
