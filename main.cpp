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

float calcCorrect(const float * pClassifications, const std::vector<float> *pTargets, const unsigned int pSampleSize)
{
    if( pTargets != nullptr )
    {
        return calcMeanSquaredError(pSampleSize, pTargets->data(), pClassifications);
    }
    return -1.0f;
}

float calcCorrect(const float * pClassifications, const std::vector<int> *pTargets, const unsigned int pSampleSize)
{
    if( pTargets != nullptr )
    {
        const unsigned int classifications = pTargets->size()/pSampleSize;
        unsigned int correct = 0;
        for(unsigned int i=0; i<classifications; ++i)
        {
            if(findHighestIndex(pClassifications+(i*pSampleSize),pSampleSize) == pTargets->data()[i])
            {
                ++correct;
            }
        }
        return correct/(float)classifications;
    }
    return -1.0f;
}

int OCLTest() {
    std::vector<float> *trainingData            = nullptr;
    std::vector<float> *trainingClassifications = nullptr;
    std::vector<float> *testingData             = nullptr;
    std::vector<int>   *testingClassifications  = nullptr;

    int inputWidth  = 0;
    int outputWidth = 0;

    std::cout << "loading and initialising images." << std::endl;
    if(loadImageData(&trainingData,&trainingClassifications,&testingData,&testingClassifications,inputWidth,outputWidth))
    {
        std::cout << "constructing opencl perceptron." << std::endl;
        OpenCLPerceptron oclp(0.05f, inputWidth, 300, outputWidth);
        float *outputBuffer = new float[trainingClassifications->size()];

        std::cout << "setting up opencl." << std::endl;
        if(oclp.initOpenCL())
        {
            std::cout << "initialising opencl buffers." << std::endl;
            if(oclp.initTraining(trainingData,trainingClassifications, testingData))
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
                    correctness = calcCorrect(outputBuffer, testingClassifications, outputWidth);

                    std::cout << "[" << outputBuffer[testingClassifications->size()-1] << "] - ";
                    std::cout << "c:" << correctness << " - e:" << epoch << std::endl;
                }
                delete [] outputBuffer;
                trainingData->clear();
                trainingClassifications->clear();
                testingData->clear();
                testingClassifications->clear();
                return 0;
            }
            else
            {
                delete [] outputBuffer;
                return 1;
            }
        }
        else
        {
            delete [] outputBuffer;
            return 2;
        }
    }
    else
    {
        return 3;
    }
}

int main()
{
    return OCLTest();
}
