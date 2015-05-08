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
    std::vector<float> *trainingImages          = nullptr;
    std::vector<float> *trainingClassifications = nullptr;
    std::vector<float> *testingImages           = nullptr;
    std::vector<float> *testingClassifications  = nullptr;

    int inputWidth  = 0;
    int outputWidth = 0;

    std::cout << "loading and initialising images." << std::endl;
    if(loadXORData(&trainingImages,&trainingClassifications,&testingImages,&testingClassifications,inputWidth,outputWidth))
    {
        std::cout << "constructing opencl perceptron." << std::endl;
        OpenCLPerceptron oclp(1.0f, inputWidth, 30, outputWidth);
        float *outputBuffer = new float[testingClassifications->size()]{0.0f};

        std::cout << "setting up opencl." << std::endl;
        if(oclp.initOpenCL())
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
                    for(unsigned int i=0; i<testingClassifications->size()-1; ++i)
                    {
                        std::cout << outputBuffer[i] << ",";
                    }
                    std::cout << outputBuffer[testingClassifications->size()-1] << "] - ";
                    std::cout << "c:" << correctness << " - e:" << epoch << std::endl;
                }
                delete [] outputBuffer;
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
