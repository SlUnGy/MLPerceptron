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

int parallelOCR(std::vector<float> *pTrainingImages, std::vector<float> *pTrainingClassifications,
             std::vector<float> *pTestImages, std::vector<int> *pTestClassifications,
             int &pInputwidth, int &pOutputwidth )
{
    std::cout << "constructing opencl perceptron." << std::endl;
    OpenCLPerceptron oclp(0.025f, pInputwidth, 300, pOutputwidth);
    float *outputBuffer = new float[pTrainingClassifications->size()];

    std::cout << "setting up opencl." << std::endl;
    if(oclp.initOpenCL())
    {
        std::cout << "initialising opencl buffers." << std::endl;
        if(oclp.initTraining(pTrainingImages,pTrainingClassifications, pTestImages))
        {
            std::cout << "training." << std::endl;
            unsigned int epoch  = 0;
            float correctness   = 0.0f;
            const float target  = 0.95f;
            while(correctness < target)
            {
                ++epoch;
                oclp.trainAll();
                oclp.testAll(outputBuffer);
                correctness = calcCorrect(outputBuffer, pTestClassifications, pOutputwidth);

                std::cout << "correct: " << correctness << " - epoch: " << epoch << std::endl;
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

int sequentialOCR(std::vector<float> *pTrainingImages, std::vector<float> *pTrainingClassifications,
                  std::vector<float> *pTestImages, std::vector<int> *pTestClassifications,
                  int &pInputwidth, int &pOutputwidth )
{
    std::cout << "constructing one layered perceptron" << std::endl;
    OneLayerPerceptron olp(0.025f, pInputwidth, 300, pOutputwidth);

    const unsigned int trainingSamples  = pTrainingImages->size()/pInputwidth;
    const unsigned int testingSamples   = pTestImages->size()/pInputwidth;
    float *outputBuffer = new float[pTrainingClassifications->size()];

    unsigned int epoch  = 0;
    float correctness   = 0.0f;
    const float target  = 0.95f;
    while(correctness < target)
    {
        ++epoch;
        for(unsigned int i=0; i<trainingSamples; ++i)
        {
            const unsigned int imageOffset = i*pInputwidth;
            const unsigned int classOffset = i*pOutputwidth;
            olp.train(pTrainingImages->data()+imageOffset, pTrainingClassifications->data()+classOffset);
        }
        for(unsigned int i=0; i<testingSamples; ++i)
        {
            const unsigned int imageOffset = i*pInputwidth;
            const unsigned int classOffset = i*pOutputwidth;
            olp.classify(pTestImages->data()+imageOffset, outputBuffer+classOffset);
        }
        correctness = calcCorrect(outputBuffer, pTestClassifications, pOutputwidth);

        std::cout << "correct: " << correctness << " - epoch: " << epoch << std::endl;
    }

    return 0;
}

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
