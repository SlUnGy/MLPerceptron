#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <iterator>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include "nlp.h"
#include "olp.h"
#include "oclp.h"
#include "idxfile.h"

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

void trainOCR()
{
    IDXFile trainLabels("./data/train-labels.idx1-ubyte" );
    IDXFile *trainImages = new IDXFile("./data/train-images.idx3-ubyte" );

    IDXFile testLabels("./data/t10k-labels.idx1-ubyte" );
    IDXFile *testImages = new IDXFile("./data/t10k-images.idx3-ubyte" );

    //test if all files have been correctly read and have the right sizes
    if(!trainLabels.hasError() && trainLabels.getDimensionNumber() == 1 &&
       !trainImages->hasError() && trainImages->getDimensionNumber() == 3 &&
       !testLabels.hasError() && testLabels.getDimensionNumber() == 1 &&
       !testImages->hasError() && testImages->getDimensionNumber() == 3 &&
       trainImages->getDimensions()[1]*trainImages->getDimensions()[2] ==
       testImages->getDimensions()[1]*testImages->getDimensions()[2])
    {
        const unsigned int imageSize        = trainImages->getDimensions()[1]*trainImages->getDimensions()[2];
        std::cout << "Preparing images." << std::endl;
        float **fTrainImages = new float*[trainImages->getDimensions()[0]];
        for(unsigned int i=0;i<trainImages->getDimensions()[0];++i)
        {
            fTrainImages[i] = new float[imageSize];
            for(unsigned int j=0;j<imageSize;++j)
            {
                fTrainImages[i][j] = *(trainImages->getDataPointer()+i*imageSize+j);
            }
        }
        trainImages->deleteData();

        float **fTestImages = new float*[testImages->getDimensions()[0]];
        for(unsigned int i=0;i<testImages->getDimensions()[0];++i)
        {
            fTestImages[i] = new float[imageSize];
            for(unsigned int j=0;j<imageSize;++j)
            {
                fTestImages[i][j] = *(testImages->getDataPointer()+i*imageSize+j);
            }
        }
        testImages->deleteData();

        std::cout << "OCR-Training." << std::endl;
        constexpr unsigned int samples      = 10;
        constexpr float eta                 = 0.025f;
        const unsigned int hiddenNodes[]    = {300};
        const unsigned int hiddenLayers     = 1;
        std::cout << "using images with: " << imageSize << " pixels." << std::endl;
        std::cout << "using mlp with: ";
        for(unsigned int i=0; i<hiddenLayers; ++i)
        {
            std::cout << hiddenNodes[i];
            if(i<hiddenLayers-1)
            {
                std::cout << ",";
            }
        }
        std::cout << " hidden nodes and eta: " << eta << "." << std::endl;

        std::cout << "setting up data." << std::endl;
        float targets[samples][samples]={0};
        for(unsigned int i=0; i<samples; ++i)
        {
            targets[i][i]=1.0f;
        }

        OneLayerPerceptron mlp(eta,imageSize,hiddenNodes[0],samples)/*(eta,imageSize,hiddenLayers,hiddenNodes,samples)*/;

        float error     = 1.0f;
        const unsigned int testTotal = testImages->getDimensions()[0];
        while(error > 0.05)
        {
            std::cout << "training the mlp." << std::endl;
            for(unsigned int iterations=0; iterations<1; ++iterations)
            {
                for(unsigned int i=0; i<trainImages->getDimensions()[0]; ++i)
                {
                    const int targetIndex = (int)trainLabels.getDataPointer()[i];
                    mlp.train(fTrainImages[i],targets[targetIndex]);
                }
            }
            std::cout << "classifying test data." << std::endl;
            int correct = 0;
            for(unsigned int i=0; i<testTotal; ++i)
            {
                const int targetIndex = (int)testLabels.getDataPointer()[i];
                float* tmpResults = mlp.classify(fTestImages[i]);
                if(findHighestIndex(tmpResults,10)==targetIndex)
                {
                    ++correct;
                }
                delete [] tmpResults;
            }
            error = (1-correct/(float)testTotal);
            std::cout << "error: " << error << "." <<std::endl;
        }

        //free used memory
        for(unsigned int i=0;i<trainImages->getDimensions()[0];++i)
        {
            delete [] fTrainImages[i];
        }
        delete [] fTrainImages;
        for(unsigned int i=0;i<testImages->getDimensions()[0];++i)
        {
            delete [] fTestImages[i];
        }
        delete [] fTestImages;
    }
    else
    {
        std::cerr << "required files appear to contain errors" << std::endl;
    }
}

bool loadData(std::vector<float> &trainImg, std::vector<float> &trainClf, std::vector<float> &testImg,float** testClf)
{
    return true;
}

float calcCorrectPerc(float** pClassifications,float** pTargets)
{
    return 1.0f;
}

int OCLTest() {
    OpenCLPerceptron oclp;

    std::vector<float> trainingImages;
    std::vector<float> trainingClassifications;
    std::vector<float> testingImages;
    float** testingClassifications;

    if(oclp.initOpenCL() && loadData(trainingImages,trainingClassifications,testingImages,testingClassifications))
    {
        if(oclp.initTraining(trainingImages,trainingClassifications) && oclp.initTesting(testingImages))
        {
            float correctPercentage = 0.0f;
            const float target      = 0.5f;
            while(correctPercentage>target)
            {
                oclp.trainAll();
                correctPercentage = calcCorrectPerc(oclp.testAll(),testingClassifications);
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
