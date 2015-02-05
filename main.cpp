#include <iostream>

#include "mlp.h"
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
    IDXFile trainImages("./data/train-images.idx3-ubyte" );

    IDXFile testLabels("./data/t10k-labels.idx1-ubyte" );
    IDXFile testImages("./data/t10k-images.idx3-ubyte" );

    if(!trainLabels.hasError() && trainLabels.getDimensionNumber() == 1 &&
       !trainImages.hasError() && trainImages.getDimensionNumber() == 3 &&
       !testLabels.hasError() && testLabels.getDimensionNumber() == 1 &&
       !testImages.hasError() && testImages.getDimensionNumber() == 3)
    {
        std::cout << "OCR-Training." << std::endl;
        constexpr unsigned int samples      = 10;
        constexpr float eta                 = 0.025f;
        const unsigned int imageSize        = trainImages.getDimensions()[1]*trainImages.getDimensions()[2];
        const unsigned int hiddenNodes      = 300;
        std::cout << "using images with: " << imageSize << " pixels." << std::endl;
        std::cout << "using mlp with: " << hiddenNodes << " hidden nodes and eta: " << eta << "." << std::endl;

        std::cout << "setting up data." << std::endl;
        float targets[samples][samples]={0};
        for(unsigned int i=0; i<samples; ++i)
        {
            targets[i][i]=1.0f;
        }

        MultilayerPerceptron mlp(eta,imageSize,hiddenNodes,samples);

        float error     = 1.0f;
        const unsigned int testTotal = testImages.getDimensions()[0];
        while(error > 0.05){
        std::cout << "training the mlp." << std::endl;
            for(unsigned int iterations=0; iterations<1; ++iterations)
            {
                for(unsigned int i=0; i<trainImages.getDimensions()[0]; ++i)
                {
                    const int targetIndex = (int)trainLabels.getDataPointer()[i];
                    const uint8_t * const targetImage = trainImages.getDataPointer()+i*imageSize;
                    mlp.train(targetImage,targets[targetIndex]);
                }
            }
            std::cout << "classifying test data." << std::endl;
            int correct = 0;
            for(unsigned int i=0; i<testTotal; ++i)
            {
                const int targetIndex = (int)testLabels.getDataPointer()[i];
                const uint8_t * const targetImage = testImages.getDataPointer()+i*imageSize;
                float* tmpResults = mlp.classify(targetImage);
                if(findHighestIndex(tmpResults,10)==targetIndex)
                {
                    ++correct;
                }
                delete [] tmpResults;
            }
            error = (1-correct/(float)testTotal);
            std::cout << "error: " << error << "." <<std::endl;
        }
    }
    else
    {
        std::cerr << "required files appear to contain errors" << std::endl;
    }
}

int main()
{
    trainOCR();
    return 0;
}
