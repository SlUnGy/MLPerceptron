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

void trainXOR()
{
    std::cout << "XOR-Training:" << std::endl;
    constexpr int samples    = 4;
    constexpr int inputWidth = 2;

    const float params[samples][inputWidth] = {{0.0f, 0.0f},{0.0f, 1.0f},{1.0f, 0.0f},{1.0f, 1.0f}};
    const float targets[samples] = {0.0f, 1.0f, 1.0, 0.0f};
    const float eta = 0.25f;

    //print the used data
    std::cout << "input=" << std::endl;
    for(int i=0; i<samples; ++i)
    {
        std::cout << "(";
        for(int j=0; j<inputWidth-1; ++j)
        {
            std::cout << params[i][j] << ",";
        }
        std::cout << params[i][inputWidth-1] << ")=" << targets[i] << std::endl;
    }

    MLP mlp(eta, inputWidth, 3 ,1);
    //train the mlp
    const unsigned long iterations = 50000;
    for ( unsigned  long i = 0; i < iterations; ++i )
    {
        for(int i=0; i<samples; ++i)
        {
            mlp.train(params[i],&targets[i]);
        }
    }

    //see if the mlp learned something
    float results[samples];
    for(int i=0; i<samples; ++i)
    {
        float *tmp= mlp.run(params[i]);
        results[i]= tmp[0];
        delete [] tmp;
    }

    //calculate mean squared error
    float error = calcMeanSquaredError(samples, targets, results);

    std::cout << std::endl << "results= " << std::endl;
    for(int i=0; i<samples; ++i)
    {
        std::cout << "(";
        for(int j=0; j<inputWidth-1; ++j)
        {
            std::cout << params[i][j] << ",";
        }
        std::cout << params[i][inputWidth-1] << ")=" << results[i] << std::endl;
    }
    std::cout << "XOR-MSE=" << error << std::endl;
}

void trainOCR()
{
    IDXFile trainLabels("./data/train-labels.idx1-ubyte" );
    IDXFile trainImages("./data/train-images.idx3-ubyte" );

    IDXFile testLabels("./data/t10k-labels.idx1-ubyte" );
    IDXFile testImages("./data/t10k-images.idx3-ubyte" );

    if(trainLabels.getDimensionNumber() == 1 && trainImages.getDimensionNumber() == 3
       && testLabels.getDimensionNumber() == 1 && testImages.getDimensionNumber() == 3)
    {
        std::cout << "OCR-Training" << std::endl;
        constexpr unsigned int samples = 10;
        const unsigned int iterationCount = 500;
        const unsigned int imageSize = trainImages.getDimensions()[1]*trainImages.getDimensions()[2];

        float targets[samples+1][10]={0};
        for(int i=0; i<10; ++i)
        {
            targets[i][i]=1.0f;
        }

        MLP mlp(0.25f,imageSize,30,10);

        for(unsigned int iterations=0;iterations<1;++iterations)
        {
            for(unsigned int i=0;i<trainImages.getDimensions()[0];++i)
            {
                const int targetIndex = (int)trainLabels.getDataPointer()[i];
                if(targetIndex==0 || targetIndex == 8)
                {
                    const uint8_t * const targetImage = trainImages.getDataPointer()+i*imageSize;
//                    for(int j=0;j<imageSize;++j)
//                    {
//                        std::cout << targetImage[j];
//                        if(!(j%(trainImages.getDimensions()[1])))
//                        {
//                            std::cout << std::endl;
//                        }
//                    }
                    mlp.train(targetImage,targets[targetIndex]);
                }
            }
        }
        std::cout << "using test data" << std::endl;
        for(unsigned int i=0,t=0;t<30&&i<trainImages.getDimensions()[0];++i)
        {
            const int targetIndex = (int)testLabels.getDataPointer()[i];
            if(targetIndex == 0 || targetIndex == 8)
            {
                const uint8_t * const targetImage = testImages.getDataPointer()+i*imageSize;
                float* tmpResults = mlp.run(targetImage);
                std::cout << "[";
                for(int k=0;k<samples-1;++k)
                {
                    std::cout << tmpResults[k] << ",";
                }
                std::cout << tmpResults[samples-1] << "],mse="
                << calcMeanSquaredError(samples,targets[targetIndex],tmpResults) << ","
                << findHighestIndex(tmpResults,10) << "?=" << targetIndex
                << std::endl << std::endl;
                delete [] tmpResults;
                ++t;
            }
        }
    }
    else
    {
        std::cerr << "files appear not to have the correct amount of dimensions" << std::endl;
    }
}

int main()
{
    //trainXOR();
    trainOCR();
    return 0;
}
