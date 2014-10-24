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

void trainXOR()
{

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
}

int main()
{
    trainXOR();
    return 0;
}
