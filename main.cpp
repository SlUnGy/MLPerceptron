#include <iostream>

#include "mlp.h"
#include "idxfile.h"

int main()
{
    IDXFile trainLabels("./data/train-labels.idx1-ubyte", false);


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

    MLP bp(eta, inputWidth, 3 ,1);
    //train the mlp
    const unsigned long iterations = 5000;
    for ( unsigned  long i = 0; i < iterations; ++i )
    {
        for(int i=0; i<samples; ++i)
        {
            bp.train(params[i],targets[i]);
        }
    }

    //see if it learned something
    float results[samples];
    for(int i=0; i<samples; ++i)
    {
        float *tmp= bp.run(params[i]);
        results[i]= tmp[0];
        delete [] tmp;
    }

    //calculate mean squared error
    float error = 0;
    for(int i=0; i<samples; ++i)
    {
        error += (results[i]-targets[i])*(results[i]-targets[i]);
    }
    error *= 1.0f/samples;

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
    std::cout << "MSE  = " << error << std::endl;

    return 0;
}
