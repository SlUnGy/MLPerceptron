#ifndef SEQOCR_H_INCLUDED
#define SEQOCR_H_INCLUDED

#include <vector>
#include <iostream>

#include "olp.h"
#include "calcCorrect.h"

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
            const unsigned int imageOffset = i*pInputwidth;//offsets are needed, due to all data being stored in a contiguous array
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


#endif // SEQOCR_H_INCLUDED
