#ifndef LOADDATA_H_INCLUDED
#define LOADDATA_H_INCLUDED

#include <vector>
#include <algorithm>

#include "idxfile.h"

//used in loadData to convert the IDXFile data to floats
float copyU8T2F(uint8_t pVal)
{
    return pVal;//reducing the interval from [0,255] to [0.0f,1.0f] might be useful?
}

bool loadXORData(std::vector<float> **pTrainingImages, std::vector<float> **pTrainingClassifications,
                 std::vector<float> **pTestImages, std::vector<float> **pTestClassifications,
                 int &pInputwidth, int &pOutputwidth )
{
    *pTrainingImages            = new std::vector<float>{0.f,0.f, 0.f,1.f, 1.f,0.f, 1.f,1.f};
    *pTrainingClassifications   = new std::vector<float>{    0.f,     1.f,     1.f,     0.f};
    *pTestImages                = new std::vector<float>{0.f,0.f, 0.f,1.f, 1.f,0.f, 1.f,1.f};
    *pTestClassifications       = new std::vector<float>{    0.f,     1.f,     1.f,     0.f};
    pInputwidth     = 2;
    pOutputwidth    = 1;
    return true;
}

bool loadData(std::vector<float> **pTrainingImages, std::vector<float> **pTrainingClassifications,
              std::vector<float> **pTestImages, std::vector<float> **pTestClassifications,
              int &pInputwidth, int &pOutputwidth )
{
    IDXFile idxTrainLabels("./data/train-labels.idx1-ubyte" );
    IDXFile idxTrainImages("./data/train-images.idx3-ubyte" );

    IDXFile idxTestLabels("./data/t10k-labels.idx1-ubyte" );
    IDXFile idxTestImages("./data/t10k-images.idx3-ubyte" );

    //test if all files have been correctly read and have the right sizes
    //dimensions==3 on images is needed due to dim[0] being the number of images and dim[1,2] being width and height
    if(!idxTrainLabels.hasError() && idxTrainLabels.getDimensionNumber() == 1 &&
       !idxTrainImages.hasError() && idxTrainImages.getDimensionNumber() == 3 &&
       !idxTestLabels.hasError()  && idxTestLabels.getDimensionNumber()  == 1 &&
       !idxTestImages.hasError()  && idxTestImages.getDimensionNumber()  == 3 )
    {
        const unsigned int trainImageSize   = idxTrainImages.getDimensions()[1]*idxTrainImages.getDimensions()[2];
        const unsigned int testImageSize    = idxTestImages.getDimensions()[1]*idxTestImages.getDimensions()[2];

        if(trainImageSize == testImageSize)
        {
            pInputwidth  = trainImageSize;
            pOutputwidth = 1;

            *pTrainingImages = new std::vector<float>(idxTrainImages.getTotalSize());
            std::transform(idxTrainImages.getDataPointer(), idxTrainImages.getDataPointer()+idxTrainImages.getTotalSize(),
                           (*pTrainingImages)->data(), copyU8T2F);
            idxTrainImages.deleteData();
//padding could be done in both classifications, to make classification easier
            *pTrainingClassifications = new std::vector<float>(idxTrainLabels.getTotalSize());
            std::transform(idxTrainLabels.getDataPointer(), idxTrainLabels.getDataPointer()+idxTrainLabels.getTotalSize(),
                           (*pTrainingClassifications)->data(), copyU8T2F);
            idxTrainLabels.deleteData();

            *pTestImages = new std::vector<float>(idxTestImages.getTotalSize());
            std::transform(idxTestImages.getDataPointer(), idxTestImages.getDataPointer()+idxTestImages.getTotalSize(),
                           (*pTestImages)->data(), copyU8T2F);
            idxTestImages.deleteData();

            *pTestClassifications = new std::vector<float>(idxTestLabels.getTotalSize());
            std::transform(idxTestLabels.getDataPointer(), idxTestLabels.getDataPointer()+idxTestLabels.getTotalSize(),
                           (*pTestClassifications)->data(), copyU8T2F);
            idxTestLabels.deleteData();

            return true;
        }
        else
        {
            std::cout << "error: image size of training (" << trainImageSize
            << ") and image size of test data (" << testImageSize << ") is not equal" << std::endl;
            return false;
        }
    }
    else
    {
        std::cerr << "error while loading data" << std::endl;
        return false;
    }
}

#endif // LOADDATA_H_INCLUDED
