#ifndef LOADDATA_H_INCLUDED
#define LOADDATA_H_INCLUDED

#include <vector>
#include <algorithm>
#include <iostream>

#include "idxfile.h"
#include "tdata.h"

//used in loadData to convert the IDXFile data to floats
float normalizeU8(uint8_t pVal)
{
    return (pVal/255.0f);//reducing the interval from [0,255] to [0.0f,1.0f]
}

int copyU8T2I(uint8_t pVal)
{
    return pVal;
}

bool loadImageData( TrainingEnvironment &pTData )
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
            pTData.setInputSampleWidth(trainImageSize);
            pTData.setOutputSampleWidth(10);//one for each classification

            //load all training data as normalized floats into the vector
            pTData.setTrainingData( new std::vector<float>(idxTrainImages.getTotalSize()) );
            std::transform(idxTrainImages.getDataPointer(), idxTrainImages.getDataPointer()+idxTrainImages.getTotalSize(),
                           pTData.getTrainingData()->data(), normalizeU8);
            idxTrainImages.deleteData();

            pTData.setTrainingClassifications( new std::vector<float>(idxTrainLabels.getTotalSize()*10) );
            for(unsigned int i=0; i<idxTrainLabels.getTotalSize(); ++i)
            {
                const unsigned int index = *(idxTrainLabels.getDataPointer()+i);
                pTData.getTrainingClassifications()->data()[i*10+index] = 1.0f;//zeroing not needed, std::vector takes care of that
            }
            idxTrainLabels.deleteData();

            pTData.setTestingData( new std::vector<float>(idxTestImages.getTotalSize()) );
            std::transform(idxTestImages.getDataPointer(), idxTestImages.getDataPointer()+idxTestImages.getTotalSize(),
                           pTData.getTestingData()->data(), normalizeU8);
            idxTestImages.deleteData();

            pTData.setTestingClassifications( new std::vector<int>(idxTestLabels.getTotalSize()) );
            std::transform(idxTestLabels.getDataPointer(), idxTestLabels.getDataPointer()+idxTestLabels.getTotalSize(),
                           pTData.getTestingClassifications()->data(), copyU8T2I);
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
