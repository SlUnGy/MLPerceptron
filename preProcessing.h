#ifndef PREPROCESSING_H_INCLUDED
#define PREPROCESSING_H_INCLUDED

enum PreprocessingType
{
    none,
    scaling,
    rotation,
    deskewing
};

void applyPreprocessing( std::vector<float> *pTEnv, const PreprocessingType &pPP,
                         const int imageWidth, const int imageHeight, const int imageN )
{
    if(pPP != none)
    {

    }
}


#endif // PREPROCESSING_H_INCLUDED
