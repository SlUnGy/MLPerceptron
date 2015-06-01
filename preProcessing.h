#ifndef PREPROCESSING_H_INCLUDED
#define PREPROCESSING_H_INCLUDED

enum PreprocessingType
{
    none,
    scaling,
    rotation,
    deskewing
};

void deskewImages( std::vector<float> *pImages, const int pImgWidth, const int pImgHeight, const int pImgN )
{

}

void applyPreprocessing( std::vector<float> *pImages, const PreprocessingType &pPP,
                         const int pImgWidth, const int pImgHeight, const int pImgN )
{
    switch(pPP)
    {
    case none:
        break;
    case scaling:
        break;
    case rotation:
        break;
    case deskewing:
        deskewImages(pImages, pImgWidth, pImgHeight, pImgN);
        break;
    }
}


#endif // PREPROCESSING_H_INCLUDED
