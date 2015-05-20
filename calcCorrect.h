#ifndef CALCCORRECT_H_INCLUDED
#define CALCCORRECT_H_INCLUDED

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

float calcCorrect(const float * pClassifications, const std::vector<float> *pTargets, const unsigned int pSampleSize)
{
    if( pTargets != nullptr )
    {
        return calcMeanSquaredError(pSampleSize, pTargets->data(), pClassifications);
    }
    return -1.0f;
}

float calcCorrect(const float * pClassifications, const std::vector<int> *pTargets, const unsigned int pSampleSize)
{
    if( pTargets != nullptr )
    {
        const unsigned int classifications = pTargets->size();
        unsigned int correct = 0;
        for(unsigned int i=0; i<classifications; ++i)
        {
            if(findHighestIndex(pClassifications+(i*pSampleSize),pSampleSize) == pTargets->data()[i])
            {
                ++correct;
            }
        }
        return correct/(float)classifications;
    }
    return -1.0f;
}

#endif // CALCCORRECT_H_INCLUDED
