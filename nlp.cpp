#include "nlp.h"

#include <ctime>

#include <random>

NLayerPerceptron::NLayerPerceptron(const float pEta, const unsigned int pInput, const unsigned int pLayers, const unsigned int *pHidden, const unsigned int pOut)
    : m_eta{pEta}, m_maxHiddenLayer{pLayers}, m_hiddenLayers{new Layer[m_maxHiddenLayer]}, m_outputLayer{new Layer()}
{
    m_hiddenLayers[0].setupWeights(pInput+1, pHidden[0]);
    m_hiddenLayers[0].randomizeWeights();
    for(unsigned int i=1; i< m_maxHiddenLayer; ++i)
    {
        m_hiddenLayers[i].setupWeights(pHidden[i-1]+1, pHidden[i]);
        m_hiddenLayers[i].randomizeWeights();
    }

    m_outputLayer->setupWeights(m_hiddenLayers[m_maxHiddenLayer-1].m_width+1,pOut);
    m_outputLayer->randomizeWeights();
}

bool NLayerPerceptron::writeToFile(const std::string& pFilename)
{
    return true;
}

void NLayerPerceptron::train(const float* pIn,const float* pTarget)
{
    const float *tmpInput   = pIn;
    float **tmpHiddenOutput = new float*[m_maxHiddenLayer];

    for(unsigned int i=0; i<m_maxHiddenLayer; ++i)
    {
        tmpHiddenOutput[i] = new float[m_hiddenLayers[i].m_width];
        for(unsigned int j=0; j<m_hiddenLayers[i].m_width; ++j)
        {
            tmpHiddenOutput[i][j] = 1*m_hiddenLayers[i].m_weights[0][j];
            for(unsigned int k=1; k<m_hiddenLayers[i].m_in; ++k)
            {
                tmpHiddenOutput[i][j] += tmpInput[j-1]*m_hiddenLayers[i].m_weights[k][j];
            }
            tmpHiddenOutput[i][j] = sigmoid(tmpHiddenOutput[i][j]);
        }
        tmpInput = tmpHiddenOutput[i];
    }

    float tmpOutOutput [m_outputLayer->m_width];
    for(unsigned int i=0; i<m_outputLayer->m_width; ++i)
    {
        tmpOutOutput[i] = 1*m_outputLayer->m_weights[0][i];
        for(unsigned int j=1; j<m_outputLayer->m_in; ++j)
        {
            tmpOutOutput[i] += tmpInput[j-1] * m_outputLayer->m_weights[j][i];
        }
        tmpOutOutput[i] = sigmoid(tmpOutOutput[i]);
    }

    //Backpropagation
    //error calculation
    float outDelta[m_outputLayer->m_width];
    for(unsigned int i=0; i<m_outputLayer->m_width; ++i)
    {
        outDelta[i] = tmpOutOutput[i]*(1-tmpOutOutput[i])*(pTarget[i]-tmpOutOutput[i]);
    }

    //applying delta to reduce error in output layer
    for(unsigned int i=0; i<m_outputLayer->m_width; ++i)
    {
        //apply to bias
        m_outputLayer->m_weights[0][i] += m_eta*1*outDelta[i];
        //weights[j+1][] -> skip the constant coeffecient we already did above
        for(unsigned int j=0; j<m_outputLayer->m_in-1; ++j)
        {
            m_outputLayer->m_weights[j+1][i] += m_eta*tmpOutOutput[j]*outDelta[i];
        }
    }

    Layer *previousLayer    = m_outputLayer;
    float **hidDelta        = new float*[m_maxHiddenLayer];
    float *previousDelta    = outDelta;

    //calculate hidden layer error
    for(int k=m_maxHiddenLayer-1;k>-1;--k)
    {
        tmpInput = tmpHiddenOutput[k];
        hidDelta[k] = new float[m_hiddenLayers[k].m_width];
        for(unsigned int i=0; i<m_hiddenLayers[k].m_width; ++i)
        {
            hidDelta[k][i] = tmpInput[i]*(1-tmpInput[i]);
            float tmpSum = 0;
            for(unsigned int j=0; j<previousLayer->m_width; ++j)
            {
                tmpSum += previousLayer->m_weights[i+1][j]*previousDelta[j];
            }
            hidDelta[k][i] *= tmpSum;
        }
        previousLayer = &m_hiddenLayers[k];
        previousDelta = hidDelta[k];
    }

    //apply delta to weights
    tmpInput = pIn;
    for(unsigned int k=0; k<m_maxHiddenLayer; ++k)
    {
        for(unsigned int i=0; i<m_hiddenLayers[k].m_width; ++i)
        {
            m_hiddenLayers[k].m_weights[0][i] += m_eta*1*hidDelta[k][i];
            for(unsigned int j=0; j<m_hiddenLayers[k].m_in-1; ++j)
            {
                m_hiddenLayers[k].m_weights[j+1][i] += m_eta*tmpInput[j]*hidDelta[k][i];
            }
        }
        tmpInput = tmpHiddenOutput[k];
    }

    for(unsigned int i=0; i<m_maxHiddenLayer;++i)
    {
        delete [] hidDelta[i];
        delete [] tmpHiddenOutput[i];
    }
    delete [] hidDelta;
    delete [] tmpHiddenOutput;
}

float* NLayerPerceptron::classify(const float *pIn)
{
    const float *tmpInput   = pIn;
    float **tmpHiddenOutput = new float*[m_maxHiddenLayer];

    for(unsigned int i=0; i<m_maxHiddenLayer; ++i)
    {
        tmpHiddenOutput[i] = new float[m_hiddenLayers[i].m_width];
        for(unsigned int j=0; j<m_hiddenLayers[i].m_width; ++j)
        {
            tmpHiddenOutput[i][j] = 1*m_hiddenLayers[i].m_weights[0][j];
            for(unsigned int k=1; k<m_hiddenLayers[i].m_in; ++k)
            {
                tmpHiddenOutput[i][j] += tmpInput[j-1]*m_hiddenLayers[i].m_weights[k][j];
            }
            tmpHiddenOutput[i][j] = sigmoid(tmpHiddenOutput[i][j]);
        }
        tmpInput = tmpHiddenOutput[i];
    }

    float *tmpOutOutput = new float[m_outputLayer->m_width];
    for(unsigned int i=0; i<m_outputLayer->m_width; ++i)
    {
        tmpOutOutput[i] = 1*m_outputLayer->m_weights[0][i];
        for(unsigned int j=1; j<m_outputLayer->m_in; ++j)
        {
            tmpOutOutput[i] += tmpInput[j-1] * m_outputLayer->m_weights[j][i];
        }
        tmpOutOutput[i] = sigmoid(tmpOutOutput[i]);
    }

    for(unsigned int i=0; i<m_maxHiddenLayer;++i)
    {
        delete [] tmpHiddenOutput[i];
    }
    delete [] tmpHiddenOutput;

    return tmpOutOutput;
}
