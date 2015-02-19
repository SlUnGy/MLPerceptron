#include "mlp.h"

#include <ctime>

#include <random>

MultilayerPerceptron::MultilayerPerceptron(const float pEta, const unsigned int pInput, const unsigned int pLayers, const unsigned int *pHidden, const unsigned int pOut)
    : m_eta{pEta}, m_maxHiddenLayer{pLayers}, m_hiddenLayers{new Layer[m_maxHiddenLayer]}, m_outputLayer{new Layer(m_hiddenLayers[m_maxHiddenLayer]+1,pOut)}
{
    m_hiddenLayers[0].setupWeights(pInput+1, pHidden[0]);
    m_hiddenLayers[0].randomizeWeights();
    for(unsigned int i=1; i< m_maxHiddenLayer; ++i)
    {
        m_hiddenLayers[i].setupWeights(pHidden[i-1]+1, pHidden[i]);
        m_hiddenLayers[i].randomizeWeights();
    }

    m_outputLayer->randomizeWeights();
}

bool MultilayerPerceptron::writeToFile(const std::string& pFilename)
{
    return true;
}
