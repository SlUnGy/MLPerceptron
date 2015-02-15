#include "mlp.h"

#include <ctime>

#include <random>

MultilayerPerceptron::MultilayerPerceptron(const float pEta, const unsigned int pInput, const unsigned int pLayers, const unsigned int *pHidden, const unsigned int pOut)
    : m_eta{pEta}, m_maxHiddenLayer{pLayers}, m_hiddenLayers{new Layer[m_maxHiddenLayer]}, m_outputLayer{new Layer}
{
    for(unsigned int i=0; i< m_maxHiddenLayer; ++i)
    {
        m_hiddenLayers[i]->setupWeights();
        m_hiddenLayers[i]->randomizeWeights();
    }
    //+1 due to constant coefficient, i.e. bias
//	m_hidWeights = new float*[m_inpPerceptrons+1];
//    for (int i=0;i<m_inpPerceptrons+1;i++)
//    {
//        m_hidWeights[i] = new float[m_hidPerceptrons];
//	}
//
//	m_outWeights = new float*[m_hidPerceptrons+1];
//    for (int i=0;i<m_hidPerceptrons+1;i++)
//    {
//        m_outWeights[i] = new float[m_outPerceptrons];
//	}
//    randomizeWeights();
}

bool MultilayerPerceptron::writeToFile(const std::string& pFilename)
{
    return true;
}
