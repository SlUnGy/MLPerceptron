#include "olp.h"

#include <ctime>

#include <random>

OneLayerPerceptron::OneLayerPerceptron(const float pEta, const int pInputPerceptrons, const int pHiddenPerceptrons, const int pOutputPerceptrons)
    : m_eta{pEta}, m_hidPerceptrons{pHiddenPerceptrons}, m_inpPerceptrons{pInputPerceptrons}, m_outPerceptrons{pOutputPerceptrons}
{
    //+1 due to constant coefficient, i.e. bias
	m_hidWeights = new float*[m_inpPerceptrons+1];
    for (int i=0;i<m_inpPerceptrons+1;i++)
    {
        m_hidWeights[i] = new float[m_hidPerceptrons];
	}

	m_outWeights = new float*[m_hidPerceptrons+1];
    for (int i=0;i<m_hidPerceptrons+1;i++)
    {
        m_outWeights[i] = new float[m_outPerceptrons];
	}
    randomizeWeights();
}

void OneLayerPerceptron::randomizeWeights()
{
	/*
        doesn't work on mingw&windows....
        std::random_device rd;
    */
	std::mt19937 mt(time(NULL));
	std::uniform_real_distribution<> distribution(-1, 1);

    for (int i=0;i<m_inpPerceptrons+1;i++)
    {
		for (int j=0;j<m_hidPerceptrons;j++) {
			m_hidWeights[i][j] = distribution(mt);
		}
	}

    for (int i=0;i<m_hidPerceptrons+1;i++)
    {
		for (int j=0;j<m_outPerceptrons;j++) {
			m_outWeights[i][j] = distribution(mt);
		}
	}
}

bool OneLayerPerceptron::writeToFile(const std::string& pFilename)
{
    return true;
}

void OneLayerPerceptron::train(const float* pIn,const float* pTarget)
{
    float hidOutput[m_hidPerceptrons];
    for(int i=0; i<m_hidPerceptrons; ++i)
    {
        //add constant
        hidOutput[i] = 1*m_hidWeights[0][i];
        //sum up all inputs*weightings
        for(int j=1; j<m_inpPerceptrons+1; ++j)
        {
            hidOutput[i] += pIn[j-1] * m_hidWeights[j][i];
        }
        hidOutput[i] = sigmoid(hidOutput[i]);
    }

    float output[m_outPerceptrons];
    for(int i=0; i<m_outPerceptrons; ++i)
    {
        output[i] = 1*m_outWeights[0][i];
        for(int j=1; j<m_hidPerceptrons+1; ++j)
        {
            output[i] += hidOutput[j-1] * m_outWeights[j][i];
        }
        output[i] = sigmoid(output[i]);
    }

    //Backpropagation
    //error calculation
    float outDelta[m_outPerceptrons];
    for(int i=0; i<m_outPerceptrons; ++i)
    {
        outDelta[i] = output[i]*(1-output[i])*(pTarget[i]-output[i]);
    }

    //applying delta to reduce error in output layer
    for(int i=0; i<m_outPerceptrons; ++i)
    {
        m_outWeights[0][i] += m_eta*1*outDelta[i];
        for(int j=0; j<m_hidPerceptrons; ++j)
        {
            m_outWeights[j+1][i] += m_eta*hidOutput[j]*outDelta[i];
        }
    }

    float hidError[m_hidPerceptrons];
    for(int i=0; i<m_hidPerceptrons; ++i)
    {
        //m_outWeights[i+1][] -> skip the constant coeffecient
        hidError[i] = hidOutput[i]*(1-hidOutput[i]);
        float tmpSum = 0;
        for(int j=0; j<m_outPerceptrons; ++j)
        {
            tmpSum += m_outWeights[i+1][j]*outDelta[j];
        }
        hidError[i] *= tmpSum;
    }

    for(int i=0; i<m_hidPerceptrons; ++i)
    {
        m_hidWeights[0][i] += m_eta*1*hidError[i];
        for(int j=0; j<m_inpPerceptrons; ++j)
        {
            m_hidWeights[j+1][i] += m_eta*pIn[j]*hidError[i];
        }
    }
}

float* OneLayerPerceptron::classify(const float *pIn)
{
    float hidOutput[m_hidPerceptrons];
    for(int i=0; i<m_hidPerceptrons; ++i)
    {
        //add constant
        hidOutput[i] = 1*m_hidWeights[0][i];
        //sum up all inputs*weightings
        for(int k=1; k<m_inpPerceptrons+1; ++k)
        {
            hidOutput[i] += pIn[k-1] * m_hidWeights[k][i];
        }
        hidOutput[i] = sigmoid(hidOutput[i]);
    }

    float* output = new float[m_outPerceptrons]();
    for(int i=0; i<m_outPerceptrons; ++i)
    {
        output[i] = 1*m_outWeights[0][i];
        for(int j=1; j<m_hidPerceptrons+1; ++j)
        {
            output[i] += hidOutput[j-1] * m_outWeights[j][i];
        }
        output[i] = sigmoid(output[i]);
    }

    return output;
}
