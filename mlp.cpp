#include "mlp.h"

#include <random>
#include <ctime>

#include <iostream>

MLP::MLP()
    : m_eta{0.5f}, m_hidPerceptrons{2}, m_inpPerceptrons{2}, m_outPerceptrons{1}
{
	/*
        doesn't work on mingw&windows....
        std::random_device rd;
    */
	std::mt19937 mt(time(NULL));
	std::uniform_real_distribution<> distribution(-1, 1);

    //+1 due to constant coefficient
	m_hidWeights = new float*[m_inpPerceptrons+1];
    for (int i=0;i<m_inpPerceptrons+1;i++) {
        m_hidWeights[i] = new float[m_hidPerceptrons];

		for (int j=0;j<m_hidPerceptrons;j++) {
			m_hidWeights[i][j] = distribution(mt);
		}
	}

	m_outWeights = new float*[m_hidPerceptrons+1];
    for (int i=0;i<m_hidPerceptrons+1;i++) {
        m_outWeights[i] = new float[m_outPerceptrons];

		for (int j=0;j<m_outPerceptrons;j++) {
			m_outWeights[i][j] = distribution(mt);
		}
	}
}


void MLP::train(const float* pIn,const float pTarget)
{
    float hidOutput[m_hidPerceptrons][m_outPerceptrons];
    for(int i=0; i<m_hidPerceptrons; ++i)
    {
        for(int j=0; j<m_outPerceptrons; ++j)
        {
            //add constant
            hidOutput[i][j] = 1*m_hidWeights[0][i];
            //sum up all inputs*weightings
            for(int k=1; k<m_inpPerceptrons+1; ++k)
            {
                hidOutput[i][j] += pIn[k-1] * m_hidWeights[k][i];
            }
            hidOutput[i][j] = sigmoid(hidOutput[i][j]);
        }
        //std::cout << "hO [" << i << "][" << 0 << "] =" << hidOutput[i][0] << std::endl;
    }

    float output[m_outPerceptrons];
    for(int i=0; i<m_outPerceptrons; ++i)
    {
        output[i] = sigmoid( 1 * m_outWeights[0][i] + hidOutput[0][i] * m_outWeights[1][i] + hidOutput[1][i] * m_outWeights[2][i] );
        //std::cout << "output [" << i << "] =" << output[i] << std::endl;
    }

    //Backpropagation
    //error calculation
    float outError[m_outPerceptrons];
    for(int i=0; i<m_outPerceptrons; ++i)
    {
        outError[i] = output[i]*(1-output[i])*(pTarget-output[i]);
        //std::cout << "oE [" << i << "] =" << outError[i] << std::endl;
    }

    float hidError[m_hidPerceptrons];
    for(int i=0; i<m_hidPerceptrons; ++i)
    {
        //m_outWeights[i+1][] -> skip the constant coeffecient
        hidError[i] = hidOutput[i][0]*(1-hidOutput[i][0])*m_outWeights[i+1][0]*outError[0];
        //std::cout << "hE [" << i << "] =" << hidError[i] << std::endl;
    }

    //applying delta to reduce error
    for(int i=0; i<m_outPerceptrons; ++i)
    {
        m_outWeights[0][i] += m_eta*1*outError[i];
        m_outWeights[1][i] += m_eta*hidOutput[0][0]*outError[i];
        m_outWeights[2][i] += m_eta*hidOutput[1][0]*outError[i];
        //std::cout << "oW [1] = [" << m_outWeights[0] << "," << m_outWeights[1] << ","
        //                << m_outWeights[2] << "]" << std::endl;
    }

    for(int i=0; i<m_hidPerceptrons; ++i)
    {
        m_hidWeights[0][i] += m_eta*1*hidError[i];
        m_hidWeights[1][i] += m_eta*pIn[0]*hidError[i];
        m_hidWeights[2][i] += m_eta*pIn[1]*hidError[i];
        //std::cout << "hW [" << i << "] = [" << m_hidWeights[0][i] << "," << m_hidWeights[1][i] << ","
        //                << m_hidWeights[2][i] << "]" << std::endl;
    }
    //std::cout << std::endl;
}

float MLP::run(const float *pIn)
{
    float hidOutput[m_hidPerceptrons][m_outPerceptrons];
    for(int i=0; i<m_hidPerceptrons; ++i)
    {
        hidOutput[i][0] = sigmoid(1 * m_hidWeights[0][i] + pIn[0] * m_hidWeights[1][i] + pIn[1] * m_hidWeights[2][i]);
    }

    float output[m_outPerceptrons];
    for(int i=0; i<m_outPerceptrons; ++i)
    {
        output[i] = sigmoid( 1 * m_outWeights[0][i] + hidOutput[0][i] * m_outWeights[1][i] + hidOutput[1][i] * m_outWeights[2][i] );
    }

    return output[0];
}
