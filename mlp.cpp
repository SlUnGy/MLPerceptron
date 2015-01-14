#include "mlp.h"

#include <ctime>

#include <random>

MultilayerPerceptron::MultilayerPerceptron(const float pEta, const int pInputPerceptrons, const int pHiddenPerceptrons, const int pOutputPerceptrons)
    : m_eta{pEta}, m_hidPerceptrons{pHiddenPerceptrons}, m_inpPerceptrons{pInputPerceptrons}, m_outPerceptrons{pOutputPerceptrons}
{
    randomizeWeights();
}

void MultilayerPerceptron::randomizeWeights()
{
	/*
        doesn't work on mingw&windows....
        std::random_device rd;
    */
	std::mt19937 mt(time(NULL));
	std::uniform_real_distribution<> distribution(-1, 1);

    //+1 due to constant coefficient, i.e. bias
	m_hidWeights = new float*[m_inpPerceptrons+1];
    for (int i=0;i<m_inpPerceptrons+1;i++)
    {
        m_hidWeights[i] = new float[m_hidPerceptrons];

		for (int j=0;j<m_hidPerceptrons;j++) {
			m_hidWeights[i][j] = distribution(mt);
		}
	}

	m_outWeights = new float*[m_hidPerceptrons+1];
    for (int i=0;i<m_hidPerceptrons+1;i++)
    {
        m_outWeights[i] = new float[m_outPerceptrons];

		for (int j=0;j<m_outPerceptrons;j++) {
			m_outWeights[i][j] = distribution(mt);
		}
	}
}

bool MultilayerPerceptron::writeToFile(const std::string& pFilename)
{
    return true;
}
