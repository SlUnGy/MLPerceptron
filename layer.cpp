#include "layer.h"

#include <ctime>

#include <random>

Layer::Layer()
    :m_in{0},m_width{0},m_weights{nullptr}
{

}

Layer::Layer(const unsigned int pIn, const unsigned int pWidth):
    m_in {pIn},m_width {pWidth},m_weights {new float*[m_in]}
{
    for(unsigned int i=0; i<m_in; ++i)
    {
        m_weights[i] = new float[m_width];
    }
}

Layer::~Layer()
{
    for(unsigned int i=0; i<m_in; ++i)
    {
        delete [] m_weights[i];
    }
    delete [] m_weights;
    m_weights = nullptr;
}

void Layer::setupWeights(const unsigned int pIn, const unsigned int pWidth)
{
    if(m_in==0 && m_width == 0 && m_weights == nullptr)
    {
        m_in        = pIn;
        m_width     = pWidth;
        m_weights   = new float*[m_in];
        for(unsigned int i=0; i<m_in; ++i)
        {
            m_weights[i] = new float[m_width];
        }
    }
}

void Layer::randomizeWeights()
{
    std::mt19937 mt(time(NULL));
	std::uniform_real_distribution<> distribution(-1, 1);

    for (unsigned int i=0;i<m_in;i++)
    {
		for (unsigned int j=0;j<m_width;j++) {
			m_weights[i][j] = distribution(mt);
		}
	}
}
