#include "layer.h"


Layer::Layer(const int pIn, const int pWidth):
    m_in {pIn},m_width {pWidth},m_weights {new float*[m_in]}
{
    for(int i=0; i<m_in; ++i)
    {
        m_weights[i] = new float[m_width];
    }
}

Layer::~Layer()
{
    for(int i=0; i<m_in; ++i)
    {
        delete [] m_weights[i];
    }
    delete [] m_weights;
    m_weights = nullptr;
}
