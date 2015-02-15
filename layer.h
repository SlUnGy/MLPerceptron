#ifndef LAYER_H_INCLUDED
#define LAYER_H_INCLUDED

class Layer
{
    public:
        Layer();
        Layer(const unsigned int, const unsigned int);
        ~Layer();

        setupWeights(const unsigned int, const unsigned int);
        randomizeWeights();

        const unsigned int m_in;
        const unsigned int m_width;
        float **m_weights;
};

#endif // LAYER_H_INCLUDED
