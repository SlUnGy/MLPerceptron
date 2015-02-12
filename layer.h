#ifndef LAYER_H_INCLUDED
#define LAYER_H_INCLUDED

class Layer
{
    public:
        Layer(const int, const int);
        ~Layer();

        const int m_in;
        const int m_width;
        float **m_weights;
};

#endif // LAYER_H_INCLUDED
