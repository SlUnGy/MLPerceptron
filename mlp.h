#ifndef MLP_H_INCLUDED
#define MLP_H_INCLUDED

#include <cmath>
#include <cstdint>
#include <iostream>

#include "layer.h"

class MultilayerPerceptron {
	public:
		MultilayerPerceptron(const float, const unsigned int, const unsigned int, const unsigned int*, const unsigned int);

		~MultilayerPerceptron()
        {
            if(m_hiddenLayers)
            {
                delete [] m_hiddenLayers;
            }
            if(m_outputLayer)
            {
                delete m_outputLayer;
            }
        }

		void train(const float*, const float*);
		float* classify(const float*);

        template<typename T> static float sigmoid(const T pNum)
        {
            return (1.0f/(1.0f+exp(-pNum)));
        }

        static float sigmoid( const float pNum )
        {
            return (1.0f/(1.0f+expf(-pNum)));
        }

        bool writeToFile(const std::string&);
	protected:
		const float m_eta;
        const unsigned int m_maxHiddenLayer;

		Layer *m_hiddenLayers;
		Layer *m_outputLayer;
    private:
};

#endif // MLP_H_INCLUDED
