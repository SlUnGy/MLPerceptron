#ifndef NLP_H_INCLUDED
#define NLP_H_INCLUDED

#include <cmath>
#include <cstdint>
#include <iostream>

#include "layer.h"

class NLayerPerceptron {
	public:
		NLayerPerceptron(const float, const unsigned int, const unsigned int, const unsigned int*, const unsigned int);

		~NLayerPerceptron()
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

#endif // NLP_H_INCLUDED
