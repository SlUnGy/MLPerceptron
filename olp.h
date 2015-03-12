#ifndef OLP_H_INCLUDED
#define OLP_H_INCLUDED

#include <cmath>
#include <cstdint>
#include <iostream>

class OneLayerPerceptron {
	public:
		OneLayerPerceptron():OneLayerPerceptron(0.25f, 2, 3, 1){}
		OneLayerPerceptron(const float, const int, const int, const int);

		~OneLayerPerceptron()
        {
            for(int i=0; i<m_hidPerceptrons; ++i)
            {
                delete [] m_hidWeights[i];
            }
            delete [] m_hidWeights;


            for(int i=0; i<m_outPerceptrons; ++i)
            {
                delete [] m_outWeights[i];
            }
            delete [] m_outWeights;
        }


		void train(const float*, const float*);
		float* classify(const float*);

        static float sigmoid( const float pNum )
        {
            return (1.0f/(1.0f+expf(-pNum)));
        }

        bool writeToFile(const std::string&);

        void randomizeWeights();

	protected:
		const float m_eta;

		const int m_hidPerceptrons;
		const int m_inpPerceptrons;
		const int m_outPerceptrons;

		float **m_hidWeights;
		float **m_outWeights;
    private:
};

#endif // OLP_H_INCLUDED
