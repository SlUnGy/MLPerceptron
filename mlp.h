#ifndef MLP_H_INCLUDED
#define MLP_H_INCLUDED

#include <cmath>
#include <cstdint>
#include <iostream>

class MLP {
	public:
		MLP():MLP(0.25f, 2, 3, 1){}
		MLP(const float, const int, const int, const int);
		~MLP()
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


		template<typename T> void train(const T*, const T*);
		template<typename T> T* run(const T*);

        template<typename T> static float sigmoid(const T pNum)
        {
            return (1.0f/(1.0f+exp(-pNum)));
        }

        static float sigmoid( const float pNum )
        {
            return (1.0f/(1.0f+expf(-pNum)));
        }


	protected:
		const float m_eta;

		const int m_hidPerceptrons;
		const int m_inpPerceptrons;
		const int m_outPerceptrons;

		float **m_hidWeights;
		float **m_outWeights;
    private:
};

template<typename T> void MLP::train(const T* pIn,const T* pTarget)
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

template<typename T> T* MLP::run(const T *pIn)
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

    T* output = new T[m_outPerceptrons]();
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

#endif // MLP_H_INCLUDED
