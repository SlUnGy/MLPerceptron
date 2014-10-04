#ifndef MLP_H_INCLUDED
#define MLP_H_INCLUDED

#include <cmath>

class MLP {
	public:
		MLP();
		~MLP() {};

		void train(const float, const float, const float);
		float run(const float, const float);

		static float sigmoid( const float pNum) {
            return (1.0f/(1.0f+expf(-pNum)));// The sigmoid function.
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

#endif // MLP_H_INCLUDED
