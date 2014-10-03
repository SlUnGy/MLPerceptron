/* ========================================== *
 * Filename:	bpnet.h                       *
 * Author:		James Matthews.               *
 *											  *
 * Description:								  *
 * This is a tiny neural network that uses	  *
 * back propagation for weight adjustment.	  *
 * ========================================== */

#include <cmath>
#include <cstdlib>
#include <ctime>

#define BP_LEARNING	(float)(0.5)	// The learning coefficient.

class CBPNet {
	public:
		CBPNet();
		~CBPNet() {};

		float Train(float, float, float);
		float Run(float, float);

	private:
		float m_fWeights[3][3];		// Weights for the 3 neurons.

		float Sigmoid(float);		// The sigmoid function.
};

CBPNet::CBPNet() {
	srand((unsigned)(time(NULL)));

	for (int i=0;i<3;i++) {
		for (int j=0;j<3;j++) {
			// For some reason, the Microsoft rand() function
			// generates a random integer. So, I divide by the
			// number by MAXINT/2, to get a num between 0 and 2,
			// the subtract one to get a num between -1 and 1.
			m_fWeights[i][j] = (float)(rand())/(32767/2) - 1;
		}
	}
}

float CBPNet::Train(float i1, float i2, float d) {
	// These are all the main variables used in the
	// routine. Seems easier to group them all here.
	float net1, net2, i3, i4, out;

	// Calculate the net values for the hidden layer neurons.
	net1 = 1 * m_fWeights[0][0] + i1 * m_fWeights[1][0] +
		  i2 * m_fWeights[2][0];
	net2 = 1 * m_fWeights[0][1] + i1 * m_fWeights[1][1] +
		  i2 * m_fWeights[2][1];

	// Use the hardlimiter function - the Sigmoid.
	i3 = Sigmoid(net1);
	i4 = Sigmoid(net2);

	// Now, calculate the net for the final output layer.
	net1 = 1 * m_fWeights[0][2] + i3 * m_fWeights[1][2] +
	   	  i4 * m_fWeights[2][2];
	out = Sigmoid(net1);

	// We have to calculate the deltas for the two layers.
	// Remember, we have to calculate the errors backwards
	// from the output layer to the hidden layer (thus the
	// name 'BACK-propagation').
	float deltas[3];

	deltas[2] = out*(1-out)*(d-out);
	deltas[1] = i4*(1-i4)*(m_fWeights[2][2])*(deltas[2]);
	deltas[0] = i3*(1-i3)*(m_fWeights[1][2])*(deltas[2]);

	// Now, alter the weights accordingly.
	float v1 = i1, v2 = i2;
	for(int i=0;i<3;i++) {
		// Change the values for the output layer, if necessary.
		if (i == 2) {
			v1 = i3;
			v2 = i4;
		}

		m_fWeights[0][i] += BP_LEARNING*1*deltas[i];
		m_fWeights[1][i] += BP_LEARNING*v1*deltas[i];
		m_fWeights[2][i] += BP_LEARNING*v2*deltas[i];
	}

	return out;
}

float CBPNet::Sigmoid(float num) {
	return (float)(1/(1+exp(-num)));
}

float CBPNet::Run(float i1, float i2) {
	// I just copied and pasted the code from the Train() function,
	// so see there for the necessary documentation.

	float net1, net2, i3, i4;

	net1 = 1 * m_fWeights[0][0] + i1 * m_fWeights[1][0] +
		  i2 * m_fWeights[2][0];
	net2 = 1 * m_fWeights[0][1] + i1 * m_fWeights[1][1] +
		  i2 * m_fWeights[2][1];

	i3 = Sigmoid(net1);
	i4 = Sigmoid(net2);

	net1 = 1 * m_fWeights[0][2] + i3 * m_fWeights[1][2] +
	   	  i4 * m_fWeights[2][2];
	return Sigmoid(net1);
}

