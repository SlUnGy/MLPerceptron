#ifndef OCLP_H_INCLUDED
#define OCLP_H_INCLUDED

#include <string>
#include <vector>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

class OpenCLPerceptron
{
public:
    OpenCLPerceptron():OpenCLPerceptron(0.25f, 2, 3, 1){}
	OpenCLPerceptron(const float, const int, const int, const int);
    ~OpenCLPerceptron();

    bool hasFoundDevice(){return m_foundDevice;}

    bool initOpenCL();
    bool initTraining(std::vector<float>*, std::vector<float>*, std::vector<float>*);

    void trainAll();
    //will only do classification
    std::vector<float>* testAll();

    void randomizeWeights();
protected:
    bool m_foundDevice;
    const std::string m_sourceFile;

    const float m_eta;

    const int m_inpPerceptrons;
    const int m_hidPerceptrons;
    const int m_outPerceptrons;

    std::vector<float> m_outputBuffer;
    std::vector<float> m_hidWeights;
    std::vector<float> m_outWeights;
private:
    std::vector<cl::Device> m_device;
    cl::Context m_context;
    cl::Program m_program;

    cl::Kernel m_calcLayerOutput, m_calcLayerDelta, m_calcOutputDelta, m_applyDelta;

    cl::Buffer m_bTrImg;
    cl::Buffer m_bTrClf;
    cl::Buffer m_bHOut, m_bOOut;
    cl::Buffer m_bHWeights, m_bOWeights;
};

#endif // OCLP_H_INCLUDED
