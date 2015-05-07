#ifndef OCLP_H_INCLUDED
#define OCLP_H_INCLUDED

#include <string>
#include <vector>

//the amount of opencl header warnings is too damn high
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Weffc++"
#pragma GCC diagnostic ignored "-pedantic"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#pragma GCC diagnostic pop

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
    void testAll(float*);

    void randomizeWeights();
protected:
    bool m_foundDevice;
    const std::string m_sourceFile;

    const float m_eta;

    const int m_inpPerceptrons;
    const int m_hidPerceptrons;
    const int m_outPerceptrons;

    const int m_ndGlobal;

    std::vector<float> m_hidWeights;
    std::vector<float> m_outWeights;

    int m_trainingDataSets;
    int m_testDataSets;
private:
    std::vector<cl::Device> m_device;
    cl::Context m_context;
    cl::Program m_program;

    cl::Kernel m_calcLayerOutput, m_calcLayerDelta, m_calcOutputDelta, m_applyDelta;

    cl::Buffer m_bTrImg, m_bTrClf;
    cl::Buffer m_bTeImg;
    cl::Buffer m_bHOut, m_bOOut;
    cl::Buffer m_bHWeights, m_bOWeights;
    cl::Buffer m_bHDelta, m_bODelta;
};

#endif // OCLP_H_INCLUDED
