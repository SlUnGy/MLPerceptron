#ifndef TDATA_H_INCLUDED
#define TDATA_H_INCLUDED

#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Weffc++"
#pragma GCC diagnostic ignored "-pedantic"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#pragma GCC diagnostic pop

enum TrainingType
{
    invalid,
    parallel,
    sequential
};

enum PreprocessingType
{
    none,
    distortions
};

class TrainingEnvironment
{
public:
    TrainingEnvironment();
    ~TrainingEnvironment();

    TrainingType getType(){return m_type;}
    void setType(TrainingType pType){m_type = pType;}

    std::vector<float>* getTrainingData(){ return m_trDat; }
    void setTrainingData( std::vector<float> *pTrDat ){ m_trDat = pTrDat; }

    std::vector<float>* getTrainingClassifications(){ return m_trCls; }
    void setTrainingClassifications( std::vector<float> *pTrCls ){ m_trCls = pTrCls; }

    std::vector<float>* getTestingData(){ return m_teDat; }
    void setTestingData( std::vector<float> *pTeDat ){ m_teDat = pTeDat; }

    std::vector<int>* getTestingClassifications(){ return m_teCls; }
    void setTestingClassifications( std::vector<int> *pTeCls ){ m_teCls = pTeCls; }

    unsigned int getInputSampleWidth(){ return m_inputWidth; }
    void setInputSampleWidth( const unsigned int pInputWidth ) { m_inputWidth = pInputWidth; }

    unsigned int getOutputSampleWidth(){ return m_outputWidth; }
    void setOutputSampleWidth( const unsigned int pOutputWidth ) { m_outputWidth = pOutputWidth; }

    bool isValid();
    int executeOCR();
    bool initOpenCLEnvironment();

    bool hasOpenCLContext(){ return m_opencl && m_context != nullptr && m_device != nullptr; }
protected:
    TrainingType m_type;

    std::vector<float> *m_trDat;
    std::vector<float> *m_trCls;
    std::vector<float> *m_teDat;
    std::vector<int>   *m_teCls;

    unsigned int m_inputWidth;
    unsigned int m_outputWidth;

    std::vector<cl::Device> *m_device;
    cl::Context *m_context;
private:
    bool m_opencl;
};

#endif // TDATA_H_INCLUDED
