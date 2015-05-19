#ifndef TDATA_H_INCLUDED
#define TDATA_H_INCLUDED

enum TrainingType
{
    invalid,
    parallel,
    sequential
};

class TrainingData
{
public:
    TrainingData();
    ~TrainingData();

    TrainingType getType(){return m_type;}
    TrainingType setType(TrainingType pType){m_type = pType;}
protected:
    TrainingType m_type;
};

#endif // TDATA_H_INCLUDED
