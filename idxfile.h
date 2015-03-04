#ifndef IDXFILE_H
#define IDXFILE_H

#include <string>

class IDXFile
{
public:
    IDXFile( const std::string& );
    virtual ~IDXFile();

    uint32_t getMagicNumber() { return m_magicNumber; }
    unsigned int getDimensionNumber() { return m_dimensionNumber; }
    bool hasError() { return m_error; }

    uint8_t* getDataPointer() { return m_data; }
    uint32_t* getDimensions() { return m_dimension; }

    bool readFile(const std::string& );
    void deleteData(){ if(m_data != nullptr){delete [] m_data; m_data = nullptr;}}
protected:
    bool m_error;
private:
    uint32_t m_magicNumber;

    uint8_t m_dimensionNumber;
    uint32_t* m_dimension;

    uint8_t* m_data;
};

#endif // IDXFILE_H
