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

    uint8_t* getData() { return m_data; }
    uint32_t* getDimensions() { return m_dimension; }
protected:
private:
    uint32_t m_magicNumber;

    uint8_t m_dimensionNumber;
    uint32_t* m_dimension;

    uint8_t* m_data;
};

#endif // IDXFILE_H
