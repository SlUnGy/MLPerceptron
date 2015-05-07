#ifndef IDXFILE_H
#define IDXFILE_H

#include <string>

class IDXFile
{
public:
    IDXFile( const std::string& );
    IDXFile( const IDXFile& );
    virtual ~IDXFile();

    uint32_t getMagicNumber() const { return m_magicNumber; }
    uint8_t getDimensionNumber() const { return m_dimensionNumber; }
    bool hasError() const { return m_error; }

    uint8_t* getDataPointer() const { return m_data; }
    uint32_t* getDimensions() const { return m_dimension; }

    unsigned int getTotalSize() const { return m_totalSize; }

    bool readFile(const std::string& );

    void deleteData()
    {
        if(m_data != nullptr)
        {
            delete [] m_data;
            m_data = nullptr;
        }
        if(m_dimension != nullptr)
        {
            delete [] m_dimension;
            m_dimension = nullptr;
        }
        m_totalSize = 0;
    }

    IDXFile& operator= ( const IDXFile &pCpy )
    {
        if(this != &pCpy)
        {
            deleteData();
            m_error             = pCpy.hasError();
            m_magicNumber       = pCpy.getMagicNumber();
            m_dimensionNumber   = pCpy.getDimensionNumber();
            m_dimension         = pCpy.getDimensions();
            m_data              = pCpy.getDataPointer();
        }
        return *this;
    }
protected:
    bool m_error;
private:
    uint32_t m_magicNumber;

    uint8_t m_dimensionNumber;
    uint32_t* m_dimension;

    uint8_t* m_data;
    unsigned int m_totalSize;
};

#endif // IDXFILE_H
