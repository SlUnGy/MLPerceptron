#include <iostream>

#include "mlp.h"

const unsigned long iterations = 50000;

int main()
{
    MLP bp;

    const float p00 [] = {0.0f, 0.0f};
    const float p01 [] = {0.0f, 1.0f};
    const float p10 [] = {1.0f, 0.0f};
    const float p11 [] = {1.0f, 1.0f};

    for ( unsigned  long i = 0; i < iterations; ++i )
    {

        bp.train( p00, 0 );
        bp.train( p01, 1 );
        bp.train( p10, 1 );
        bp.train( p11, 0 );
    }

    std::cout << "0,0 = " << bp.run( 0, 0 ) << std::endl;
    std::cout << "0,1 = " << bp.run( 0, 1 ) << std::endl;
    std::cout << "1,0 = " << bp.run( 1, 0 ) << std::endl;
    std::cout << "1,1 = " << bp.run( 1, 1 ) << std::endl;

    return 0;
}
