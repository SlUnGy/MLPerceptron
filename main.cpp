#include <iostream>

#include "mlp.h"

const unsigned long iterations = 20000;

int main()
{
    MLP bp;


    for ( unsigned  long i = 0; i < iterations; ++i )
    {
        bp.train( 0, 0, 0 );
        bp.train( 0, 1, 1 );
        bp.train( 1, 0, 1 );
        bp.train( 1, 1, 0 );
    }

    std::cout << "0,0 = " << bp.run( 0, 0 ) << std::endl;
    std::cout << "0,1 = " << bp.run( 0, 1 ) << std::endl;
    std::cout << "1,0 = " << bp.run( 1, 0 ) << std::endl;
    std::cout << "1,1 = " << bp.run( 1, 1 ) << std::endl;

    return 0;
}
