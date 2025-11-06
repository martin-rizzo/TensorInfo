#include <tin/tensormap.h>



int main( int argc, char *argv[] )  {
    tin::TensorMap tensorMap;

    // el primer argumento debe ser el nombre de archivo
    if ( argc < 2 ) {
        std::cerr << "Usage: test <tensorfile>" << std::endl;
        return -1;
    }
    tensorMap.load( argv[1] );
    for( auto key: tensorMap.keys() ) {
        std::cout << "key: " << key << " [" << tensorMap[key].shape() << "]" << std::endl;
    }

    //for( auto it = map.begin(); it != map.end(); ++it ) {
    //    std::cout << "key: " << it->first << " [" << it->second.shape() << "]" << std::endl;
    //}


    return 0;
}
