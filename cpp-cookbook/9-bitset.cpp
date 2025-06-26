#include <bitset>
#include <iostream>

int main() {

    std::bitset<8> bits(42);

    std::cout << "Initial bitset: " << bits << std::endl;

    std::cout << "Bit at position 1: " << bits[1] << std::endl;
    std::cout << "Bit at position 3: " << bits[3] << std::endl;

    bits.set(0);
    bits.reset(1);

    std::cout << "Modified bitset: " << bits << std::endl;

    bits.flip();

    std::cout << "After flip: " << bits << std::endl;

    std::cout << "Number of set bits: " << bits.count() << std::endl;

    std::cout << "Any set? " << std::boolalpha << bits.any() << std::endl;
    std::cout << "None set? " << bits.none() << std::endl;
    std::cout << "All set? " << bits.all() << std::endl;

    return 0;
}

/*
Initial bitset: 00101010
Bit at position 1: 1
Bit at position 3: 1
Modified bitset: 00101001
After flip: 11010110
Number of set bits: 5
Any set? true
None set? false
All set? false
*/
