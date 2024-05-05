#include <iostream>
#include "array.h"

using namespace std;

int main(int argc, char **argv) {
    Array2D array(2, {100, 10}, 100);
    cout << "array[99][9]: " << array[99][9] << endl;
    array(99, 9) = 999;
    cout << "array[99][9]: " << array(99, 9) << endl;
    // this should be error
    // cout << "array[100][9]: " << array(100, 9) << endl;
    // cout << "array[99][10]: " << array(99, 10) << endl;

    Array3D a(3, {1000, 100, 10});
    cout << "array[990][90][9]: " << a(990, 90, 9) << endl;
    a(990, 90, 9) = 999;
    cout << "array[990][90][9]: " << a(990, 90, 9) << endl;
    
    return 0;
}