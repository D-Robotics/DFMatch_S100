#!/bin/bash
clear
cd $(dirname "$0")
echo "=> curr dir: $(pwd)"

rm -rf build
rm -rf result

mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)

echo "=> ================="
./DFMatch_S100 ../image_test/ ../result/