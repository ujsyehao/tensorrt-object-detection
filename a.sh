rm ./tensorrt-ssd
rm -rf build
mkdir build
cd build 
cmake ..
make -j8 
cd ..
