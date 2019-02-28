rm -rf build
mkdir build
cd build 
reset 
cmake ..
make -j8 
cd ..
./tensorrt-ssd
