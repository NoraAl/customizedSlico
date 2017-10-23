## SLIC Superpixels
Project directory: 
```sh
|   CMakeLists.txt
|   slicEx.cpp
|   readme.md
+---data
|   |   image1.jpg
|   |   image2.jpg
|   |   ..
|   |   imageN.jpg

```
Within the directory:
```
    mkdir build
    cd build
```
Then:
```
    cmake ..
    make
```
If everything works fine, the executable slic is generated, which can be run using:  
```
    ./slicEx -i=../image1.jpg
```
If no images provided, the webcam (if available ) will turn on, and start capturing.

## Problems
I had to copy the file inernal.hpp into 'opencv2/core' directory.
It is being used for types Split and BlockedRange.
