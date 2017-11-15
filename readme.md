# SLIC Superpixels
This is a simple modificaiton for opencv_contrib library. Centroids are being shown after smoothing the images (frames). 

## Project directory: 
Files start with underscore are the modified library files:
```
├── CMakeFiles.txt
├── _private.hpp
├── _slicPrecomp.hpp
├── _slicAlgorithm.hpp
├── _slicAlgorithm.cpp
├── slicEx.cpp
├─── build 
│   ├── slic
│   ├── ..
│   ├── ..
├─── data
│   ├── image1.jpg
│   ├── image2.jpg
│   ├── ..
├── readme.md

```
## Build and Run 
Within the top level directory:
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
    ./slic -i=../data/image1.jpg
    ./slic -v=../data/video.mp4
    ./slic
```
If no images provided, the webcam (if available ) will be turned on, and start capturing and processing the frames.

## Problems
I had to add the file 'private.hpp' into the repository.
It is being used for `Split` and `BlockedRange` types.
