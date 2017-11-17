# SLIC Superpixels
This is a simple modificaiton for opencv_contrib library. Centroids are being shown after smoothing the images (frames). 

## Project directory: 
Files start with underscore are the modified library files. Directory images is not included and it can be any images database. You can add AT&T database from [here](http://www.cl.cam.ac.uk/Research/DTG/attarchive/pub/data/att_faces.zip):
```bash
├── CMakeFiles.txt
├── _private.hpp
├── _slicPrecomp.hpp
├── _slicAlgorithm.hpp
├── _slicAlgorithm.cpp
├── slicEx.cpp
├── slicEx.cpp
├─── build 
│   ├── slic
│   ├── slicfile
│   ├── ..
├─── data
│   ├── file1.ext
│   ├── file2.ext
│   ├── ..
├─── images
│   ├─── person1
│   │   ├── image1.pgm
│   │   ├── image2.pgm
│   │   ├── ...
│   ├─── person2
│   │   ├── image1.pgm
│   │   ├── image2.pgm
│   │   ├── ...
│   ├─── ...
├── readme.md

```
<hr>

## Build and Run 
Within the top level directory:
``` bash
    mkdir build
    cd build
```
Then:
```bash
    cmake ..
    make
```
If everything works fine, the executable slic is generated, which can be run using:  
```bash
    ./slic -i=../data/image1.jpg
    ./slic -v=../data/video.mp4
    ./slic
```
If no images provided, the webcam (if available ) will be turned on, and start capturing and processing the frames.

## Build and Run with AT&T face database
Same as above, but run with:
```bash
    ./slicfile -f=../images.csv
```
The output csv files will be saved into data directory. In case new database added, you can run the provided python script to prepare new images.csv file. Script can be run from build directory:
```bash
    python ../createCsv.py ../images
```
<hr>

## Problems
I had to add the file 'private.hpp' into the repository.
It is being used for `Split` and `BlockedRange` types.
<hr>

## Credit
Please refer to the copyrights in the below documentations:<br>
<ul>
<li>Library being used or modified is OpenCV. <a href="https://opencv.org">Opencv.org</a>. <a href="https://github.com/opencv">Opencv and Opencv_contrib</a> repositories. </li>

<li>Face database is the <a href="http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html">AT&T</a> database.</li>
</ul>