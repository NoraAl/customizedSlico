#include <ctype.h>
#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "_slicAlgorithm.hpp"

using namespace cv;
using namespace cv::slicNora;
using namespace std;

struct Info{
    string filename;
    int label;
};
vector <Info> files;

static void readCsv(const string& filename, char separator = ',') {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(Error::StsBadArg, error_message);
    }
    Info element;
    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
            element.filename = path;
            element.label = atoi(classlabel.c_str());
            files.push_back(element); 
        }
    }
}

static const char *window_name = "Slic Superpixels";

static const char *keys =
    "{h help      | | help menu}"
    "{c camera    |0| camera id}"
    "{f file     | | file name}"
    "{a algorithm |1| Slic(0),Slico(1),MSlic(2)}";

int main(int argc, char **argv)
{
    CommandLineParser cmd(argc, argv, keys);
    if (cmd.has("help"))
    {
        cmd.about("This program demonstrates Slic superpixels using OpenCV class SuperpixelSlic.\n"
                  "If no image file is supplied, try to open a webcam.\n"
                  "Use [space] to toggle output mode, ['q' or 'Q' or 'esc'] to exit.\n");
        cmd.printMessage();
        return 0;
    }
    int capture = cmd.get<int>("camera");
    string imagesFile = cmd.get<String>("file");
    
    int algorithmy = cmd.get<int>("algorithm");
    int region_size = 8;
    int ruler = 8;
    int min_element_size = region_size;
    int num_iterations = 3;
    bool useCam = imagesFile.empty();

    VideoCapture cap;
    Mat input_image;
    int label;
    

    if (useCam)
    {
        cout << "this file doesn't open cam."<<endl;
        return 0;
    }
    // Mat of images
    else
    {
        try
        {
            // array of files names
            readCsv(imagesFile);
        }
        catch (cv::Exception &e)
        {
            cerr << "Error opening file \"" << imagesFile << "\". Reason: " << e.msg << endl;
            // nothing more we can do
            exit(1);
        }
    }

    namedWindow(window_name, 0);
    Mat result;
    int display_mode = 2;
    for (int i = 0 ; i < files.size(); i++)
    {
        Mat frame;
        string resultName = "../results/image" + to_string(i)+".jpg";

        
        
        label = files[i].label;
        input_image = imread(files[i].filename);
        cout << "reading frame file"<<endl;
        input_image.copyTo(frame);
        

        
        Mat converted, blurred;
        medianBlur(frame, blurred , 5);
        Ptr<SuperpixelSlic> slic = 
        createSuperpixelSlic(blurred, algorithmy + Slic, region_size, float(ruler));
         
        slic->iterate(num_iterations);

        cout << "Slic" << (algorithmy ? 'O' : ' ')
             << " "<< slic->getNumberOfSuperpixels() << " superpixels" << endl;

        Mat labels, uniforms;

        slic->getUniforms(uniforms);
        slic->getLabels(labels);
        slic->saveCentroids(label);
        
        imshow("uniform", uniforms);
        imwrite(resultName, uniforms );
        //imshow("blurred", blurred);

        int c = waitKey(1) & 0xff;
        
        if (c == 'q' || c == 'Q' || c == 27 )
            break;
        else if (c == ' ')
            display_mode = (display_mode + 1) % 3;
          
    }

    return 0;
}