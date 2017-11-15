#include <ctype.h>
#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "_slicAlgorithm.hpp"

using namespace cv;
using namespace cv::slicNora;
using namespace std;

static const char *window_name = "Slic Superpixels";

static const char *keys =
    "{h help      | | help menu}"
    "{c camera    |0| camera id}"
    "{i image     | | image file}"
    "{v video     | | video file}"
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
    String imgFile = cmd.get<String>("image");
    String vidFile = cmd.get<String>("video");
    int algorithmy = cmd.get<int>("algorithm");
    int region_size = 30;
    int ruler = 10;
    int min_element_size = region_size;
    int num_iterations = 3;
    bool noImage = imgFile.empty();
    bool useCam = noImage && vidFile.empty();

    VideoCapture cap;
    Mat input_image;

    if (useCam)
    {
        if (!cap.open(capture))
        {
            cout << "Could not initialize capturing..." << capture << "\n";
            return -1;
        }
    }
    // vid file
    else if (noImage)
    {
        if (!cap.open(vidFile))
        {
            cout << "cannot open file: \"" << vidFile << "\" ." << endl;
            return -1;
        }
    }

    else
    {
        input_image = imread(imgFile);
        if (input_image.empty())
        {

            cout << "Could not open image..." << imgFile << "\n";
            return -1;
        }
    }

    namedWindow(window_name, 0);
    // createTrackbar("Algorithm", window_name, &algorithmy, 2, 0);
    // createTrackbar("Region size", window_name, &region_size, 200, 0);
    // createTrackbar("Ruler", window_name, &ruler, 100, 0);
    // createTrackbar("Connectivity", window_name, &min_element_size, 100, 0);
    // createTrackbar("Iterations", window_name, &num_iterations, 12, 0);

    Mat result, mask;
    int display_mode = 2;

    for (;;)
    {
        Mat frame;
        if (!noImage)
            input_image.copyTo(frame);
        else
            cap >> frame;

        if (frame.empty())
            break;

        result = frame;
        Mat converted, blurred;
        medianBlur(result, blurred , 5);
       
        //cvtColor(frame, converted, COLOR_BGR2HSV);/*HSV*/
        //cvtColor(frame, converted, COLOR_BGR2Lab);/*LAB*/
        //cvtColor(frame, converted, COLOR_BGR2YUV );
        
        double t = (double)getTickCount();

        Ptr<SuperpixelSlic> slic = 
        createSuperpixelSlic(blurred, algorithmy + Slic, region_size, float(ruler));
        
        slic->iterate(num_iterations);
        // if (min_element_size > 0)
        //     slic->enforceLabelConnectivity(min_element_size);

        t = ((double)getTickCount() - t) / getTickFrequency();
        cout << "Slic" << (algorithmy ? 'O' : ' ')
             << " segmentation took " << (int)(t * 1000)
             << " ms with " << slic->getNumberOfSuperpixels() << " superpixels" << endl;

        // get the contours for displaying
        slic->getLabelContourMask(mask, true);
        result.setTo(Scalar(0, 0, 255), mask);

        // display output
        switch (display_mode)
        {
        case 0: //superpixel contours
            imshow(window_name, result);
            
            break;
        case 1: //mask
            imshow(window_name, mask);
            break;
        case 2: //labels array
        {
            // use the last x bit to determine the color. Note that this does not
            // guarantee that 2 neighboring superpixels have different colors.
            // retrieve the segmentation result
            Mat labels, uniforms;
            // for(int i = 0; i<result.rows; i++){
            //     for(int j = 0; j<result.cols; j++){
            //         int l = result.at<Vec3b>(i,j)[0]?1:0;
            //         cout<<l<<", ";
            //         l = result.at<Vec3b>(i,j)[1]?1:0;
            //         cout<<l<<", ";
            //         l = result.at<Vec3b>(i,j)[2]?1:0;
            //         cout<<l<<"\t\t";
            //     }
            //     cout<<endl;
            // }
            slic->getUniforms(uniforms);
            slic->getLabels(labels);
            //
            //cout<<labels<<endl;

            // for(int i = 0; i<result.rows; i++){
            //     for(int j = 0; j<result.cols; j++){
            //         int l = labels.at<int>(i,j);
            //         cout<<"   ";
                    
            //         cout<<"   ";
                    
            //         cout<<l<<"\t\t";
            //     }
            //     cout<<endl;
            // }

            const int num_label_bits = 2;
            labels &= (1 << num_label_bits) - 1;
            labels *= 1 << (16 - num_label_bits);
            imshow("r", result);
            imshow("uniform", uniforms);
            imshow("blurred", blurred);
            
            break;
        }
        }

        int c = waitKey(1) & 0xff;
        if (c == 'q' || c == 'Q' || c == 27)
            break;
        else if (c == ' ')
            display_mode = (display_mode + 1) % 3;
    }

    return 0;
}