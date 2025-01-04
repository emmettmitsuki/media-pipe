#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core.hpp>
#include <opencv2/cudawarping.hpp> 
#include <opencv2/imgproc.hpp>     
#include <opencv2/highgui.hpp>
#include <opencv2/cudaimgproc.hpp> 
#include <iostream>
#include <filesystem>

using namespace cv;
using namespace std;
using namespace dnn;

/*
To compile, enter the build directory and run make. 
*/
int main() {
    cerr << "Attempting to open webcam." << endl;
    cerr << getBuildInformation() << endl;
    
    // Used the V4L2 Linux driver for my webcam to detect. 
    VideoCapture cap(0, cv::CAP_V4L2);
    if (!cap.isOpened()) {
        cerr << "Error: Failed to open webcam." << endl;
        return -1;
    }

    /*
    OpenCV defaults to a resolution of 640x480, but for me increasing resolution
    to 1080p makes the fps go to like 3-5 instead of 30
    */ 
    int webcamWidth = 640, webcamHeight= 480;
    cap.set(CAP_PROP_FRAME_WIDTH, webcamWidth); 
    cap.set(CAP_PROP_FRAME_HEIGHT, webcamHeight);
    cap.set(CAP_PROP_FPS, 30);
    // cap.set(CAP_PROP_AUTO_EXPOSURE, 0.01);

    /*
    Default camera output format is YUYV, which requires more bandwidth. Changing it to 
    MJPG can increase performance and in my case, gets it to actually work
    */
    cap.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M', 'J', 'P', 'G'));

    cerr << "Webcam opened successfully!" << endl;
    
    // Define location of the models. We are using OpenPose's MPI model.
    string currPath = filesystem::current_path().string();
    string protoFile = currPath + "/../models/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt";
    string weightsFile = currPath + "/../models/pose/mpi/pose_iter_160000.caffemodel";

    // Load the MPI network into memory. 
    Net net = readNetFromCaffe(protoFile, weightsFile);

    if (net.empty()) {
        cerr << "Error: Failed to load model." << endl;
        return -1;
    }

    // Enable GPU acceleration
    // net.setPreferableBackend(DNN_BACKEND_CUDA);
    // net.setPreferableTarget(DNN_TARGET_CUDA_FP16);

    Mat frame;

    while (true) {
        cerr << "Loading image" << endl;
        cap.read(frame);

        if (frame.empty()) {
            cerr << "Error: Failed to capture image." << endl;  
            break;
        }

        // Prepare the frame and feed it to the network.
        Mat inpBlob = blobFromImage(frame, 1.0 /255, Size(webcamWidth / 4, webcamHeight / 4),
        Scalar(0, 0, 0), false, false);
        net.setInput(inpBlob);

        /* Get the output, which is a 4D matrix.
        1. The image ID 
        2. Index of a key point
        3. Height of the output map
        4. Width of the output map
        */
        Mat output = net.forward(); // this line makes the webcam run really slow

        // Extract the keypoints of the image
        int height = output.size[2];
        int width = output.size[3];

        // The MPI model estimates poses using 15 important key points.
        for (int i = 0; i < 15; i++) {
            // For each keypoint, make a confidence map of each body part. 
            Mat heatMap(height, width, CV_32F, output.ptr(0, i));

            // Find the maximum value in the heatmap and save the value into
            // the confidence variable. 
            Point maxLoc;
            double confidence;
            minMaxLoc(heatMap, 0, &confidence, 0, &maxLoc);

            /* If the confidence level is above 10%, we scale the heatmap back up to the size
            of the original frame, draw a circle at the point where the keypoint is located 
            and label the point.
            */
            if (confidence > 0.1) {
                int x = (int)(maxLoc.x * frame.cols / width);
                int y = (int)(maxLoc.y * frame.rows / height);
                circle(frame, Point(x, y), 5, Scalar(0, 255, 255), -1, LINE_AA);
                putText(frame, to_string(i), Point(x, y), FONT_HERSHEY_SIMPLEX,
                0.5, Scalar(0, 255, 255), 1, LINE_AA);
            }
        }

        imshow("Pose Estimation", frame);
        waitKey(1);
    }
    cap.release();
    destroyAllWindows();
    return 0;
}

// sudo chmod 666 /dev/video0
