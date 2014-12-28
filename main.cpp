#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <fstream>
#include <sstream>
#include <iostream>
#include <stdio.h>
#include <vector>

using namespace std;
using namespace cv;

vector<Mat> images; //storing loaded images of db
vector<int> labels;  //storing labels of images
String face_cascade_name = "haarcascade_frontalface_alt.xml";
String right_eye_cascade_name = "haarcascade_righteye_2splits.xml";
String left_eye_cascade_name = "haarcascade_lefteye_2splits.xml";
CascadeClassifier face_cascade;
CascadeClassifier right_eye_cascade;
CascadeClassifier left_eye_cascade;

#define DEBUG 0


// face recognition
int detectFace( Mat frame )
{
    std::vector<Rect> faces;
    Mat frame_gray;

    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );

    //-- Detect faces
    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );
	if (faces.size() == 0 || faces.size() > 1) 
		return 1;
	for( size_t i = 0; i < faces.size(); i++ )
	{
		if (DEBUG) {
			Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
			ellipse( frame, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );
		}	

		Mat faceROI = frame_gray( faces[i] );
		std::vector<Rect> leyes;
		std::vector<Rect> reyes;
		
		//-- In each face, detect eyes
		left_eye_cascade.detectMultiScale( faceROI, leyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(30, 30) );
		right_eye_cascade.detectMultiScale( faceROI, reyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(30, 30) );
		if (leyes.size() != 1 || reyes.size() != 1)
			return 2;
		if (DEBUG) {
			for( size_t j = 0; j < leyes.size(); j++ )
			{
				Point center( faces[i].x + leyes[j].x + leyes[j].width*0.5, faces[i].y + leyes[j].y + leyes[j].height*0.5 );
				int radius = cvRound( (leyes[j].width + leyes[j].height)*0.25 );
				circle( frame, center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );
			}
			
			for( size_t j = 0; j < reyes.size(); j++ )
			{
				Point center( faces[i].x + reyes[j].x + reyes[j].width*0.5, faces[i].y + reyes[j].y + reyes[j].height*0.5 );
				int radius = cvRound( (reyes[j].width + reyes[j].height)*0.25 );
				circle( frame, center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );
			}
		}
	}
	if (DEBUG) {
		imshow( "DEBUG", frame );
		waitKey(0);
	}
    return 0;
}

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
    ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
            images.push_back(imread(path, CV_LOAD_IMAGE_COLOR));
            labels.push_back(classlabel));
        }
    }
}

//MAIN
int main(int argc, char *argv[] )
{
    Mat image;
	if (argc != 2) {
		cerr << "./POV path-to-image" << endl;
		return 1;
	}
    //-- 1. Load the cascades
    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading face cascade\n"); return -1; };
    if( !right_eye_cascade.load( right_eye_cascade_name ) ){ printf("--(!)Error loading eyes cascade\n"); return -1; };
    if( !left_eye_cascade.load( left_eye_cascade_name ) ){ printf("--(!)Error loading eyes cascade\n"); return -1; };
	
    //-- 2. Read the image
    image = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    if (DEBUG)
		imshow( "DEBUG", image );
	//-- 3. Apply the classifier to the frame
	int err=0;
	if ((err=detectFace( image ))){
		cerr << "Wrong input data, unable to process: " << err << endl;;
		return 1;
	}
	try {
        read_csv("./pics.csv", images, labels);
    } catch (Exception& e) {
        cerr << "Error opening file \"./pics\". Reason: " << e.msg << endl;
        return 1;
    }
    return 0;
}
