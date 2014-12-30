#ifndef FACEALIGN_H
#define FACEALIGN_H

#include <fstream>
#include <sstream>
#include <iostream>
#include <stdio.h>
#include <vector>


#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>

using namespace std;
using namespace cv;

class PreprocessImg
{

private:
    const string FACE_CASCADE_PATH = "haarcascade_frontalface_alt.xml"; /** */
    const string LEFT_EYE_CASCADE_PATH_1 = "haarcascade_mcs_lefteye.xml"; /** */
    const string LEFT_EYE_CASCADE_PATH_2 = "haarcascade_lefteye_2splits.xml"; /** */
    const string LEFT_EYE_CASCADE_PATH_3 = "haarcascade_eye.xml"; /** */
    const string RIGHT_EYE_CASCADE_PATH_1 = "haarcascade_mcs_righteye.xml"; /** */
    const string RIGHT_EYE_CASCADE_PATH_2 = "haarcascade_righteye_2splits.xml"; /** */
    const string RIGHT_EYE_CASCADE_PATH_3 = "haarcascade_eye.xml"; /** */

    CascadeClassifier face_cascade; /** */
    CascadeClassifier right_eye_cascade_1; /** */
    CascadeClassifier right_eye_cascade_2; /** */
    CascadeClassifier right_eye_cascade_3; /** */
    CascadeClassifier left_eye_cascade_1; /** */
    CascadeClassifier left_eye_cascade_2; /** */
    CascadeClassifier left_eye_cascade_3; /** */
    const int FACE_WIDTH = 300; /** */
    const int FACE_HEIGHT = 300; /** */


public:
    Mat imgOrig;
    Mat imgEq;
    Mat imgGray;
    Mat imgFace;
    Mat imgGrayFace;
    Mat imgRotatedFace;
    Mat imgPreprocessedFace;
    Mat imgCropedFace;

    PreprocessImg(Mat &src);
    ~PreprocessImg();
    void equalize(Mat &src, Mat &dst, bool sepEqualization);
    int detectFace( Mat frame, Mat& out);
    void toGrayScale(Mat &src, Mat &dst);
    int preprocess();
    int detectEyes(Mat &face, Point &leftEye, Point &rightEye);
    int rotateFace(const Mat &face, Mat &out, Point &leftEye, Point &rightEye);
};

#endif // FACEALIGN_H
