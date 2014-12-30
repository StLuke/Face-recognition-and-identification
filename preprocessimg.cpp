#include "preprocessimg.h"

PreprocessImg::PreprocessImg(Mat &src)
{
    src.copyTo(this->imgOrig);
    this->face_cascade.load(this->FACE_CASCADE_PATH);
    this->left_eye_cascade_1.load(this->LEFT_EYE_CASCADE_PATH_1);
    this->left_eye_cascade_2.load(this->LEFT_EYE_CASCADE_PATH_2);
    this->left_eye_cascade_3.load(this->LEFT_EYE_CASCADE_PATH_3);
    this->right_eye_cascade_1.load(this->RIGHT_EYE_CASCADE_PATH_1);
    this->right_eye_cascade_2.load(this->RIGHT_EYE_CASCADE_PATH_2);
    this->right_eye_cascade_3.load(this->RIGHT_EYE_CASCADE_PATH_3);
}

PreprocessImg::~PreprocessImg()
{

}

int PreprocessImg::preprocess()
{
    if(this->detectFace(this->imgOrig, this->imgFace))
        return 1;
    resize(imgFace, imgFace, Size(this->FACE_WIDTH, this->FACE_HEIGHT));
    this->toGrayScale(this->imgFace, this->imgGrayFace);

    Point leftEye, rightEye;
    this->detectEyes(this->imgFace, leftEye, rightEye);
    this->rotateFace(this->imgFace, this->imgRotatedFace, leftEye, rightEye);

    this->equalize(this->imgRotatedFace, this->imgPreprocessedFace, true);

    Mat filtered = Mat(this->imgPreprocessedFace.size(), CV_8U);
    bilateralFilter(this->imgPreprocessedFace, filtered, 0, 20.0, 2.0);
    Mat mask = Mat(this->imgPreprocessedFace.size(), CV_8U, Scalar(0));
    Point faceCenter = Point( this->imgPreprocessedFace.cols/2, cvRound(this->imgPreprocessedFace.rows * 0.5));
    Size size = Size( cvRound(this->imgPreprocessedFace.cols * 0.4), cvRound(this->imgPreprocessedFace.rows * 0.8) );
    ellipse(mask, faceCenter, size, 0, 0, 360, Scalar(255), CV_FILLED);

    Mat dstImg = Mat(this->imgPreprocessedFace.size(), CV_8U, Scalar(128)); // Clear the output image to a default gray.

    filtered.copyTo(dstImg, mask); // Copies non-masked pixels from filtered to dstImg.

    dstImg.copyTo(this->imgPreprocessedFace);
    return 0;
}

void PreprocessImg::equalize(Mat &src, Mat &dst, bool sepEqualization)
{
    this->toGrayScale(src, dst);

    Mat eqImg; // whole image
    if(sepEqualization)
    { // we want to do equalization of histogram for left, right and whole image and than join them together
        equalizeHist(dst, eqImg);
    }
    else
    { // equalization just for whole image
        equalizeHist(dst, dst);
        return;
    }

    // left and right side of image separately
    Mat eqImgLeft = dst(Rect(0,0, src.cols/2,src.rows));
    Mat eqImgRight = dst(Rect(src.cols/2,0, src.cols/2,src.rows));
    equalizeHist(eqImgLeft, eqImgLeft);
    equalizeHist(eqImgRight, eqImgRight);

    // combine left, right and whole egualized images
    for(int y=0;y < src.rows;y++)
    {
        for(int x=0;x < src.cols;x++)
        {
            int res = 0;
            int in = eqImg.at<uchar>(y,x);
            float ration = 0.0;
            if (x < src.cols/4)
            { // first 1/4 of image
                 res = eqImgLeft.at<uchar>(y,x);
            }
            else if (x < src.cols*2/4)
            { // second 1/4 of image
                // Join first equalized first 1/4 with second 1/4
                int leftIn = eqImgLeft.at<uchar>(y,x);
                ration = (x - src.cols*1/4.0) / (float)(src.cols*1/4.0);
                res = cvRound((1.0f - ration) * leftIn + (ration) * in);
            }
            else if (x < src.cols*3/4)
            { // third 1/4 of image
                // Join first equalized first 1/4 with second 1/4
                int rightIn = eqImgRight.at<uchar>(y,x-src.cols/2);
                ration = (x - src.cols*2/4.0) / (float)(src.cols*1/4.0);
                res = cvRound((1.0f - ration) * in + (ration) * rightIn);
            }
            else
            { // last 1/4 of image
                res = eqImgRight.at<uchar>(y,x-src.cols/2);
            }
            dst.at<uchar>(y,x) = res;
        }
    }
    return;
}

void PreprocessImg::toGrayScale(Mat &src, Mat &dst)
{
    if(src.channels() == 3)
    {
        cvtColor(src,dst, CV_BGR2GRAY);
    }
    else if(src.channels() == 4)
    {
        cvtColor(src,dst, CV_BGRA2GRAY);
    }
    else if(src.channels() == 1)
    {
        src.copyTo(dst);
    }
    return;
}

int PreprocessImg::detectEyes(Mat &face, Point &leftEye, Point &rightEye)
{
    std::vector<Rect> leftEyes;
    std::vector<Rect> rightEyes;
    Mat face_gray;
    this->equalize(face, face_gray, false);

    Mat left_eye_region = face_gray(Rect(0, 0, face_gray.cols/2, face_gray.rows/2));
    Mat right_eye_region = face_gray(Rect(face_gray.cols/2, 0, face_gray.cols/2, face_gray.rows/2));

    this->left_eye_cascade_1.detectMultiScale(left_eye_region, leftEyes, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE|CV_HAAR_FIND_BIGGEST_OBJECT, Size(20, 20));
    if (leftEyes.size() == 0)
        this->left_eye_cascade_2.detectMultiScale(left_eye_region, leftEyes, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE|CV_HAAR_FIND_BIGGEST_OBJECT, Size(20, 20));
    if (leftEyes.size() == 0)
        this->left_eye_cascade_3.detectMultiScale(left_eye_region, leftEyes, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE|CV_HAAR_FIND_BIGGEST_OBJECT, Size(20, 20));
    if (leftEyes.size() == 0)
        return 1;

    this->right_eye_cascade_1.detectMultiScale(right_eye_region, rightEyes, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE|CV_HAAR_FIND_BIGGEST_OBJECT, Size(20, 20));
    if (rightEyes.size() == 0)
        this->right_eye_cascade_2.detectMultiScale(right_eye_region, rightEyes, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE|CV_HAAR_FIND_BIGGEST_OBJECT, Size(20, 20));
    if (rightEyes.size() == 0)
        this->right_eye_cascade_3.detectMultiScale(right_eye_region, rightEyes, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE|CV_HAAR_FIND_BIGGEST_OBJECT, Size(20, 20));
    if (rightEyes.size() == 0)
        return 1;

    leftEye = Point(leftEyes.at(0).x + leftEyes.at(0).width/2, leftEyes.at(0).y + leftEyes.at(0).height/2);
    rightEye = Point(rightEyes.at(0).x + rightEyes.at(0).width/2 + face_gray.cols/2, rightEyes.at(0).y + rightEyes.at(0).height/2);

    return 0;
}

int PreprocessImg::rotateFace(const Mat &face, Mat &out, Point &leftEye, Point &rightEye)
{
    // get center between eyes
    Point2f eyesCenter = Point2f((leftEye.x + rightEye.x) * 0.5, (leftEye.y + rightEye.y) * 0.5);

    // Get angle in degrees between eyes.
    double dy = (rightEye.y - leftEye.y);
    double dx = (rightEye.x - leftEye.x);
    double angle = atan2(dy, dx) * 180.0/CV_PI;

    // Get the transformation matrix for rotating
    Mat rotation = getRotationMatrix2D(eyesCenter, angle, 1.1);

    Mat warp = Mat(face.rows, face.rows, CV_8U, Scalar(0));
    warpAffine(face, warp, rotation, warp.size());

    warp.copyTo(out);

    return 0;
}

int PreprocessImg::detectFace(Mat frame, Mat& out)
{
    std::vector<Rect> faces;
    this->equalize(this->imgOrig, this->imgEq, false);
    Mat frame_gray = this->imgEq.clone();

    //-- Detect faces
    this->face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE|CV_HAAR_FIND_BIGGEST_OBJECT, Size(30, 30));
    if (faces.size() == 0)
        return 1;

    this->imgOrig(faces[0]).copyTo(out);

    return 0;
}

