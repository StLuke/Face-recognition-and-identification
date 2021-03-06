#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QFileDialog>
#include <QTimer>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/contrib/contrib.hpp>

#include <fstream>
#include <sstream>
#include <iostream>
#include <stdio.h>
#include <vector>

#include "preprocessimg.h"

using namespace cv;
using namespace std;

namespace Ui
{
    class MainWindow;
}

/**
 *
 */
struct Weight
{
    unsigned int count;
    double distance;
};

/**
 *
 */
class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    /**
     *
     */
    explicit MainWindow(QWidget *parent = 0);
    /**
     *
     */
    ~MainWindow();
    /**
     *
     */
    void show_message(const string& msg, bool console_out);

private slots:
    /**
     *
     */
    void on_actionExit_triggered();
    /**
     *
     */
    void on_button1_clicked();
    /**
     *
     */
    void on_radioButtonCam_clicked();
    /**
     *
     */
    void on_radioButtonPhoto_clicked();
    /**
     *
     */
    void on_button4_clicked();
    /**
     *
     */
    void update_left_image();
    /**
     *
     */
    void update_right_image();
    /**
     *
     */
    void on_button3_clicked();
    /**
     *
     */
    void on_button2_clicked();
    /**
     *
     */
    void update_cam_left_image();
    /**
     *
     */
    void on_button5_clicked();

private:
    const string CSV_PATH = "pics2.csv"; /** */
    const string FACE_CASCADE_PATH = "haarcascade_frontalface_alt.xml"; /** */
    const string RIGHT_EYE_CASCADE_PATH = "haarcascade_righteye_2splits.xml"; /** */
    const string LEFT_EYE_CASCADE_PATH = "haarcascade_lefteye_2splits.xml"; /** */
    const int CAM_DEV_ID = 0; /** */
    const int IMG_WIDTH = 250; /** */
    const int IMG_HEIGHT = 250; /** */
    const int WIN_WIDTH = 100+IMG_WIDTH+IMG_WIDTH; /** */
    const int WIN_HEIGHT = 240+IMG_HEIGHT; /** */

    Ui::MainWindow *ui; /** */
    QTimer *timer; /** */

    vector<Mat> images; /** *///storing loaded images of db
    vector<Mat >projections;  /** *///storing loaded images of db
    vector<string> labels;   /** *///storing labels of images
    vector<int> groups; /** storing info about number of testing group for images*/
    CascadeClassifier face_cascade; /** */
    CascadeClassifier right_eye_cascade; /** */
    CascadeClassifier left_eye_cascade; /** */

    Size imgSize;  /** */// size of input image

    QImage qrightImage;  /** */// original displayed input image
    QImage qleftImage;  /** */// changed displayed image

    Mat leftImage;  /** */// original input image
    Mat rightImage;  /** */// changed image

    VideoCapture camSource;  /** */// camera device

    string inputPathFile; /** */ // path to input file

    Mat mean; /** */
    Mat eugenVal; /** */
    Mat transposedEV; /** */
    PCA pca; /** */

    /**
     *
     */
    void init_gui();
    /**
     *
     */
    void disable_gui();
    /**
     *
     */
    void load_input_image(const string &path);
    /**
     *
     */
    void read_csv(const string& filename, vector<Mat>& images, vector<string>& labels, char separator = ';');
    /**
     *
     */
    Mat norm_0_255(const Mat&);
    /**
     *
     */
    void train(vector<Mat> &images, vector<string> &labels, int testGroup);
    /**
     *
     */
    String recognize(Mat);
    /**
     *
     */
    void init_recognizer();
};

#endif // MAINWINDOW_H
