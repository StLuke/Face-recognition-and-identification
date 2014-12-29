#include "mainwindow.h"
#include "ui_mainwindow.h"

using namespace std;
using namespace cv;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    this->init_gui();

    if( !this->face_cascade.load(this->FACE_CASCADE_PATH) ||
        !this->right_eye_cascade.load(this->RIGHT_EYE_CASCADE_PATH) ||
        !this->left_eye_cascade.load(this->LEFT_EYE_CASCADE_PATH))
    {
        this->show_message("Error: loading cascade files\n");
        this->disable_gui();
    }

}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::show_message(const string &msg)
{
    this->ui->textEdit->appendPlainText(QString::fromStdString(msg));
    return;
}


void MainWindow::on_actionExit_triggered()
{
    this->close();
}

void MainWindow::init_gui()
{
    this->imgSize = Size(this->IMG_WIDTH, this->IMG_HEIGHT);

    this->ui->radioButtonPhoto->setChecked(true);
    this->ui->button1->setEnabled(true);
    this->ui->button2->setEnabled(false);
    this->ui->textEdit->clear();
    this->ui->labelLeft->clear();
    this->ui->labelRight->clear();

    this->ui->labelLeft->setMinimumWidth(this->IMG_WIDTH);
    this->ui->labelLeft->setMinimumHeight(this->IMG_HEIGHT);
    this->ui->labelRight->setMinimumWidth(this->IMG_WIDTH);
    this->ui->labelRight->setMinimumHeight(this->IMG_HEIGHT);

    this->resize(this->WIN_WIDTH, this->WIN_HEIGHT);
}

void MainWindow::disable_gui()
{
    this->ui->button1->setEnabled(false);
    this->ui->button2->setEnabled(false);
    this->ui->button3->setEnabled(false);
    this->ui->button4->setEnabled(false);
    this->ui->radioButtonCam->setEnabled(false);
    this->ui->radioButtonPhoto->setEnabled(false);
    this->ui->radioButtonVideo->setEnabled(false);
}

void MainWindow::load_input_image(const string &path)
{
    this->show_message("Load file: "+path);
    this->leftImage = imread(path, CV_LOAD_IMAGE_COLOR);
    if(this->leftImage.empty())
    {
        this->show_message("Error: Cannot load file: "+path);
        return;
    }

    cv::resize(this->leftImage, this->leftImage, this->imgSize,0, 0, cv::INTER_NEAREST);
    this->ui->button2->setEnabled(true);
    this->update_left_image();
}

void MainWindow::on_radioButtonCam_clicked()
{
    this->ui->button1->setEnabled(false);
}

void MainWindow::on_radioButtonVideo_clicked()
{
    this->ui->button1->setEnabled(true);
}

void MainWindow::on_radioButtonPhoto_clicked()
{
    this->ui->button1->setEnabled(true);
}

void MainWindow::on_button1_clicked()
{
    string oldPath = "";
    if(this->inputPathFile.length() > 0 && !this->inputPathFile.empty())
    {
        oldPath = this->inputPathFile;
    }
    this->inputPathFile = QFileDialog::getOpenFileName(this, tr("Open File"), "", tr("Files (*.*)")).toStdString();

    if(this->inputPathFile.length() > 0 && !this->inputPathFile.empty())
    {
        this->load_input_image(this->inputPathFile);
    }
    else
    {
        if(oldPath.length() > 0 && !oldPath.empty())
            this->inputPathFile = oldPath;
        else
            this->ui->button2->setEnabled(false);
    }

    return;
}

void MainWindow::on_button2_clicked()
{
    int err = 0;
    this->load_input_image(this->inputPathFile);
    if(err=detectFace(this->leftImage))
    {
        this->show_message("Wrong input data, unable to process");
        return;
    }
    this->update_right_image();
}

void MainWindow::on_button3_clicked()
{
    this->show_message("Start Load CSV file...");
    try
    {
        read_csv(this->CSV_PATH, this->images, this->labels);
    }
    catch (Exception& e)
    {
        this->show_message("Error opening file \""+this->CSV_PATH+"\". Reason: " + e.msg);
        this->disable_gui();
        return;
    }
    this->show_message("CSV file loaded successfully");
}

void MainWindow::on_button4_clicked()
{
    this->init_gui();
}

void MainWindow::update_left_image()
{
    Mat image;
    if(this->leftImage.empty())
        return;
    cvtColor(this->leftImage, image, CV_BGR2RGB);
    this->qleftImage = QImage((uchar *)image.data, image.cols, image.rows, image.step, QImage::Format_RGB888);
    this->ui->labelLeft->setPixmap(QPixmap::fromImage(this->qleftImage));
}

void MainWindow::update_right_image()
{
    Mat image;
    if(this->rightImage.empty())
        return;
    cvtColor(this->rightImage, image, CV_BGR2RGB);
    this->qrightImage = QImage((uchar *)image.data, image.cols, image.rows, image.step, QImage::Format_RGB888);
    this->ui->labelRight->setPixmap(QPixmap::fromImage(this->qrightImage));
}

void MainWindow::read_csv(const string &filename, vector<Mat>& images, vector<string>& labels, char separator)
{
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
            labels.push_back(classlabel);
        }
    }
}

// face recognition
int MainWindow::detectFace( Mat frame )
{
    std::vector<Rect> faces;
    Mat frame_gray;

    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );

    //-- Detect faces
    this->face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );
    if (faces.size() == 0 || faces.size() > 1)
    {
        this->show_message("Error: Cannot find face");
        //return 1;
    }
    for( size_t i = 0; i < faces.size(); i++ )
    {

        Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
        ellipse( frame, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );

        Mat faceROI = frame_gray( faces[i] );
        std::vector<Rect> leyes;
        std::vector<Rect> reyes;

        //-- In each face, detect eyes
        this->left_eye_cascade.detectMultiScale( faceROI, leyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(20, 20) );
        this->right_eye_cascade.detectMultiScale( faceROI, reyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(20, 20) );
        if (leyes.size() != 1 || reyes.size() != 1)
        {
            this->show_message("Error: Cannot find eye(s)");
            //return 2;
        }

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
            circle( frame, center, radius, Scalar( 0, 255, 0 ), 4, 8, 0 );
        }
    }

    this->rightImage = frame;

    return 0;
}


