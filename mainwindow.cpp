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

    this->timer = new QTimer(this);
    connect(timer, SIGNAL(timeout()), this, SLOT(update_cam_left_image()));
    this->timer->stop();

    // load train samples
    this->show_message("Start Load CSV file with train samples...");
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
    this->show_message("CSV file with train samples loaded successfully");
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
    this->ui->button3->setEnabled(false);
    this->ui->textEdit->clear();
    this->ui->labelLeft->clear();
    this->ui->labelRight->clear();

    this->ui->labelLeft->setMinimumWidth(this->IMG_WIDTH);
    this->ui->labelLeft->setMinimumHeight(this->IMG_HEIGHT);
    this->ui->labelRight->setMinimumWidth(this->IMG_WIDTH);
    this->ui->labelRight->setMinimumHeight(this->IMG_HEIGHT);

    //this->resize(this->WIN_WIDTH, this->WIN_HEIGHT);

    this->ui->button1->setText("File...");
}

void MainWindow::disable_gui()
{
    //disable gui in case of some error
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
        this->ui->button2->setEnabled(false);
        this->ui->labelLeft->clear();
        this->ui->labelRight->clear();
        return;
    }

    //cv::resize(this->leftImage, this->leftImage, this->imgSize,0, 0, cv::INTER_NEAREST);
    this->ui->button2->setEnabled(true);
    this->update_left_image();
}

void MainWindow::on_radioButtonCam_clicked()
{ // cam input
    this->ui->button1->setText("Start");
}

void MainWindow::on_radioButtonVideo_clicked()
{ // video file
    this->ui->button1->setText("File...");
    this->ui->button3->setEnabled(false);
    this->camSource.release();
    this->timer->stop();
}

void MainWindow::on_radioButtonPhoto_clicked()
{ // photo file
    this->ui->button1->setText("File...");
    this->ui->button3->setEnabled(false);
    this->camSource.release();
    this->timer->stop();
}

void MainWindow::on_button1_clicked()
{
    if(!this->ui->radioButtonCam->isChecked())
    { // photo or video file
        string oldPath = "";
        if(this->inputPathFile.length() > 0 && !this->inputPathFile.empty())
        {
            oldPath = this->inputPathFile;
        }

        // open file dialog
        this->inputPathFile = QFileDialog::getOpenFileName(this, tr("Open File"), "", tr("Files (*.*)")).toStdString();

        if(this->inputPathFile.length() > 0 && !this->inputPathFile.empty())
        { // load image file
            this->load_input_image(this->inputPathFile);
        }
        else
        {
            if(oldPath.length() > 0 && !oldPath.empty())
                this->inputPathFile = oldPath;
            else
                this->ui->button2->setEnabled(false);
        }
    }
    else
    { // cam input
        if(this->timer->isActive())
        { // stop recording
            this->ui->button3->setEnabled(false);
            this->ui->button1->setText("Start");
            this->timer->stop();
            this->camSource.release();
        }
        else
        { // start recording
            this->ui->button1->setText("Stop");
            this->ui->button2->setEnabled(false);
            this->ui->button3->setEnabled(true);
            this->timer->start(10);
            this->camSource = VideoCapture(CAM_DEV_ID);
        }
    }

    return;
}

void MainWindow::on_button2_clicked()
{
    int err = 0;
    err=detectFace(this->leftImage, this->rightImage);
    if(err)
        this->show_message("Warning: can not detect face/eye(s)");
    this->update_right_image();
}


void MainWindow::on_button3_clicked()
{
    // take a picture from cam input
    Mat image;
    this->camSource >> image;
    this->camSource >> this->leftImage;

    // stop recording
    this->timer->stop();
    this->camSource.release();

    // clear window for new input
    this->ui->labelLeft->clear();
    this->ui->labelRight->clear();
    this->ui->button1->setText("Start");
    this->ui->button2->setEnabled(true);
    this->ui->button3->setEnabled(false);

    if(this->leftImage.empty())
    {
        this->show_message("Error: Cannot load input image");
        this->ui->button2->setEnabled(false);
        return;
    }

    // show taken picture
    this->update_left_image();
}

void MainWindow::on_button4_clicked()
{
    // reset gui
    this->train();
    cerr << "Face recognized: " << this->recognize(this->leftImage) << endl;
     /*
    this->init_gui();
    this->timer->stop();
    this->camSource.release(); */
}

void MainWindow::update_left_image()
{
    Mat image;
    if(this->leftImage.empty())
        return;
    this->ui->labelLeft->clear();

    cvtColor(this->leftImage, image, CV_BGR2RGB);
    this->qleftImage = QImage((uchar *)image.data, image.cols, image.rows, image.step, QImage::Format_RGB888);
    this->ui->labelLeft->setPixmap(QPixmap::fromImage(this->qleftImage));

    this->ui->labelLeft->setMinimumWidth(this->qleftImage.width());
    this->ui->labelLeft->setMinimumHeight(this->qleftImage.height());
    this->imgSize = Size(this->qleftImage.width(), this->qleftImage.height());
}

void MainWindow::update_right_image()
{
    Mat image;
    if(this->rightImage.empty())
        return;
    this->ui->labelRight->clear();

    cvtColor(this->rightImage, image, CV_BGR2RGB);
    this->qrightImage = QImage((uchar *)image.data, image.cols, image.rows, image.step, QImage::Format_RGB888);
    this->ui->labelRight->setPixmap(QPixmap::fromImage(this->qrightImage));

    this->ui->labelRight->setMinimumWidth(this->qrightImage.width());
    this->ui->labelRight->setMinimumHeight(this->qrightImage.height());
}


void MainWindow::update_cam_left_image()
{
    if(!this->camSource.isOpened())
        return;
    // take one image from cam input
    Mat image;
    this->camSource >> image;
    if(image.empty())
        return;

    // show input from cam on the left image
    this->camSource >> this->leftImage;
    this->update_left_image();

    // detect face from input and show it on the right image
    int err = detectFace(image, this->rightImage);
    if(!err)
        this->update_right_image();

    this->ui->labelLeft->setMinimumWidth(this->qleftImage.width());
    this->ui->labelLeft->setMinimumHeight(this->qleftImage.height());
    this->ui->labelRight->setMinimumWidth(this->qrightImage.width());
    this->ui->labelRight->setMinimumHeight(this->qrightImage.height());
    this->imgSize = Size(this->qleftImage.width(), this->qleftImage.height());
}

void MainWindow::read_csv(const string &filename, vector<Mat>& images, vector<string>& labels, char separator)
{
	cerr << "READING CSV: " << filename << endl;
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
			Mat m = imread(path, 1);
			Mat m2;
			cvtColor(m,m2,CV_BGR2GRAY);
			images.push_back(m2);
            labels.push_back(classlabel);
        }
    }
    this->images=images;
    this->labels=labels;
}

// face recognition
int MainWindow::detectFace( Mat frame, Mat& out )
{
    std::vector<Rect> faces;
    Mat frame_gray;

    frame.copyTo(out);
    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );

    //-- Detect faces
    this->face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE|CV_HAAR_FIND_BIGGEST_OBJECT, Size(30, 30) );
    if (faces.size() == 0 || faces.size() > 1)
    {
        //this->show_message("Error: Cannot find face");
        return 1;
    }
    for( size_t i = 0; i < faces.size(); i++ )
    {

        Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
        ellipse( out, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );

        Mat faceROI = frame_gray( faces[i] );
        std::vector<Rect> leyes;
        std::vector<Rect> reyes;

        //-- In each face, detect eyes
        this->left_eye_cascade.detectMultiScale( faceROI, leyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE|CV_HAAR_FIND_BIGGEST_OBJECT, Size(30, 30) );
        this->right_eye_cascade.detectMultiScale( faceROI, reyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE|CV_HAAR_FIND_BIGGEST_OBJECT, Size(30,30) );
        if (leyes.size() < 1 || reyes.size() < 1)
        {
            //this->show_message("Error: Cannot find left or right eye");
            return 2;
        }

        for( size_t j = 0; j < leyes.size(); j++ )
        {
            Point center( faces[i].x + leyes[j].x + leyes[j].width*0.5, faces[i].y + leyes[j].y + leyes[j].height*0.5 );
            int radius = cvRound( (leyes[j].width + leyes[j].height)*0.25 );
            circle( out, center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );
        }

        for( size_t j = 0; j < reyes.size(); j++ )
        {
            Point center( faces[i].x + reyes[j].x + reyes[j].width*0.5, faces[i].y + reyes[j].y + reyes[j].height*0.5 );
            int radius = cvRound( (reyes[j].width + reyes[j].height)*0.25 );
            circle( out, center, radius, Scalar( 0, 255, 0 ), 4, 8, 0 );
        }

    }

    return 0;
}

// Normalizes a given image into a value range between 0 and 255.
Mat MainWindow::norm_0_255(const Mat& src) {
    // Create and return normalized image:
    Mat dst;
    switch(src.channels()) {
    case 1:
        cv::normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
        break;
    case 3:
        cv::normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
        break;
    default:
        src.copyTo(dst);
        break;
    }
    return dst;
}

void MainWindow::train()
{
	cerr << "TRAINING" << endl;
	if (this->images.size() == 0)
		return;
	//		 number of samples	  dimensionality		  type
	Mat matPCA(this->images.size(), this->images[0].total(), CV_32FC1);
    
    for(int i = 0; i < this->images.size(); i++) {
        //skip unconvertable images
        if(this->images.empty() || this->images[i].total() != this->images[0].total()) {
			cerr << "Skipping " << this->labels[i] << endl;
			continue;
		}
        // Make reshape happy by cloning for non-continuous matrices:
        if(this->images[i].isContinuous()) {
            this->images[i].reshape(1, 1).convertTo(matPCA.row(i), CV_32FC1, 1, 0);
        } else {
            this->images[i].clone().reshape(1, 1).convertTo(matPCA.row(i), CV_32FC1, 1, 0);
        }
        
        this->projections.push_back(this->images[i]);
    }
	this->pca(matPCA, Mat(), CV_PCA_DATA_AS_ROW, matPCA.rows);
	this->mean = this->pca.mean.reshape(1,1);
	this->eugenVal = this->pca.eigenvalues.clone();
	transpose(this->pca.eigenvectors, this->transposedEV);
	
	for(unsigned int i = 0; i< this->images.size(); i++){	
			
		this->projections[i] = subspaceProject(this->transposedEV, this->mean, matPCA.row(i));
	}

}

String MainWindow::recognize(Mat frame) {
	Mat image;
	cvtColor(frame, image, CV_BGR2GRAY);
	string name = "unknown";
	int k=5;
	if(this->labels.size() <= k) 
		return "unknown";

	//project target face to subspace
	Mat target = subspaceProject(this->transposedEV, this->mean, image.reshape(1,1));

	vector<unsigned int> classes(k,0);
	vector<double> distances(k,DBL_MAX);
	
	double distance = DBL_MAX;
	//find k nearest neighbours
	for(int i = 0; i < this->images.size(); i++){
		distance= norm(projections[i],target,NORM_L2);//norml2
		for(int j = 0; j < k; j++){
			if(distance < distances[j]){
				//discard the worst match and shift remaining down 
				for(int l = k-1; l > j; l--){
					distances[l] = distances[l-1];
					classes[l] = classes[l-1];
				}
				classes[j] = i; //set new best
				distances[j] = distance;
				break;
			}
		}
	}

	map<string,Weight> neighbours;
	//count occurence of classes
	for(unsigned int i = 0; i<k; i++){
		Weight &weight = neighbours[this->labels[classes[i]]];
		weight.count++;
		weight.distance += distances[i];
	}

//evaluate voting
	double min_weight = DBL_MAX;
	for (map<string,Weight>::iterator itr = neighbours.begin(); itr != neighbours.end();++itr){
		double weight = itr->second.distance / (double) itr->second.count;
		//concider average weight instead of number of votes
		if(weight < min_weight){
			min_weight = weight;
			name = itr->first;
		} 
	}

	return name;
}
