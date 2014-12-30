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
        this->show_message("Error: loading cascade files\n", true);
        this->disable_gui();
    }

    this->timer = new QTimer(this);
    connect(timer, SIGNAL(timeout()), this, SLOT(update_cam_left_image()));
    this->timer->stop();
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::show_message(const string &msg, bool console_out)
{
    this->ui->textEdit->appendPlainText(QString::fromStdString(msg));
    if(console_out)
        cerr << msg << endl;
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
}

void MainWindow::load_input_image(const string &path)
{
    this->show_message("Load file: "+path, false);
    this->leftImage = imread(path, CV_LOAD_IMAGE_COLOR);
    if(this->leftImage.empty())
    {
        this->show_message("Error: Cannot load file: "+path, false);
        this->ui->button2->setEnabled(false);
        this->ui->labelLeft->clear();
        this->ui->labelRight->clear();
        return;
    }

    this->ui->button2->setEnabled(true);
    this->update_left_image();
}

void MainWindow::on_radioButtonCam_clicked()
{ // cam input
    this->ui->button1->setText("Start");
}

void MainWindow::on_radioButtonPhoto_clicked()
{ // photo file
    this->ui->button1->setText("File...");
    this->ui->button3->setEnabled(false);
    this->camSource.release();
    this->timer->stop();
}

void MainWindow::init_recognizer()
{
    // load train samples
    this->show_message("Start Load CSV file with train samples...", true);
    try
    {
        read_csv(this->CSV_PATH, this->images, this->labels);
    }
    catch (Exception& e)
    {
        this->show_message("Error opening file \""+this->CSV_PATH+"\". Reason: " + e.msg, true);
        this->disable_gui();
        return;
    }
    this->show_message("CSV file with train samples loaded successfully", true);

    // init structures for training
    this->projections.clear();
    this->transposedEV = Mat();
    this->eugenVal = Mat();
    this->mean = Mat();
    this->pca = PCA();

    // train our model
    this->show_message("Training...", true);
    this->train(this->images, this->labels, -1);
    this->show_message("Training done", true);
}

void MainWindow::on_button1_clicked()
{
    if(!this->ui->radioButtonPhoto->isEnabled())
    {
        // init recognizer
        this->init_recognizer();

        // set gui to enabled mode
        this->ui->button1->setText("File...");
        this->ui->radioButtonCam->setEnabled(true);
        this->ui->radioButtonPhoto->setEnabled(true);
        this->ui->button4->setEnabled(true);
    }
    else if(!this->ui->radioButtonCam->isChecked())
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
    PreprocessImg img(this->leftImage);
    if(!img.preprocess())
    {
        img.imgCropedFace.copyTo(this->rightImage);

        this->update_right_image();
    }

    this->show_message("Face recognized: " + this->recognize(img.imgPreprocessedFace), false);
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
        this->show_message("Error: Cannot load input image", false);
        this->ui->button2->setEnabled(false);
        return;
    }

    // show taken picture
    this->update_left_image();
}

void MainWindow::on_button4_clicked()
{
    // reset gui
    this->init_gui();
    this->timer->stop();
    this->camSource.release();
}


void MainWindow::on_button5_clicked()
{
    if(this->images.empty() || this->labels.empty())
    { // in case no images and labels are loaded
        this->show_message("Start Load CSV file with train samples...", true);
        try
        {
            read_csv(this->CSV_PATH, this->images, this->labels);
        }
        catch (Exception& e)
        {
            this->show_message("Error opening file \""+this->CSV_PATH+"\". Reason: " + e.msg, true);
            this->disable_gui();
            return;
        }
        this->show_message("CSV file with train samples loaded successfully", true);
    }

    // set images to groups, sequentially
    int groupsNum = 10;
    unsigned int groupSize = ceil(this->images.size()/(double)groupsNum);
    this->groups.clear();
    int currentGroup = 0;
    for(unsigned int i = 0, j = 0; i < this->images.size(); i++, j++)
    {
        if(j >= groupSize)
        {
            currentGroup++;
            j = 0;
        }
        this->groups.push_back(currentGroup);
    }
    // now randomly
    random_shuffle ( this->groups.begin(), this->groups.end() );

    int err = 0;
    int testNum = 0;

    for(int j = 0; j < groupsNum; j++)
    {
        // init structures for training
        this->projections.clear();
        this->transposedEV = Mat();
        this->eugenVal = Mat();
        this->mean = Mat();
        this->pca = PCA();
        this->show_message("Training for test "+to_string(j)+"...", true);
        this->train(this->images, this->labels, j);
        this->show_message("Training for test "+to_string(j)+" done", true);

        int actErr = 0;
        int actTestNum = 0;

        for(unsigned int i = 0; i < this->images.size(); i++)
        {
            if(this->groups[i] != j)
                continue;
            actTestNum++;
            if(this->labels[i].compare(this->recognize(this->images[i])) != 0)
                actErr++;
        }

        err += actErr;
        testNum += actTestNum;

        this->show_message("Test "+to_string(j)+" done... success in "+to_string(actTestNum-actErr)+"/"+to_string(actTestNum), true);
    }

    this->show_message("Cross-validation done... success in "+to_string(testNum-err)+"/"+to_string(testNum)+" => "+to_string((testNum-err)/(double)testNum)+"%", true);

}

void MainWindow::update_left_image()
{
    Mat image;
    if(this->leftImage.empty())
        return;
    this->ui->labelLeft->clear();

    if(this->leftImage.channels() == 3)
        cvtColor(this->leftImage, image, CV_BGR2RGB);
    if(this->leftImage.channels() == 4)
        cvtColor(this->leftImage, image, CV_BGRA2RGB);
    if(this->leftImage.channels() == 1)
        cvtColor(this->leftImage, image, CV_GRAY2RGB);
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

    if(this->rightImage.channels() == 3)
        cvtColor(this->rightImage, image, CV_BGR2RGB);
    if(this->rightImage.channels() == 4)
        cvtColor(this->rightImage, image, CV_BGRA2RGB);
    if(this->rightImage.channels() == 1)
        cvtColor(this->rightImage, image, CV_GRAY2RGB);
    cv::resize(image, image, Size(this->leftImage.rows, this->leftImage.rows));
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

    // Preprocess online image and show it on the right image
    PreprocessImg img(this->leftImage);
    if(!img.preprocess())
    {
        img.imgCropedFace.copyTo(this->rightImage);
        this->update_right_image();
    }
    this->show_message("Face recognized: " + this->recognize(img.imgPreprocessedFace), false);

    this->ui->labelLeft->setMinimumWidth(this->qleftImage.width());
    this->ui->labelLeft->setMinimumHeight(this->qleftImage.height());
    this->ui->labelRight->setMinimumWidth(this->qrightImage.width());
    this->ui->labelRight->setMinimumHeight(this->qrightImage.height());
    this->imgSize = Size(this->qleftImage.width(), this->qleftImage.height());
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
            Mat m = imread(path, 1);
            if(m.empty())
                continue;
            this->show_message("Loading and preprocessing training image: " + path, true);
            PreprocessImg img = PreprocessImg(m);
            img.preprocess();
            images.push_back(img.imgPreprocessedFace);
            labels.push_back(classlabel);
        }
    }
    this->images=images;
    this->labels=labels;
}


// Normalizes a given image into a value range between 0 and 255.
Mat MainWindow::norm_0_255(const Mat& src)
{
    // Create and return normalized image:
    Mat dst;
    switch(src.channels())
    {
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

void MainWindow::train(vector<Mat> &images, vector<string> &labels, int testGroup)
{
    if (images.size() == 0)
        return;
    //          number of samples	  dimensionality		  type
    Mat matPCA(images.size(), images[0].total(), CV_32FC1);

    for(unsigned int i = 0; i < images.size(); i++)
    {
        if(testGroup != -1 && i < this->groups.size())
        {
            if(this->groups[i] == testGroup)
            {
                //this->show_message("Skipping testing image: " + labels[i], true);
                continue;
            }
        }
        if(images.empty() || images[i].total() != images[0].total())
        { //skip unconvertable images
            this->show_message("Skipping uncovertable image: " + labels[i], true);
            continue;
        }
        if(images[i].isContinuous())
        { // Make reshape happy by cloning for non-continuous matrices:
            images[i].reshape(1, 1).convertTo(matPCA.row(i), CV_32FC1, 1, 0);
        }
        else
        {
            images[i].clone().reshape(1, 1).convertTo(matPCA.row(i), CV_32FC1, 1, 0);
        }
        this->projections.push_back(images[i]);
    }
    this->pca(matPCA, Mat(), CV_PCA_DATA_AS_ROW, matPCA.rows);
    this->mean = this->pca.mean.reshape(1,1);
    this->eugenVal = this->pca.eigenvalues.clone();
    transpose(this->pca.eigenvectors, this->transposedEV);

    for(unsigned int i = 0; i< this->projections.size(); i++)
    {
        this->projections[i] = subspaceProject(this->transposedEV, this->mean, matPCA.row(i));
    }

    return;
}

String MainWindow::recognize(Mat frame)
{
    Mat image;
    if(frame.channels() == 3)
        cvtColor(frame,image, CV_BGR2GRAY);
    else if(frame.channels() == 4)
        cvtColor(frame,image, CV_BGRA2GRAY);
    else if(frame.channels() == 1)
        frame.copyTo(image);
    string name = "unknown";
    unsigned int k=5;
    if(this->labels.size() <= k)
        return "unknown";

    //project target face to subspace
    Mat target = subspaceProject(this->transposedEV, this->mean, image.reshape(1,1));

    vector<unsigned int> classes(k,0);
    vector<double> distances(k,DBL_MAX);

    double distance = DBL_MAX;
    //find k nearest neighbours
    for(unsigned int i = 0; i < this->projections.size(); i++)
    {
        distance= norm(projections[i],target,NORM_L2);//norml2
        for(unsigned int j = 0; j < k; j++)
        {
            if(distance < distances[j])
            { //discard the worst match and shift remaining down
                for(unsigned int l = k-1; l > j; l--)
                {
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
    for(unsigned int i = 0; i < k; i++)
    {
        Weight &weight = neighbours[this->labels[classes[i]]];
        weight.count++;
        weight.distance += distances[i];
    }

    //evaluate voting
    double min_weight = DBL_MAX;
    for (map<string,Weight>::iterator itr = neighbours.begin(); itr != neighbours.end();++itr)
    {
        double weight = itr->second.distance / (double) itr->second.count;
        if(weight < min_weight)
        { //concider average weight instead of number of votes
            min_weight = weight;
            name = itr->first;
        }
    }

    return name;
}

