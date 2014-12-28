#include <QApplication>
#include "mainwindow.h"

int main(int argc, char *argv[])
{
    /*
	int err=0;
	if ((err=detectFace( image ))){
		cerr << "Wrong input data, unable to process: " << err << endl;;
		return 1;
	}
    */
    QApplication a(argc, argv);
    MainWindow w;
    w.show();

    return a.exec();
}
