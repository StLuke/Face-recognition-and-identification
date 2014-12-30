#-------------------------------------------------
#
# Project created by QtCreator 2014-12-28T15:13:49
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = pov
TEMPLATE = app

CONFIG += c++11

SOURCES += main.cpp\
        mainwindow.cpp \
    preprocessimg.cpp

HEADERS  += mainwindow.h \
    preprocessimg.h

FORMS    += mainwindow.ui

INCLUDEPATH += C:/opencv-mingw/install/includes

LIBS        += -LC:/opencv-mingw/install/x64/mingw/bin
LIBS        += -lopencv_calib3d2410 \
    -lopencv_contrib2410 \
    -lopencv_core2410 \
    -lopencv_features2d2410 \
    -lopencv_flann2410 \
    -lopencv_gpu2410 \
    -lopencv_highgui2410 \
    -lopencv_imgproc2410 \
    -lopencv_legacy2410 \
    -lopencv_ml2410 \
    -lopencv_nonfree2410 \
    -lopencv_objdetect2410 \
    -lopencv_photo2410 \
    -lopencv_stitching2410 \
    -lopencv_superres2410 \
    -lopencv_video2410 \
    -lopencv_videostab2410

