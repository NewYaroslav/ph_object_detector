#include <iostream>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"

#include "PerceptualHashObjectDetector.hpp"
#include "Mouse.hpp"

using namespace std;

int main() {
    std::cout << "enter camera number: " << std::endl;
    int nCamera = 0; // номер камеры
    std::cin >> nCamera;
    cv::VideoCapture camera(nCamera);
    if(!camera.isOpened())  {
        // если не можем открыть камеру, выводим сообщение
        printf("Cannot open initialize webcam or video file!\n" );
        return 0;
    }

    MouseHandler iMouse;
    // создадим окно и поставим обработчик мышки
    cvNamedWindow("result", CV_WINDOW_AUTOSIZE );
    iMouse.setWindows("result");

    const int HASH25_SIZE = 33554432;
    static unsigned char hash25[HASH25_SIZE]; // массив перцептивных хэшей
    static unsigned char meanData[256];
    memset(meanData, 0xFF, 256);

    unsigned long trainHash = 0xFFFFFFFF; // перцептивный хэш для обучения
    char nObj = 1; // номер объекта для обучения
    int hammingDistance = 1; // максимальная дистнация хэмминга
    double scaleMin = 0.8; // минимальный машстаб окна поиска от первоначального размера ограничительной рамки
    double scaleMax = 1.2; // максимальный машстаб окна поиска от первоначального размера ограничительной рамки
    double scaleStep = 0.05; // шаг масштаба
    double stepLength = 0.05; // сдвиг окна поиска объекта в процентах от его ширины или высоты

    const int MAX_HASH_BIT = 25; // количество бит в хэше
    cv::Mat inputImage; // изображение на входе

    camera >> inputImage; // получаем изображение от камеры

    cv::Mat inputGrayImage; // серое изображение
    cv::Mat imageIntegral(inputImage.cols,inputImage.rows,CV_32FC1); // интегральное изображение
    cv::Mat imageTrainHash; // фрагмент для запоминания

    const int HASH_BOX = 5; // ширина и высота окна хэша 5 пикселей
    const int TRAIN_BOX = 30; // ширина и высота окна для отображения хэша в ЧБ
    PerceptualHashObjectDetector::showHash32(trainHash, imageTrainHash, HASH_BOX, HASH_BOX, TRAIN_BOX, TRAIN_BOX, true);

    cv::Rect bbTrainHash; // ограничительная рамка
    bbTrainHash.width = TRAIN_BOX;
    bbTrainHash.height = TRAIN_BOX;

    bool isRun = false;

    while(1) {
        camera >> inputImage;

        cvtColor( inputImage, inputGrayImage, CV_BGR2GRAY ); // cоздаем ЧБ изображение
        cv::integral(inputGrayImage, imageIntegral);

        // делаем проверку ограничительных рамок
        iMouse.checkBb(inputImage);
        if(iMouse.boundingBox.x < 1) iMouse.boundingBox.x = 1;
        if(iMouse.boundingBox.y < 1) iMouse.boundingBox.y = 1;
        if(iMouse.boundingBox.x > inputImage.cols - TRAIN_BOX - iMouse.boundingBox.width) iMouse.boundingBox.x = inputImage.cols - TRAIN_BOX - iMouse.boundingBox.width;
        if(iMouse.boundingBox.y > inputImage.rows - TRAIN_BOX - iMouse.boundingBox.height) iMouse.boundingBox.y = inputImage.rows - TRAIN_BOX - iMouse.boundingBox.height;

        bbTrainHash.x = iMouse.boundingBox.x + iMouse.boundingBox.width;
        bbTrainHash.y = iMouse.boundingBox.y;
        // получаем перцептивный хэш
        trainHash = PerceptualHashObjectDetector::getHash32(inputGrayImage, iMouse.boundingBox, HASH_BOX, HASH_BOX);
        PerceptualHashObjectDetector::showHash32(trainHash, imageTrainHash, HASH_BOX, HASH_BOX, TRAIN_BOX, TRAIN_BOX, true); // получаем изображение перцептивного хэша

        // выводим значение перцептивного хэша на экран
        char texthash[512];
        sprintf(texthash, "hash: %8.8X", trainHash);
        cv::putText(inputImage, texthash, cv::Point(iMouse.boundingBox.x, iMouse.boundingBox.y - 10), CV_FONT_HERSHEY_PLAIN, 0.9,cv::Scalar(0, 0, 255), 1, 8, 0);

        cv::rectangle(inputImage, iMouse.boundingBox, cv::Scalar(255, 0, 0)); // нарисуем ограничительную рамку
        // нарисуем перцептивный хэш
        cv::Mat roi(inputImage, bbTrainHash);
        imageTrainHash.copyTo(roi);
        roi.release();

        if(isRun) { // если детектор объектов работает
            std::vector<cv::Rect> vBb; // вектор содержащий ограничительные рамки объектов
            std::vector<unsigned long> vHash; // вектор содержаший хэши объектов
            std::vector<cv::Scalar> vColor; // вектор содержащий разные цвета найденных объектов
            vColor.push_back(cv::Scalar(0, 255, 0));
            vColor.push_back(cv::Scalar(0, 255, 255));
            vColor.push_back(cv::Scalar(0, 0, 255));
            vColor.push_back(cv::Scalar(255, 255, 0));
            vColor.push_back(cv::Scalar(255, 255, 255));
            // ищем объекты
            PerceptualHashObjectDetector::searhIntegralImage(imageIntegral,
                hash25, meanData,
                scaleMin, scaleMax, scaleStep, stepLength,
                iMouse.boundingBox.width, iMouse.boundingBox.height, vBb, vHash);
            // отображаем найденное
            for(int i = 0; i < vBb.size(); ++i) {
                int hashArrayData = hash25[vHash[i]];
                int numColor = hashArrayData == 0x01 ? 1 : (hashArrayData == 0x02 ? 2 : (hashArrayData == 0x04 ? 3 : 4));
                cv::rectangle(inputImage, vBb[i], vColor[numColor]); // нарисуем ограничительную рамку
                char texthash[512];
                sprintf(texthash, "%8.8X", vHash[i]);
                cv::putText(inputImage, texthash, cv::Point(vBb[i].x, vBb[i].y - 10), CV_FONT_HERSHEY_PLAIN, 1.0,vColor[numColor], 1, 8, 0);
            }
        }

        cv::imshow("result", inputImage);

        char symbol = cv::waitKey(20);
        if(symbol == '1' || symbol == '1') { // выбрать для обучения номер объекта 1
            nObj = 1; printf("set obj 1\n");
        } else
        if(symbol == '2' || symbol == '2') { // выбрать для обучения номер объекта 2
            nObj = 2; printf("set obj 2\n");
        } else
        if(symbol == '3' || symbol == '3') { // выбрать для обучения номер объекта 3
            nObj = 4; printf("set obj 3\n");
        } else
        if(symbol == 'B' || symbol == 'b') { // выйти из программы
            break;
        } else
        if(symbol == 'T' || symbol == 't') { // запомнить обучающую выборку
            PerceptualHashObjectDetector::getNoise32(trainHash, hash25, nObj, hammingDistance, MAX_HASH_BIT);
            printf("add hash: %8.8X\n", trainHash);
        } else
        if(symbol == 'R' || symbol == 'r') { // запустить или остановить детектор объектов
            isRun = !isRun;
            if(isRun) printf("hash detector on\n");
            else printf("hash detector off\n");
        } else
        if(symbol == 'C' || symbol == 'c') { // очистить цели
            memset(hash25, 0, HASH25_SIZE);
            printf("clear hash\n");
        }
    }
    // освобождаем память
    inputImage.release();
    imageIntegral.release();
    imageTrainHash.release();
    return 0;
}
