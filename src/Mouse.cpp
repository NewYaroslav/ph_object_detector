#include "Mouse.hpp"
#include <stdio.h>


static void onMouse( int event, int x, int y, int, void* userdata) {
  MouseHandler* app = static_cast<MouseHandler*>(userdata);
  if (app)
    app->onMouse(event, x, y);
}
// обрабочтки мыши
void MouseHandler::onMouse( int event, int x, int y) {

    switch (event) {
        case CV_EVENT_LBUTTONDOWN:
            //position.x = x;
            //position.y = y;
            boundingBox.x = x;
            boundingBox.y = y;
            isPositionSet = false;
            isPush = true;
            isSetBb = true;
        break;
        case CV_EVENT_LBUTTONUP:
            position.x = x;
            position.y = y;
            boundingBox.width = std::abs( x - boundingBox.x );
            boundingBox.height = std::abs( y - boundingBox.y );
            boundingBox.width = (boundingBox.width + boundingBox.height) / 2;
            boundingBox.height = boundingBox.width;
            isSetBb = false;
            isPositionSet = true;
            isPush = false;
            //printf("set %d %d\n",x,y);
        break;
        case CV_EVENT_MOUSEMOVE:
            movePosition.x = x;
            movePosition.y = y;
            if(isSetBb) {
                // если ранее мышкой была выбрана точка верхнего уголка рамки
                // то изменяем ширину и высоту рамки
                boundingBox.width = std::abs( x - boundingBox.x );
                boundingBox.height = std::abs( y - boundingBox.y );
                boundingBox.width = (boundingBox.width + boundingBox.height) / 2;
                boundingBox.height = boundingBox.width;
            }
            //printf("move %d %d\n",movePosition.x,movePosition.y);
        break;
    }
}

MouseHandler::MouseHandler() {
    position.x = 0;
    position.y = 0;
}

MouseHandler::~MouseHandler() {

}

void MouseHandler::setWindows(std::string name) {
    this->winname = name;
    cv::setMouseCallback(winname, ::onMouse, this);
}

void MouseHandler::checkBb(cv::Mat& inputImage) {
    // если размеры изображения стали меньше положения или размеров ограничительной рамки
    if (boundingBox.x >= inputImage.cols) {
        boundingBox.x = inputImage.cols - 2;
    }
    if (boundingBox.y >= inputImage.rows) {
        boundingBox.y = inputImage.rows - 2;
    }
    if (boundingBox.width < 20) {
        // минимальная ширина
        boundingBox.width = 20;
    }
    if (boundingBox.height < 20) {
        // минимальная высота
        boundingBox.height = 20;
    }
    if ((boundingBox.x + boundingBox.width) >= inputImage.cols) {
        boundingBox.width = inputImage.cols - boundingBox.x - 1;
    }
    if ((boundingBox.y + boundingBox.height) >= inputImage.rows) {
        boundingBox.height = inputImage.rows - boundingBox.y - 1;
    }
    // если размеры изображения стали меньше положения или размеров ограничительной рамки
    if ((boundingBox.x >= inputImage.cols) || (boundingBox.y >= inputImage.rows) || ((boundingBox.x + boundingBox.width) >= inputImage.cols) || ((boundingBox.height + boundingBox.y) >= inputImage.rows)) {
        boundingBox.x = 0.0; boundingBox.y = 0.0;
        boundingBox.width = (float)((int)inputImage.cols - (int)1); boundingBox.height = (float)((int)inputImage.rows - (int)1);
    }
}
