#ifndef MOUSE_HPP_INCLUDED
#define MOUSE_HPP_INCLUDED

#include <vector>
#include <string>
#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"

/**@brief Класс обработчика мыши.
    @version 2.0
    @date 02.11.17
    @code
    pcWin.setWindows("image"); // задать имя рабочего окна а также запустить обработчик мыши
    //...
    cv:;Mat image;
    cv::rectangle(image, pcWin.boundingBox, cv::Scalar(50, 50, 50), 1); // нарисовать ограничительную рамку
    cv::imshow("image", image);
    @endcode
*/
class MouseHandler {
//private:

public:
    void onMouse( int event, int x, int y);
    std::string winname; ///< имя окна
    cv::Point position;
    cv::Point movePosition;
    cv::Rect boundingBox;
    bool isSetBb = false;
    bool isPush = 0;
    bool isPositionSet = 0;
    /**@brief Инициализация класса с параметрами по умолчанию.
    */
    MouseHandler();

    ~MouseHandler();

    /**@brief Создать окно и запустить обработчик мыши
        @param[in] name имя окна
    */
    void setWindows(std::string name);

    void checkBb(cv::Mat& inputImage);
};

#endif // MOUSE_HPP_INCLUDED
