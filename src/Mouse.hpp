#ifndef MOUSE_HPP_INCLUDED
#define MOUSE_HPP_INCLUDED

#include <vector>
#include <string>
#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"

/**@brief ����� ����������� ����.
    @version 2.0
    @date 02.11.17
    @code
    pcWin.setWindows("image"); // ������ ��� �������� ���� � ����� ��������� ���������� ����
    //...
    cv:;Mat image;
    cv::rectangle(image, pcWin.boundingBox, cv::Scalar(50, 50, 50), 1); // ���������� ��������������� �����
    cv::imshow("image", image);
    @endcode
*/
class MouseHandler {
//private:

public:
    void onMouse( int event, int x, int y);
    std::string winname; ///< ��� ����
    cv::Point position;
    cv::Point movePosition;
    cv::Rect boundingBox;
    bool isSetBb = false;
    bool isPush = 0;
    bool isPositionSet = 0;
    /**@brief ������������� ������ � ����������� �� ���������.
    */
    MouseHandler();

    ~MouseHandler();

    /**@brief ������� ���� � ��������� ���������� ����
        @param[in] name ��� ����
    */
    void setWindows(std::string name);

    void checkBb(cv::Mat& inputImage);
};

#endif // MOUSE_HPP_INCLUDED
