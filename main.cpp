#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <unistd.h>
#include <ctime>
#include <stdio.h>
enum OptionType {
    TYPE_UNKNOWN = 0,
    TYPE_CALIBRATE,
    TYPE_AUTO_RUN,
    TYPE_DEBUG
};


void HSV_calib(const cv::Mat img, int *thres, int mode);
void RemoveSmallRegion(cv::Mat &Src, cv::Mat &Dst, int AreaLimit, int CheckMode, int NeihborMode);
void find_apple();
void txtRead(int **hsv);
void txtWrite(int **hsv);
void* address;
void qsort(int s, int e, int op, int a[][2]);
int remainder2[9] = {4, 2, 1, 4, 2, 1, 4, 2, 1};
int remainder3[9] = {2, 3, 1, 5, 4, 6, 2, 3, 1};

int main(int argc, char *argv[]) {
    int opt = 0;
    bool out = false, in = false;
    std::string inFile, outFile;
    OptionType optiontype = TYPE_AUTO_RUN;
    while ((opt = getopt(argc, argv, "hm:i:o:")) != -1) {
        switch (opt) {
            case 'h':
                printf("Usage: ./<filename> -m <mode> -o <outfile>\n");
                return 0;
            case 'm':
                if (!strcmp("debug", optarg))
                    optiontype = TYPE_DEBUG;
                else if (!strcmp("calib", optarg))
                    optiontype = TYPE_CALIBRATE;
                else if (!strcmp("auto", optarg))
                    optiontype = TYPE_AUTO_RUN;
                break;
            case 'i':
                in = true;
                inFile = std::string(optarg);
                break;
            case 'o':
                out = true;
                outFile = std::string(optarg);
                break;
            default:
                ;
        }
    }
    if (optiontype == TYPE_CALIBRATE){
        ; // TODO
    }
    if (out)
        std::cout << outFile << std::endl;
    else
        std::cout << "no record" << std::endl;

    cv::VideoCapture cap;
    if (!in)
        cap.open(0);
    else
        cap.open("../test.mp4");
    assert(cap.isOpened()); // TODO: if failed then let diankong randomly choose number
    if (out){
        cv::VideoWriter writer;
        cv::Size size = cv::Size(640, 480);
        // 基于当前系统的当前日期/时间
        time_t now = time(0);
        // 把 now 转换为字符串形式
        char* dt = ctime(&now);
        writer.open("../orig.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, size, true);//TODO
    }
    cv::Mat img;
    cap >> img;
//    if (cap.isOpened()) std::cout << "Camera is Opened.\n";
//    else std::cout << "Camera init failed.\n";

    // choose 3 groups of threshold for HSV
//    int hsv[3][6] = {{0},{0},{0}}; // iLowH, iHighH, iLowS, iHighS, iLowV, iHighV;
    int **hsv = new int* [3];
    for (int i = 0; i < 3; ++i)
        hsv[i] = new int[6];
    address = (void *)hsv[0];
    if (optiontype == TYPE_CALIBRATE){
        for (int i = 0; i < 3; ++i) {
            HSV_calib(img, hsv[i], i);
            std::cout << "LowH:" << hsv[i][0] << std::endl;
            std::cout << "HighH:" << hsv[i][1] << std::endl;
            std::cout << "LowS:" << hsv[i][2] << std::endl;
            std::cout << "HighS:" << hsv[i][3] << std::endl;
            std::cout << "LowV:" << hsv[i][4] << std::endl;
            std::cout << "HighV:" << hsv[i][5] << std::endl;
        }
        for (int i=0; i<3; ++i)
            for(int j=0; j<6; ++j)
                std::cout << hsv[i][j] << std::endl;
        txtWrite(hsv);
    }

    // amazing part
    txtRead(hsv);
    std::cout << "read hsv" << std::endl;
    for (int i = 0; i < 3; ++i) {
        std::cout << "LowH:" << hsv[i][0] << std::endl;
        std::cout << "HighH:" << hsv[i][1] << std::endl;
        std::cout << "LowS:" << hsv[i][2] << std::endl;
        std::cout << "HighS:" << hsv[i][3] << std::endl;
        std::cout << "LowV:" << hsv[i][4] << std::endl;
        std::cout << "HighV:" << hsv[i][5] << std::endl;
    }
    cv::Scalar RedLo(hsv[0][0], hsv[0][2], hsv[0][4]);
    cv::Scalar RedHi(hsv[0][1], hsv[0][3], hsv[0][5]);
    cv::Scalar BlueLo(hsv[1][0], hsv[1][2], hsv[1][4]);
    cv::Scalar BlueHi(hsv[1][1], hsv[1][3], hsv[1][5]);
    cv::Scalar GreenLo(hsv[2][0], hsv[2][2], hsv[2][4]);
    cv::Scalar GreenHi(hsv[2][1], hsv[2][3], hsv[2][5]);
    while (true) {
        cap >> img;
        if(!img.data) return 0;//判断是否有数据
        if (optiontype == TYPE_DEBUG)
            cv::imshow("orig", img);
        cv::Mat imageHSV;
        cv::cvtColor(img, imageHSV, cv::COLOR_BGR2HSV);
        cv::Mat imageThresholdRed, imageThresholdBlue, imageThresholdGreen, imageThreshold;
        // do threshold
        cv::inRange(imageHSV, RedLo, RedHi, imageThresholdRed);
        // create kernel
        cv::Mat element = getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        // open operation -- remove some small noise
        cv::morphologyEx(imageThresholdRed, imageThresholdRed, cv::MORPH_OPEN, element);
        // close operation -- join some small regions
        cv::morphologyEx(imageThresholdRed, imageThresholdRed, cv::MORPH_CLOSE, element);

        cv::inRange(imageHSV, BlueLo, BlueHi, imageThresholdBlue);
        cv::morphologyEx(imageThresholdBlue, imageThresholdBlue, cv::MORPH_OPEN, element);
        cv::morphologyEx(imageThresholdBlue, imageThresholdBlue, cv::MORPH_CLOSE, element);

        cv::inRange(imageHSV, GreenLo, GreenHi, imageThresholdGreen);
        cv::morphologyEx(imageThresholdGreen, imageThresholdGreen, cv::MORPH_OPEN, element);
        cv::morphologyEx(imageThresholdGreen, imageThresholdGreen, cv::MORPH_CLOSE, element);

        imageThreshold = (imageThresholdRed + imageThresholdBlue + imageThresholdGreen) > 0;
        RemoveSmallRegion(imageThreshold, imageThreshold, 500, 1, 1);
        if (optiontype == TYPE_DEBUG) {
            cv::imshow("thres", imageThreshold);
/*            cv::imshow("thresR", imageThresholdRed);
            cv::imshow("thresB", imageThresholdBlue);
            cv::imshow("thresG", imageThresholdGreen);*/
        }

        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Point> contour;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(imageThreshold, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point());
        cv::Moments moment;//矩
        std::vector<cv::RotatedRect> rect;
        int center[15][2];
        int cnt=0;
        for (int i = 0; i < contours.size(); ++i){
            contour = contours[i];
            cv::Mat temp(contour);
            moment = moments(temp, false);
//            std::cout << "size: " << cv::contourArea(contour) << std::endl;
            cv::RotatedRect rect_tmp = cv::minAreaRect(contour);
//            std::cout << "rect size: " << rect_tmp.size.area() << std::endl;
            if (contourArea(contour)/rect_tmp.size.area() >= 0.9)
                rect.push_back(rect_tmp);
            else continue;
            cv::Point pt1;
            if (moment.m00 != 0)//除数不能为0
            {
                pt1.x = cvRound(moment.m10 / moment.m00);//计算重心横坐标
                pt1.y = cvRound(moment.m01 / moment.m00);//计算重心纵坐标
            }
            center[cnt][0]=pt1.x;
            center[cnt++][1]=pt1.y;
            cv::circle(img, pt1, 5, cv::Scalar(0,0,0), 2);//draw the center point
            if(optiontype == TYPE_DEBUG) std::cout << "contour" << i << " x=" << pt1.x << ", y=" << pt1.y << std::endl;
        }
        if(optiontype == TYPE_DEBUG) std::cout << "rects size:" << rect.size() << std::endl;
        if (optiontype == TYPE_DEBUG){
            cv::imshow("orig", img);
        }
        if (rect.size()<9) continue;

        //TODO

        if (rect.size()!=9) continue;

        //sort
        qsort(0,8,1,center);
        for (int i=0; i < 3; ++i) qsort(i*3,i*3+2,0,center);
//        for (int i=0; i < 9; ++i) std::cout<<center[i][0]<<' '<<center[i][1]<<std::endl;

        int halfWidth=20;
        int res[9];
//        for(int i = 0; i < imageThresholdRed.size().height; ++i)
//            for (int j = 0; j < imageThresholdRed.size().width; ++j)
//                if (imageThresholdRed.at<uchar>(i, j)) std::cout<<'|'<<i<<' '<<j<<'|'<<std::endl;
        for(int i=0; i < 9; ++i){
            int valRed=0, valGreen=0, valBlue=0;
            for(int j=center[i][0]-halfWidth; j<=center[i][0]+halfWidth; ++j){
                for(int k=center[i][1]-halfWidth; k<=center[i][1]+halfWidth; ++k){
//                    std::cout<<j<<' '<<k<<std::endl;
//                    cv::circle(img, cv::Point(j,k), 5, cv::Scalar(255,80,160), 2);
//                    std::cout<<imageThresholdRed.at<uchar>(j,k)<<' '<<imageThresholdGreen.at<uchar>(j,k)<<' '<<imageThresholdBlue.at<uchar>(j,k)<<std::endl;
//                    std::cout<<imageThresholdRed;
                    if (imageThresholdRed.at<uchar>(k,j)) valRed++;
                    if (imageThresholdGreen.at<uchar>(k,j)) valGreen++;
                    if (imageThresholdBlue.at<uchar>(k,j)) valBlue++;
                }
            }
            if (optiontype == TYPE_DEBUG) std::cout<<valRed<<' '<<valGreen<<' '<<valBlue<<' '<<std::endl;
            if (valRed>valBlue)
                if (valRed > valGreen) res[i]=0;
                else res[i]=2;
            else
                if (valBlue > valGreen) res[i]=1;
                else res[i]=2;
        }

        int finalRes=0;
        for(int i=0; i<9; ++i) finalRes+=remainder3[i]*res[i];
        finalRes%=7;

        if(optiontype == TYPE_DEBUG){
            for(int i = 0; i < 9; ++i) std::cout<<res[i]<<' ';
            std::cout<<std::endl;
        }
        std::cout<<"final result:"<<finalRes<<std::endl;


        char key = (char) cv::waitKey(300);
        if (key == 27)
            break;
    }

    for (int i = 0; i < 3;++i)
        delete[] hsv[i];
    delete[] hsv;
    return 0;
}

/*
void colorEnhancement(cv::Mat &src, cv::Mat &dst, int filter)
{

    cv::Mat orig_img = src.clone();
    cv::Mat simg;

    if (orig_img.channels() != 1)
    {
        cvtColor(orig_img, simg, cv::COLOR_BGR2GRAY);
    }
    else
    {
        return;
    }

    long int N = simg.rows*simg.cols;

    int histo_b[256];
    int histo_g[256];
    int histo_r[256];

    for(int i=0; i<256; i++)
    {
        histo_b[i] = 0;
        histo_g[i] = 0;
        histo_r[i] = 0;
    }
    cv::Vec3b intensity;

    for(int i=0; i<simg.rows; i++)
    {
        for(int j=0; j<simg.cols; j++)
        {
            intensity = orig_img.at<cv::Vec3b>(i,j);

            histo_b[intensity.val[0]] = histo_b[intensity.val[0]] + 1;
            histo_g[intensity.val[1]] = histo_g[intensity.val[1]] + 1;
            histo_r[intensity.val[2]] = histo_r[intensity.val[2]] + 1;
        }
    }

    for(int i = 1; i<256; i++)
    {
        histo_b[i] = histo_b[i] + filter * histo_b[i-1];
        histo_g[i] = histo_g[i] + filter * histo_g[i-1];
        histo_r[i] = histo_r[i] + filter * histo_r[i-1];
    }

    int vmin_b=0;
    int vmin_g=0;
    int vmin_r=0;
    int s1 = 3;
    int s2 = 3;

    while(histo_b[vmin_b+1] <= N*s1/100)
    {
        vmin_b = vmin_b +1;
    }
    while(histo_g[vmin_g+1] <= N*s1/100)
    {
        vmin_g = vmin_g +1;
    }
    while(histo_r[vmin_r+1] <= N*s1/100)
    {
        vmin_r = vmin_r +1;
    }

    int vmax_b = 255-1;
    int vmax_g = 255-1;
    int vmax_r = 255-1;

    while(histo_b[vmax_b-1]>(N-((N/100)*s2)))
    {
        vmax_b = vmax_b-1;
    }
    if(vmax_b < 255-1)
    {
        vmax_b = vmax_b+1;
    }
    while(histo_g[vmax_g-1]>(N-((N/100)*s2)))
    {
        vmax_g = vmax_g-1;
    }
    if(vmax_g < 255-1)
    {
        vmax_g = vmax_g+1;
    }
    while(histo_r[vmax_r-1]>(N-((N/100)*s2)))
    {
        vmax_r = vmax_r-1;
    }
    if(vmax_r < 255-1)
    {
        vmax_r = vmax_r+1;
    }

    for(int i=0; i<simg.rows; i++)
    {
        for(int j=0; j<simg.cols; j++)
        {

            intensity = orig_img.at<cv::Vec3b>(i,j);

            if(intensity.val[0]<vmin_b)
            {
                intensity.val[0] = vmin_b;
            }
            if(intensity.val[0]>vmax_b)
            {
                intensity.val[0]=vmax_b;
            }


            if(intensity.val[1]<vmin_g)
            {
                intensity.val[1] = vmin_g;
            }
            if(intensity.val[1]>vmax_g)
            {
                intensity.val[1]=vmax_g;
            }

            if(intensity.val[2]<vmin_r)
            {
                intensity.val[2] = vmin_r;
            }
            if(intensity.val[2]>vmax_r)
            {
                intensity.val[2]=vmax_r;
            }

            orig_img.at<cv::Vec3b>(i,j) = intensity;
        }
    }

    for(int i=0; i<simg.rows; i++)
    {
        for(int j=0; j<simg.cols; j++)
        {

            intensity = orig_img.at<cv::Vec3b>(i,j);
            intensity.val[0] = (intensity.val[0] - vmin_b)*255/(vmax_b-vmin_b);
            intensity.val[1] = (intensity.val[1] - vmin_g)*255/(vmax_g-vmin_g);
            intensity.val[2] = (intensity.val[2] - vmin_r)*255/(vmax_r-vmin_r);
            orig_img.at<cv::Vec3b>(i,j) = intensity;
        }
    }


    cv::Mat blurred;
    double sigma = 1;
    double threshold = 50;
    double amount = 1;
    GaussianBlur(orig_img, blurred, cv::Size(), sigma, sigma);
    cv::Mat lowContrastMask = abs(orig_img - blurred) < threshold;
    cv::Mat sharpened = orig_img*(1+amount) + blurred*(-amount);
    orig_img.copyTo(sharpened, lowContrastMask);
    dst = sharpenelone();
}
 */


void HSV_calib(const cv::Mat img, int *thres, int mode) {
    // mode: 0 for red; 1 for green; 2 for blue;
    cv::Mat imgHSV;
    cv::cvtColor(img, imgHSV, cv::COLOR_BGR2HSV);

    cv::namedWindow("Control", cv::WINDOW_AUTOSIZE); //create a window called "Control"
    thres[0] = (mode == 0) ? 156 : ((mode == 1) ? 100 : 35);
    thres[1] = (mode == 0) ? 180 : ((mode == 1) ? 140 : 70);
    thres[2] = (mode == 0) ? 43 : ((mode == 1) ? 90 : 43);
    thres[3] = (mode == 0) ? 255 : ((mode == 1) ? 255 : 255);
    thres[4] = (mode == 0) ? 46 : ((mode == 1) ? 90 : 43);
    thres[5] = (mode == 0) ? 255 : ((mode == 1) ? 255 : 255);
    //Create trackbars in "Control" window
    cv::createTrackbar("LowH", "Control", &thres[0], 179); //Hue (0 - 179)
    cv::createTrackbar("HighH", "Control", &thres[1], 179);
    cv::createTrackbar("LowS", "Control", &thres[2], 255); //Saturation (0 - 255)
    cv::createTrackbar("HighS", "Control", &thres[3], 255);
    cv::createTrackbar("LowV", "Control", &thres[4], 255); //Value (0 - 255)
    cv::createTrackbar("HighV", "Control", &thres[5], 255);
    std::vector<cv::Mat> hsvSplit;
    //因为我们读取的是彩色图，直方图均衡化需要在HSV空间做
    cv::split(imgHSV, hsvSplit);

    cv::equalizeHist(hsvSplit[2], hsvSplit[2]);
    cv::merge(hsvSplit, imgHSV);
    cv::Mat imgThresholded;
    while (true) {
        cv::inRange(imgHSV, cv::Scalar(thres[0], thres[2], thres[4]), cv::Scalar(thres[1], thres[3], thres[5]),
                    imgThresholded); //Threshold the image

        //开操作 (去除一些噪点)
        cv::Mat element = getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::morphologyEx(imgThresholded, imgThresholded, cv::MORPH_OPEN, element);

        //闭操作 (连接一些连通域)
        cv::morphologyEx(imgThresholded, imgThresholded, cv::MORPH_CLOSE, element);

        cv::imshow("Thresholded Image", imgThresholded); //show the thresholded image
        cv::imshow("Original", img); //show the original image

        char key = (char) cv::waitKey(300);
        if (key == 27) {
            cv::destroyWindow("Control");
            break;
        } else continue;
    }
}

//CheckMode: 0代表去除黑区域，1代表去除白区域; NeihborMode：0代表4邻域，1代表8邻域;
void RemoveSmallRegion(cv::Mat &Src, cv::Mat &Dst, int AreaLimit, int CheckMode, int NeihborMode) {
    int RemoveCount = 0;       //记录除去的个数
    //记录每个像素点检验状态的标签，0代表未检查，1代表正在检查,2代表检查不合格（需要反转颜色），3代表检查合格或不需检查
    cv::Mat Pointlabel = cv::Mat::zeros(Src.size(), CV_8UC1);

    if (CheckMode == 1) {
//        std::cout << "Mode: 去除小区域. ";
        for (int i = 0; i < Src.rows; ++i) {
            uchar *iData = Src.ptr<uchar>(i);
            uchar *iLabel = Pointlabel.ptr<uchar>(i);
            for (int j = 0; j < Src.cols; ++j) {
                if (iData[j] < 10) {
                    iLabel[j] = 3;
                }
            }
        }
    } else {
//        std::cout << "Mode: 去除孔洞. ";
        for (int i = 0; i < Src.rows; ++i) {
            uchar *iData = Src.ptr<uchar>(i);
            uchar *iLabel = Pointlabel.ptr<uchar>(i);
            for (int j = 0; j < Src.cols; ++j) {
                if (iData[j] > 10) {
                    iLabel[j] = 3;
                }
            }
        }
    }

    std::vector<cv::Point2i> NeihborPos;  //记录邻域点位置
    NeihborPos.push_back(cv::Point2i(-1, 0));
    NeihborPos.push_back(cv::Point2i(1, 0));
    NeihborPos.push_back(cv::Point2i(0, -1));
    NeihborPos.push_back(cv::Point2i(0, 1));
    if (NeihborMode == 1) {
//        std::cout << "Neighbor mode: 8邻域." << std::endl;
        NeihborPos.push_back(cv::Point2i(-1, -1));
        NeihborPos.push_back(cv::Point2i(-1, 1));
        NeihborPos.push_back(cv::Point2i(1, -1));
        NeihborPos.push_back(cv::Point2i(1, 1));
    }// else std::cout << "Neighbor mode: 4邻域." << std::endl;
    int NeihborCount = 4 + 4 * NeihborMode;
    int CurrX = 0, CurrY = 0;
    //开始检测
    for (int i = 0; i < Src.rows; ++i) {
        uchar *iLabel = Pointlabel.ptr<uchar>(i);
        for (int j = 0; j < Src.cols; ++j) {
            if (iLabel[j] == 0) {
                //********开始该点处的检查**********
                std::vector<cv::Point2i> GrowBuffer;                                      //堆栈，用于存储生长点
                GrowBuffer.push_back(cv::Point2i(j, i));
                Pointlabel.at<uchar>(i, j) = 1;
                int CheckResult = 0;                                               //用于判断结果（是否超出大小），0为未超出，1为超出

                for (int z = 0; z < GrowBuffer.size(); z++) {

                    for (int q = 0; q < NeihborCount; q++)                                      //检查四个邻域点
                    {
                        CurrX = GrowBuffer.at(z).x + NeihborPos.at(q).x;
                        CurrY = GrowBuffer.at(z).y + NeihborPos.at(q).y;
                        if (CurrX >= 0 && CurrX < Src.cols && CurrY >= 0 && CurrY < Src.rows)  //防止越界
                        {
                            if (Pointlabel.at<uchar>(CurrY, CurrX) == 0) {
                                GrowBuffer.push_back(cv::Point2i(CurrX, CurrY));  //邻域点加入buffer
                                Pointlabel.at<uchar>(CurrY, CurrX) = 1;           //更新邻域点的检查标签，避免重复检查
                            }
                        }
                    }

                }
                if (GrowBuffer.size() > AreaLimit) CheckResult = 2;                 //判断结果（是否超出限定的大小），1为未超出，2为超出
                else {
                    CheckResult = 1;
                    RemoveCount++;
                }
                for (int z = 0; z < GrowBuffer.size(); z++)                         //更新Label记录
                {
                    CurrX = GrowBuffer.at(z).x;
                    CurrY = GrowBuffer.at(z).y;
                    Pointlabel.at<uchar>(CurrY, CurrX) += CheckResult;
                }
                //********结束该点处的检查**********


            }
        }
    }

    CheckMode = 255 * (1 - CheckMode);
    //开始反转面积过小的区域
    for (int i = 0; i < Src.rows; ++i) {
        uchar *iData = Src.ptr<uchar>(i);
        uchar *iDstData = Dst.ptr<uchar>(i);
        uchar *iLabel = Pointlabel.ptr<uchar>(i);
        for (int j = 0; j < Src.cols; ++j) {
            if (iLabel[j] == 2) {
                iDstData[j] = CheckMode;
            } else if (iLabel[j] == 3) {
                iDstData[j] = iData[j];
            }
        }
    }

//    std::cout << RemoveCount << " objects removed." << std::endl;
}

void find_apple() {
    std::cout << "Hello, World!" << std::endl;

//    std::string filename = "../apple.png";
//    HSV_calib(filename);

    cv::Mat img = cv::imread("../apple.png");
    cv::Mat imgHSV;
    cv::cvtColor(img, imgHSV, cv::COLOR_BGR2HSV);

    int iLowH = 156;//0-21 156-180
    int iHighH = 180;
    int iLowS = 43;//43-255
    int iHighS = 255;
    int iLowV = 46;//46-255
    int iHighV = 255;

    cv::Mat imgThresholded1, imgThresholded2;
    cv::inRange(imgHSV, cv::Scalar(iLowH, iLowS, iLowV), cv::Scalar(iHighH, iHighS, iHighV),
                imgThresholded1); //Threshold the image

    //开操作 (去除一些噪点)
    cv::Mat element = getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(imgThresholded1, imgThresholded1, cv::MORPH_OPEN, element);

    //闭操作 (连接一些连通域)
    cv::morphologyEx(imgThresholded1, imgThresholded1, cv::MORPH_CLOSE, element);

    iLowH = 0;
    iHighH = 21;
    iLowS = 148;
    cv::inRange(imgHSV, cv::Scalar(iLowH, iLowS, iLowV), cv::Scalar(iHighH, iHighS, iHighV),
                imgThresholded2); //Threshold the image

    //开操作 (去除一些噪点)
    cv::morphologyEx(imgThresholded2, imgThresholded2, cv::MORPH_OPEN, element);

    //闭操作 (连接一些连通域)
    cv::morphologyEx(imgThresholded2, imgThresholded2, cv::MORPH_CLOSE, element);

    cv::imshow("Thresholded Image1", imgThresholded1); //show the thresholded image
    cv::imshow("Thresholded Image2", imgThresholded2); //show the thresholded image
    cv::Mat imgThresholded;
    imgThresholded = ((imgThresholded1 + imgThresholded2) > 0);
    RemoveSmallRegion(imgThresholded, imgThresholded, 5000, 0, 0);
    RemoveSmallRegion(imgThresholded, imgThresholded, 1000, 1, 1);
    cv::imshow("Thresholded Image", imgThresholded); //show the thresholded image
    cv::imshow("Original", img); //show the original image

    cv::waitKey(0);
//    cv::Mat enhance;
//    cv::Mat hsv2;
//    colorEnhancement(img, enhance, 5);
//    cv::cvtColor(enhance, hsv2, cv::COLOR_BGR2HSV);

//    cv::imshow("test", img);
//    cv::imshow("test2", imgHSV);
//    cv::imshow("test3", hsv2);
//    cv::waitKey(0);

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    findContours(imgThresholded, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
    std::vector<cv::Rect> boundRect(contours.size());
    cv::Mat srcImage = img.clone();
    for (int i = 0; i < contours.size(); i++) {
        boundRect[i] = boundingRect(cv::Mat(contours[i]));
        rectangle(srcImage, boundRect[i].tl(), boundRect[i].br(), (0, 0, 255), 2, 8, 0);
        //rectangle(srcImage, boundRect[i].tl(), boundRect[i].br(), (255, 0, 0), 2, 8, 0);
        //rectangle(srcImage,rect,(255, 0, 0), 2, 8, 0);
    }
    cv::imshow("srcImage", srcImage);
    cv::waitKey(0);
    printf("Question1 finished.");
}

void txtWrite(int **hsv){
    FILE *fpWrite=fopen("../data.txt","w");
    if(fpWrite==NULL)
    {
        return;
    }
    for(int i=0; i<3; i++)
        for(int j=0; j < 6; ++j)
            fprintf(fpWrite,"%d ",hsv[i][j]);
    fclose(fpWrite);
}

void txtRead(int **hsv){
    FILE *fpRead=fopen("../data.txt","r");
    if(fpRead==NULL) return;
    for (int i=0; i<3; i++){
        for (int j=0; j < 6; ++j)
            fscanf(fpRead,"%d ", &hsv[i][j]);
//        printf("%d ",hsv[i][j]);
    }
}

void qsort(int s, int e, int op, int a[][2]){
    if(s>=e) return;
    int s1=s, e1=e;
    int tmp[2]={a[s][0],a[s][1]};
    do{
        while(a[e1][op]>=tmp[op] && s1<e1) --e1;
        if(s1<e1) {a[s1][0]=a[e1][0]; a[s1][1]=a[e1][1]; ++s1;}
        while(a[s1][op]<=tmp[op] && s1<e1) ++s1;
        if(s1<e1) {a[e1][0]=a[s1][0]; a[e1][1]=a[s1][1]; --e1;}
    }while(s1!=e1);
    a[s1][0]=tmp[0];
    a[s1][1]=tmp[1];
    qsort(s, s1-1,op,a);
    qsort(s1+1, e,op,a);
}