#include "include/detection.h"

int main()
{
    std::string prototxt = "./model/deploy_512.prototxt";
    std::string model = "./model/68mAP-512-refinedet.plan";
    Detect detector(prototxt.c_str(), model.c_str(), 512, 512, 0.1);
    //Detect detector(prototxt.c_str(), model.c_str(), 320, 320, 0.3);

    cv::VideoCapture cap;
    cv::VideoWriter video("./out.avi", CV_FOURCC('M', 'J', 'P', 'G'), 60, cv::Size(1600, 900), true);

    //cap.open("rtsp://admin:youwillsee!@192.168.1.41");
	//cap.open("rtsp://admin:youwillsee2019@192.168.1.89");
	cap.open("/media/yehao/disk1/工作测试视频/密集人群/crowd4.mp4");    
	
    cv::Mat frame;
    while (true) {
        bool success = cap.read(frame);
        //frame = cv::imread("/home/yehao/Downloads/test.jpg");
        if (!success) {
            std::cout << "process current frame failure" << std::endl;
            continue;
        }
        detector.ImgInference(frame);
        cv::resize(frame, frame, cv::Size(1600, 900), (0, 0), (0, 0), cv::INTER_LINEAR);
        //video.write(frame);
		cv::imshow("demo", frame);
		cv::waitKey(10);
    }
    return 0;
}
