/**
 * @author sol
 * @version 1.0
 * @date 2022/03/06
*/

#include "lite/lite.h"
#include<ctime>

static void test_video() {
    std::string onnx_path = "../inputs/model/rvm_mobilenetv3_fp32.onnx";
    std::string video_path = "../inputs/video/TEST_01.mp4";
    std::string output_path = "../outputs/interview_onnx.mp4";

    auto *rvm = new lite::cv::matting::RobustVideoMatting(onnx_path, 1); // 16 threads
    std::vector<lite::types::MattingContent> contents;

    auto start = clock();
    rvm->detect_video(video_path, output_path, contents, false, 0.25f, 30);
    auto end = clock();
    printf("耗时：%lf\n", 1.0 * (end - start) / CLOCKS_PER_SEC);

    delete rvm;
}

static void test_image() {
    std::string onnx_path = "../inputs/model/rvm_mobilenetv3_fp32.onnx";
    std::string img_path = "../inputs/img/test.jpg";
    std::string save_fgr_path = "../outputs/test_rvm_fgr.jpg";
    std::string save_pha_path = "../outputs/test_rvm_pha.jpg";
    std::string save_merge_path = "../outputs/test_rvm_merge.jpg";

    auto *rvm = new lite::cv::matting::RobustVideoMatting(onnx_path, 1); // 16 threads
    lite::types::MattingContent content, content_2;
    cv::Mat img_bgr = cv::imread(img_path);

    rvm->detect(img_bgr, content, 0.25f);
    if (content.flag) {
        if (!content.fgr_mat.empty()) cv::imwrite(save_fgr_path, content.fgr_mat);
        if (!content.pha_mat.empty()) cv::imwrite(save_pha_path, content.pha_mat * 255.);
        if (!content.merge_mat.empty()) cv::imwrite(save_merge_path, content.merge_mat);
        std::cout << "Saved " << save_merge_path << "\n";
    }
    delete rvm;
}

int main() {
//    test_video();
    test_image();
    return 0;
}