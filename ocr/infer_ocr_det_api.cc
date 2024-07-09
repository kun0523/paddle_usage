#include "fastdeploy/vision.h"
#ifdef WIN32
const char sep = '\\';
#else
const char sep = '/';
#endif

void InitAndInfer(const std::string &det_model_dir,
                  const std::string &image_file,
                  const fastdeploy::RuntimeOption &option) {
  auto det_model_file = det_model_dir + sep + "inference.pdmodel";
  auto det_params_file = det_model_dir + sep + "inference.pdiparams";
  auto det_option = option;

  auto det_model = fastdeploy::vision::ocr::DBDetector(
      det_model_file, det_params_file, det_option);
  assert(det_model.Initialized());

  // Parameters settings for pre and post processing of Det Model.
  det_model.GetPreprocessor().SetMaxSideLen(960);
  det_model.GetPostprocessor().SetDetDBThresh(0.3);
  det_model.GetPostprocessor().SetDetDBBoxThresh(0.6);
  det_model.GetPostprocessor().SetDetDBUnclipRatio(1.5);
  det_model.GetPostprocessor().SetDetDBScoreMode("slow");
  det_model.GetPostprocessor().SetUseDilation(0);

  auto im = cv::imread(image_file);
  auto im_bak = im.clone();

  fastdeploy::vision::OCRResult result;
  if (!det_model.Predict(im, &result)) {
    std::cerr << "Failed to predict." << std::endl;
    return;
  }

  std::cout << result.Str() << std::endl;

  auto vis_im = fastdeploy::vision::VisOcr(im_bak, result);
  cv::imwrite("vis_result.jpg", vis_im);
  std::cout << "Visualized result saved in ./vis_result.jpg" << std::endl;
}

int main(int argc, char *argv[]) {
  if (argc < 4) {
    std::cout << "Usage: infer_demo path/to/det_model path/to/image "
                 "run_option, "
                 "e.g ./infer_demo ./ch_PP-OCRv3_det_infer ./12.jpg 0"
              << std::endl;
    std::cout << "The data type of run_option is int, 0: run with cpu; 1: run "
                 "with gpu;."
              << std::endl;
    return -1;
  }

  fastdeploy::RuntimeOption option;
  int flag = std::atoi(argv[3]);

  if (flag == 0) {
    option.UseCpu();
  } else if (flag == 1) {
    option.UseGpu();
  }

  std::string det_model_dir = argv[1];
  std::string test_image = argv[2];
  InitAndInfer(det_model_dir, test_image, option);
  return 0;
}