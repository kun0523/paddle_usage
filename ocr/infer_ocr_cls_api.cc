#include "fastdeploy/vision.h"
#ifdef WIN32
const char sep = '\\';
#else
const char sep = '/';
#endif

// 二分类  0 正向  1 180度颠倒

void InitAndInfer(const std::string &cls_model_dir,
                  const std::string &image_file,
                  const fastdeploy::RuntimeOption &option) {
  auto cls_model_file = cls_model_dir + sep + "inference.pdmodel";
  auto cls_params_file = cls_model_dir + sep + "inference.pdiparams";
  auto cls_option = option;

  auto cls_model = fastdeploy::vision::ocr::Classifier(
      cls_model_file, cls_params_file, cls_option);
  assert(cls_model.Initialized());

  // Parameters settings for pre and post processing of Cls Model.
  cls_model.GetPostprocessor().SetClsThresh(0.9);

  auto im = cv::imread(image_file);
  auto im_bak = im.clone();

  fastdeploy::vision::OCRResult result;
  if (!cls_model.Predict(im, &result)) {
    std::cerr << "Failed to predict." << std::endl;
    return;
  }

  // User can infer a batch of images by following code.
  // if (!cls_model.BatchPredict({im}, &result)) {
  //   std::cerr << "Failed to predict." << std::endl;
  //   return;
  // }

  std::cout << result.Str() << std::endl;
}

int main(int argc, char *argv[]) {
  if (argc < 4) {
    std::cout << "Usage: infer_demo path/to/cls_model path/to/image "
                 "run_option, "
                 "e.g ./infer_demo ./ch_ppocr_mobile_v2.0_cls_infer ./12.jpg 0"
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

  std::string cls_model_dir = argv[1];
  std::string test_image = argv[2];
  InitAndInfer(cls_model_dir, test_image, option);
  return 0;
}