#include "fastdeploy/vision.h"

#ifdef WIN32
const char sep = '\\';
#else
const char sep = '/';
#endif

void CpuInfer(const std::string& model_dir, const std::string& image_file) {
  auto model_file = model_dir + sep + "model.pdmodel";
  auto params_file = model_dir + sep + "model.pdiparams";
  auto config_file = model_dir + sep + "deploy.yaml";
  auto option = fastdeploy::RuntimeOption();
  option.UseCpu();
  auto model = fastdeploy::vision::segmentation::PaddleSegModel(
      model_file, params_file, config_file, option);

  if (!model.Initialized()) {
    std::cerr << "Failed to initialize." << std::endl;
    return;
  }

  auto im = cv::imread(image_file);

  fastdeploy::vision::SegmentationResult res;
  if (!model.Predict(im, &res)) {
    std::cerr << "Failed to predict." << std::endl;
    return;
  }

  std::cout << res.Str() << std::endl;
  auto vis_im = fastdeploy::vision::VisSegmentation(im, res, 0.5);
  cv::imwrite("vis_result.jpg", vis_im);
  std::cout << "Visualized result saved in ./vis_result.jpg" << std::endl;
}

void GpuInfer(const std::string& model_dir, const std::string& image_file) {
  auto model_file = model_dir + sep + "model.pdmodel";
  auto params_file = model_dir + sep + "model.pdiparams";
  auto config_file = model_dir + sep + "deploy.yaml";

  auto option = fastdeploy::RuntimeOption();
  option.UseGpu();
  auto model = fastdeploy::vision::segmentation::PaddleSegModel(
      model_file, params_file, config_file, option);

  if (!model.Initialized()) {
    std::cerr << "Failed to initialize." << std::endl;
    return;
  }

  auto im = cv::imread(image_file);

  fastdeploy::vision::SegmentationResult res;
  if (!model.Predict(im, &res)) {
    std::cerr << "Failed to predict." << std::endl;
    return;
  }

  std::cout << res.Str() << std::endl;
  auto vis_im = fastdeploy::vision::VisSegmentation(im, res, 0.5);
  cv::imwrite("vis_result.jpg", vis_im);
  std::cout << "Visualized result saved in ./vis_result.jpg" << std::endl;
}

void TrtInfer(const std::string& model_dir, const std::string& image_file) {
  auto model_file = model_dir + sep + "model.pdmodel";
  auto params_file = model_dir + sep + "model.pdiparams";
  auto config_file = model_dir + sep + "deploy.yaml";

  auto option = fastdeploy::RuntimeOption();
  option.UseGpu();
  option.UseTrtBackend();
  // If use original Tensorrt, not Paddle-TensorRT,
  // comment the following two lines
  option.EnablePaddleToTrt();
  option.EnablePaddleTrtCollectShape();
  option.SetTrtInputShape("x", {1, 3, 256, 256}, {1, 3, 1024, 1024},
                          {1, 3, 2048, 2048});

  auto model = fastdeploy::vision::segmentation::PaddleSegModel(
      model_file, params_file, config_file, option);

  if (!model.Initialized()) {
    std::cerr << "Failed to initialize." << std::endl;
    return;
  }

  auto im = cv::imread(image_file);

  fastdeploy::vision::SegmentationResult res;
  if (!model.Predict(im, &res)) {
    std::cerr << "Failed to predict." << std::endl;
    return;
  }

  std::cout << res.Str() << std::endl;
  auto vis_im = fastdeploy::vision::VisSegmentation(im, res, 0.5);
  cv::imwrite("vis_result.jpg", vis_im);
  std::cout << "Visualized result saved in ./vis_result.jpg" << std::endl;
}

int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cout
        << "Usage: infer_demo path/to/model_dir path/to/image run_option, "
           "e.g ./infer_model ./ppseg_model_dir ./test.jpeg 0"
        << std::endl;
    std::cout << "The data type of run_option is int, 0: run with cpu; 1: run "
                 "with gpu; 2: run with gpu and use tensorrt backend; 3: run "
                 "with kunlunxin."
              << std::endl;
    return -1;
  }

  if (std::atoi(argv[3]) == 0) {
    CpuInfer(argv[1], argv[2]);
  } else if (std::atoi(argv[3]) == 1) {
    GpuInfer(argv[1], argv[2]);
  } else if (std::atoi(argv[3]) == 2) {
    TrtInfer(argv[1], argv[2]);
  }
  return 0;
}