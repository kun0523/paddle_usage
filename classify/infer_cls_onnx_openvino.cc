#include "fastdeploy/runtime.h"
#include <cassert>
#include <opencv2/opencv.hpp>

namespace fd = fastdeploy;

int main(int argc, char* argv[]) {
//   // Download from https://bj.bcebos.com/paddle2onnx/model_zoo/pplcnet.onnx
//   std::string model_file = R"(E:\my_paddle\models\pplcnet.onnx)";

//   // configure runtime
//   // How to configure by RuntimeOption, refer its api doc for more information
//   // https://baidu-paddle.github.io/fastdeploy-api/cpp/html/structfastdeploy_1_1RuntimeOption.html
//   fd::RuntimeOption runtime_option;
//   runtime_option.SetModelPath(model_file, "", fd::ModelFormat::ONNX);
// //   runtime_option.UseOpenVINOBackend();
//   runtime_option.UseOrtBackend();
//   runtime_option.EnableValidBackendCheck();
//   std::cout << "backend option: " << runtime_option.backend << std::endl;

    // Paddle model
    std::string model_file = R"(E:\my_paddle\models\mobilenetv2\inference.pdmodel)";
    std::string params_file = R"(E:\my_paddle\models\mobilenetv2\inference.pdiparams)";
    fd::RuntimeOption runtime_option;
    runtime_option.SetModelPath(model_file, params_file, fd::ModelFormat::PADDLE);

    // // onnx model
    // std::string model_file = R"(E:\my_paddle\models\best.onnx)";
    // fd::RuntimeOption runtime_option;
    // runtime_option.SetModelPath(model_file, "", fd::ModelFormat::ONNX);

    // runtime_option.UseOpenVINOBackend();
    runtime_option.UseOrtBackend();
    runtime_option.SetCpuThreadNum(12);
    std::unique_ptr<fd::Runtime> runtime = std::unique_ptr<fd::Runtime>(new fd::Runtime());
    if(!runtime->Init(runtime_option)){
        std::cerr << "---Init Fastdeploy Runtime Failed! " << "\n---Model: " << model_file << std::endl;
    }else{
        std::cout << "---Init FastDeploy Runtime Done!" << "\n--- Model: " << model_file << std::endl;
    }


  
//   // Use CPU to inference
//   // If need to configure OpenVINO backend for more option, we can configure runtime_option.openvino_option
//   // refer https://baidu-paddle.github.io/fastdeploy-api/cpp/html/structfastdeploy_1_1OpenVINOBackendOption.html
//   runtime_option.UseCpu();
//   runtime_option.SetCpuThreadNum(12);

//   fd::Runtime runtime;
//   assert(runtime.Init(runtime_option));
//   std::cout << model_file << std::endl;
//   fd::TensorInfo input_info = runtime->GetInputInfo(0);
//   std::cout << input_info.name << std::endl;

//   std::cout << "runtime option: " << runtime->option.backend << std::endl;
//   std::cout << "input num: " << 5 << std::endl;

  // Get model's inputs information
  // API doc refer https://baidu-paddle.github.io/fastdeploy-api/cpp/html/structfastdeploy_1_1Runtime.html
  fd::TensorInfo input_info = runtime->GetInputInfo(0); 
  std::cout << "inputs info: " <<  input_info.shape.at(2) << std::endl;

  // Create dummy data fill with 0.5
//   std::vector<float> dummy_data(1 * 3 * 224 * 224, 0.5);
    std::string img_pth = R"(E:\my_paddle\test_images\download.jfif)";
    // std::string img_pth = R"(E:\DataSets\vacuum_package\cut_roi\NG_unseal\2024_1_17\NG_unseal_1705451055.jpg)";
    cv::Mat img = cv::imread(img_pth);
    cv::Mat inputs_data = cv::dnn::blobFromImage(img, 1.0/255.0, cv::Size(224,224), 0.0, true, false, CV_32F);
    std::cout << "inputs size: " <<  inputs_data.size << std::endl;

  // Create inputs/outputs tensors
  fd::FDTensor inputs;
  std::vector<fd::FDTensor> outputs(1);

  // Initialize input tensors
  // API doc refer https://baidu-paddle.github.io/fastdeploy-api/cpp/html/structfastdeploy_1_1FDTensor.html
  inputs.SetExternalData({1, 3, 224, 224}, fd::FDDataType::FP32, inputs_data.data);
  inputs.name = input_info.name;

  // Inference
  assert(runtime->Infer(inputs, &outputs));  // 有问题！！输出值不对
  std::cout << "output num: " << outputs[0].Numel() << std::endl;
 
//   // Print debug information of outputs 
//   outputs.PrintInfo();   
//   std::cout << "-----------------" << std::endl;


//   // Get data pointer and print it's elements
//   const float* data_ptr = reinterpret_cast<const float*>(outputs.GetData());
//   for (size_t i = 0; i < outputs.Numel(); ++i) {
//     std::cout << data_ptr[i] << " ";
//   }
//   std::cout << std::endl;
  return 0;
}