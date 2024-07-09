#include "fastdeploy/runtime.h"
#include "opencv2/opencv.hpp"

namespace fd = fastdeploy;

int main(int argc, char* argv){

    std::string model_file = R"(E:\le_ppocr\test01\models\mobilenetv2\inference.pdmodel)";
    std::string params_file = R"(E:\le_ppocr\test01\models\mobilenetv2\inference.pdiparams)";

    fd::RuntimeOption runtime_option;
    runtime_option.SetModelPath(model_file, params_file, fd::ModelFormat::PADDLE);
    runtime_option.UseOrtBackend();
    runtime_option.SetCpuThreadNum(6);

    std::unique_ptr<fd::Runtime> runtime = std::unique_ptr<fd::Runtime>(new fd::Runtime());
    if(!runtime->Init(runtime_option)){
        std::cerr<<"----Init FastDeploy Runtime Failed!" << "\n----- Model: " << model_file << std::endl;
        return -1;
    }
    else{
        std::cout << "----- Init FastDeploy Runtime Done!" << "\n--- Model: " << model_file << std::endl;
    }
    fd::TensorInfo info = runtime->GetInputInfo(0);
    info.shape={1,3,224,224};
    std::vector<fd::FDTensor> input_tensors(1);
    std::vector<fd::FDTensor> output_tensors(1);

    std::string img_pth = R"(E:\le_ppocr\test01\test_images\download.jfif)";
    cv::Mat img = cv::imread(img_pth);
    cv::Mat inputs_data = cv::dnn::blobFromImage(img, 1.0/255.0, cv::Size(224,224), 0.0, true, false, CV_32F);

    // std::vector<float> inputs_data;
    // inputs_data.resize(1*3*224*224);
    // for(size_t i=0; i<inputs_data.size(); ++i){
    //     inputs_data[i] = std::rand()%1000/1000.0f;
    // }

    input_tensors[0].SetExternalData({1,3,224,224}, fd::FDDataType::FP32, inputs_data.data);
    input_tensors[0].name = info.name;
    runtime->Infer(input_tensors, &output_tensors);
    output_tensors[0].PrintInfo();
    const float* data_ptr = reinterpret_cast<const float*>(output_tensors[0].GetData());
    std::vector<double> scores;
    for(size_t i=0; i<output_tensors[0].Numel();++i){
        std::cout << i << " ---- score: " << data_ptr[i] << "\n";
        scores.push_back(data_ptr[i]);
    }
    std::cout << std::endl;

    
    double max_val{0.0};
    cv::Point max_pos;

    cv::minMaxLoc(scores, 0, &max_val, 0, &max_pos);
    std::cout << "max_val: " << max_val << std::endl;
    std::cout << "max_Loc: " << max_pos.x << " " << max_pos.y << std::endl;

    return 0;
}