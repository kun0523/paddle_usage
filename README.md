# paddle_usage
use paddlepaddle to train and deploy models

- 预训练模型：
  - classification: `https://bj.bcebos.com/paddlehub/fastdeploy/MobileNetV3_small_x1_0_infer.tgz`
  - detection: `https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_300e_coco.tgz`
  - segmentation: `https://bj.bcebos.com/paddlehub/fastdeploy/PP_HumanSegV2_Mobile_192x192_with_argmax_infer.tgz`
  - ocr
    - det: `https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_det_infer.tar`
    - cls: `https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar`
    - rec: `https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar`
      - 字符识别字典: `https://gitee.com/paddlepaddle/PaddleOCR/raw/release/2.6/ppocr/utils/en_dict.txt`

- 部署时需要的模型文件：
  - 模型结构文件：`xxxx.pdmodel`
  - 模型参数文件：`xxxx.pdiparams`

- CPP可执行文件依赖项：
  - `fastdeploy_init.bat` 可自动将依赖项复制到可执行文件同级目录下
  - `fastdeploy_init.bat install Path/to/fastdeploy_root Path/to/infer.exe`
  - `E:\cpp_packages\FastDeploy\fastdeploy-win-x64-0.0.0\fastdeploy_init.bat install E:\cpp_packages\FastDeploy\fastdeploy-win-x64-0.0.0 E:\my_paddle\bin\Segmentation\Release`

- TODO:
  - 四个任务(cls det seg ocr) 训练过程说明
  - 还没搞懂，没有明确传 config.yaml 文件，模型前处理是怎么定义的？根据模型的类型固定的前处理逻辑？