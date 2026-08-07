#include "NvInferPlugin.h"
#include <fstream>
#include "include/Mask/common.hpp"

using namespace det;
class YOLOv8 {
public:
    explicit YOLOv8(const std::string& engine_file_path);
    ~YOLOv8();

    void                 make_pipe(bool warmup = true);
    void                 copy_from_Mat(const cv::Mat& image);
    void                 copy_from_Mat(const cv::Mat& image, cv::Size& size);
    void                 letterbox(const cv::Mat& image, cv::Mat& out, cv::Size& size);
    void                 infer();
    void                 postprocess(std::vector<Object>& objs);
    void                 postprocess_standard(std::vector<Object>& objs, float conf_thresh = 0.5f, float nms_thresh = 0.45f);
    bool                 is_end2end() const { return num_outputs == 4; }
    static void          draw_objects(const cv::Mat&                                image,
                                      cv::Mat&                                      res,
                                      const std::vector<Object>&                    objs,
                                      const std::vector<std::string>&               CLASS_NAMES,
                                      const std::vector<std::vector<unsigned int>>& COLORS);
    int                  num_bindings;
    int                  num_inputs  = 0;
    int                  num_outputs = 0;
    std::vector<Binding> input_bindings;
    std::vector<Binding> output_bindings;
    std::vector<void*>   host_ptrs;
    std::vector<void*>   device_ptrs;

    PreParam pparam;

private:
    nvinfer1::ICudaEngine*       engine  = nullptr;
    nvinfer1::IRuntime*          runtime = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    cudaStream_t                 stream  = nullptr;
    Logger                       gLogger{nvinfer1::ILogger::Severity::kERROR};
};