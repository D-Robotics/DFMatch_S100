#include <iostream>
#include <vector>
#include <chrono>
#include <map>
#include <cmath>
#include <unordered_map>
#include <filesystem>

#include "hobot/dnn/hb_dnn.h"
#include "hobot/dnn/hb_dnn_status.h"
#include "hobot/hb_ucp.h"
#include "hobot/hb_ucp_sys.h"

#include "opencv2/core/mat.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include "eigen3/Eigen/Core"
#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Geometry"

namespace fs = std::filesystem;

// ==================================================== Common ========================================================
#define HB_CHECK_SUCCESS(value, errmsg)                                                                                                                                                                \
    do                                                                                                                                                                                                 \
    {                                                                                                                                                                                                  \
        /*value can be call of function*/                                                                                                                                                              \
        auto ret_code = value;                                                                                                                                                                         \
        if (ret_code != 0)                                                                                                                                                                             \
        {                                                                                                                                                                                              \
            std::cout << "=> [BPU ERROR] " << errmsg << ", error code: " << ret_code << "!" << std::endl;                                                                                              \
            return ret_code;                                                                                                                                                                           \
        }                                                                                                                                                                                              \
    } while (0);

#define ALIGN(value, alignment) (((value) + ((alignment)-1)) & ~((alignment)-1))
#define ALIGN_32(value) ALIGN(value, 32)

int dnn_infer(hbUCPTaskHandle_t *taskHandle, 
              hbDNNTensor **output,
              hbDNNTensor const *input, 
              hbDNNHandle_t dnnHandle,
              hbUCPSchedParam *inferCtrlParam) {
    int ret = 0;
    ret = hbDNNInferV2(taskHandle, *output, input, dnnHandle);
    if (ret != HB_DNN_SUCCESS)
    {
        std::cout << "hbDNNInferV2 failed: "<< ret << std::endl;
        return ret;
    }
    inferCtrlParam->backend = HB_UCP_BPU_CORE_ANY;
    ret = hbUCPSubmitTask(*taskHandle, inferCtrlParam);
    if (ret != HB_DNN_SUCCESS)
    {
        std::cout << "hbUCPSubmitTask failed: "<< ret << std::endl;
        return ret;
    }
    return ret;
}

// ==================================================== DFeat Infer ===================================================
// img size must be 640x480, because the model input size is 640x480
int img_height = 480;
int img_width = 640;

hbDNNPackedHandle_t packed_dnn_handle_df;
hbUCPTaskHandle_t dnn_handle_df;
const char **model_name_list_df;
bool flag_init_df = false;
float point_th = 0.012;
int windowSize = 5;

struct KeyPoint
{
    int x, y;
    float score;
};

struct pair_hash
{
    template <class T1, class T2> size_t operator()(std::pair<T1, T2> const &pair) const
    {
        size_t h1 = std::hash<T1>()(pair.first);
        size_t h2 = std::hash<T2>()(pair.second);
        return h1 ^ h2;
    }
};

const float GRID_SIZE = 10.0f;

using Grid = std::unordered_map<std::pair<int, int>, std::vector<KeyPoint>, pair_hash>;

std::pair<int, int> hashKeyPoint(const KeyPoint &kp)
{
    int xHash = static_cast<int>(kp.x / GRID_SIZE);
    int yHash = static_cast<int>(kp.y / GRID_SIZE);
    return {xHash, yHash};
}

std::vector<cv::Point2f> applyNMS_grid(const std::vector<KeyPoint> &keypoints, float threshold, int windowSize)
{
    std::vector<cv::Point2f> nmsKeypoints;
    Grid grid;

    for (const auto &kp : keypoints)
    {
        auto key = hashKeyPoint(kp);
        grid[key].push_back(kp);
    }

    std::vector<bool> suppressed(keypoints.size(), false);

    for (size_t i = 0; i < keypoints.size(); ++i)
    {
        if (suppressed[i])
            continue;

        KeyPoint kp = keypoints[i];
        bool isLocalMaximum = true;

        auto key = hashKeyPoint(kp);
        for (int dx = -1; dx <= 1; ++dx)
        {
            for (int dy = -1; dy <= 1; ++dy)
            {
                auto neighborKey = std::make_pair(key.first + dx, key.second + dy);
                if (grid.find(neighborKey) != grid.end())
                {
                    for (const auto &neighbor : grid[neighborKey])
                    {
                        if (neighbor.x == kp.x && neighbor.y == kp.y)
                            continue;

                        float dist = std::sqrt(std::pow(kp.x - neighbor.x, 2) + std::pow(kp.y - neighbor.y, 2));
                        if (dist < windowSize && neighbor.score > kp.score - threshold)
                        {
                            isLocalMaximum = false;
                            break;
                        }
                    }
                }
                if (!isLocalMaximum)
                    break;
            }
            if (!isLocalMaximum)
                break;
        }

        if (isLocalMaximum)
        {
            nmsKeypoints.push_back(cv::Point2f(kp.x * 1.0, kp.y * 1.0));
        }
        else
        {
            suppressed[i] = true;
        }
    }

    return nmsKeypoints;
}

/** You can define read_image_2_tensor_as_other_type to prepare your data **/
int32_t read_image_2_tensor_as_gray_dfeat(cv::Mat &bgr_mat, hbDNNTensor *input_tensor)
{
    hbDNNTensor *input = input_tensor;
    hbDNNTensorProperties Properties = input->properties;
    int input_h = Properties.validShape.dimensionSize[2];
    int input_w = Properties.validShape.dimensionSize[3];
    if (input_h % 2 || input_w % 2)
    {
        std::cout << "=> input img height and width must aligned by 2!" << std::endl;
        return -1;
    }
    else
    {
        std::cout << "=> model input [height, width] is [" << input_h << ", " << input_w << "]" << std::endl;
    }

    cv::Mat gray_mat;
    cv::cvtColor(bgr_mat, gray_mat, cv::COLOR_BGR2GRAY);
    cv::Mat gray_float_mat;
    gray_mat.convertTo(gray_float_mat, CV_32FC1, 1.0 / 255.0);  // 转换并归一化

    auto data = input->sysMem.virAddr;
    int32_t data_size = input_h * input_w * sizeof(float);
    memcpy(data, gray_float_mat.data, data_size);

    return 0;
}

int prepare_tensor(hbDNNTensor *input_tensor, hbDNNTensor *output_tensor, hbDNNHandle_t dnn_handle)
{
    int input_count = 0;
    int output_count = 0;
    hbDNNGetInputCount(&input_count, dnn_handle);
    hbDNNGetOutputCount(&output_count, dnn_handle);
    /** Tips:
     * For input memory size:s
     * *   input_memSize = input[i].properties.alignedByteSize
     * For output memory size:
     * *   output_memSize = output[i].properties.alignedByteSize
     */
     hbDNNTensor *input = input_tensor;
     for (int i = 0; i < input_count; i++)
     {
        // Get the properties of the input tensor
        HB_CHECK_SUCCESS(hbDNNGetInputTensorProperties(&input[i].properties, dnn_handle, i), "hbDNNGetInputTensorProperties failed");
        // Calculate the stride of the input tensor
        auto dim_len = input[i].properties.validShape.numDimensions;
        for (int32_t dim_i = dim_len - 1; dim_i >= 0; --dim_i)
        {
            if (input[i].properties.stride[dim_i] == -1)
            {
                auto cur_stride = input[i].properties.stride[dim_i + 1] * input[i].properties.validShape.dimensionSize[dim_i + 1];
                input[i].properties.stride[dim_i] = ALIGN_32(cur_stride);
                std::cout << "=> input[" << i << "] dim " << dim_i << " stride is " << input[i].properties.stride[dim_i] << std::endl;
            }
        }
        // Calculate the memory size of the input tensor and allocate cache memory
        int input_memSize = input[i].properties.stride[0] * input[i].properties.validShape.dimensionSize[0];
        HB_CHECK_SUCCESS(hbUCPMallocCached(&input[i].sysMem, input_memSize, 0), "hbUCPMallocCached failed");

        // Show how to get input name
        const char *input_name;
        HB_CHECK_SUCCESS(hbDNNGetInputName(&input_name, dnn_handle, i), "hbDNNGetInputName failed");
        std::cout << "=> input[" << i << "] name is " << input_name << ", memsize is " << input_memSize << std::endl;
    }

    hbDNNTensor *output = output_tensor;
    for (int i = 0; i < output_count; i++)
    {
        // Get the properties of the output tensor
        HB_CHECK_SUCCESS(hbDNNGetOutputTensorProperties(&output[i].properties, dnn_handle, i), "hbDNNGetOutputTensorProperties failed");
        // Calculate the memory size of the output tensor and allocate cache memory
        int output_memSize = output[i].properties.alignedByteSize;
        // if (output_memSize == 4096) output_memSize = 307200;   // 307200 / 4096 = 75
        // if (output_memSize == 1024) output_memSize = 78643200; // 78643200 / 1024 = 76800
        HB_CHECK_SUCCESS(hbUCPMallocCached(&output[i].sysMem, output_memSize, 0), "hbUCPMallocCached failed");

        // Show how to get output name
        const char *output_name;
        HB_CHECK_SUCCESS(hbDNNGetOutputName(&output_name, dnn_handle, i), "hbDNNGetOutputName failed");
        std::cout << "=> output[" << i << "] name is " << output_name << ", memsize is " << output_memSize << std::endl;
    }
    return 0;
}

int infer_s100_dfeat(cv::Mat &bgr_mat, std::pair<std::vector<cv::Point2f>, Eigen::MatrixXd> &extractor_result)
{

    // ==================================================== load model ====================================================
    if (flag_init_df == false)
    {
        flag_init_df = true;

        std::string model_name = "../df.hbm";

        auto modelFileName = model_name.c_str();
        int model_count = 0;
        // Step1: get model handle
        {
            HB_CHECK_SUCCESS(hbDNNInitializeFromFiles(&packed_dnn_handle_df, &modelFileName, 1), "hbDNNInitializeFromFiles failed");
            HB_CHECK_SUCCESS(hbDNNGetModelNameList(&model_name_list_df, &model_count, packed_dnn_handle_df), "hbDNNGetModelNameList failed");
            HB_CHECK_SUCCESS(hbDNNGetModelHandle(&dnn_handle_df, packed_dnn_handle_df, model_name_list_df[0]), "hbDNNGetModelHandle failed");
        }
        // Show how to get dnn version
        std::cout << "=> Load Moldel:" << model_name << " success, DNN runtime version: " << hbDNNGetVersion() << std::endl;
    }

    // ==================================================== prepare input and output tensor ===============================
    // Step2: prepare input and output tensor
    std::vector<hbDNNTensor> input_tensors;
    std::vector<hbDNNTensor> output_tensors;
    int input_count = 0;
    int output_count = 0;
    {
        HB_CHECK_SUCCESS(hbDNNGetInputCount(&input_count, dnn_handle_df), "hbDNNGetInputCount failed");
        HB_CHECK_SUCCESS(hbDNNGetOutputCount(&output_count, dnn_handle_df), "hbDNNGetOutputCount failed");
        input_tensors.resize(input_count);
        output_tensors.resize(output_count);
        prepare_tensor(input_tensors.data(), output_tensors.data(), dnn_handle_df);
    }

    // ==================================================== set input data to input tensor ================================
    // Step3: set input data to input tensor
    {
        // read a single picture for input_tensor[0], for multi_input model, you
        // should set other input data according to model input properties.
        HB_CHECK_SUCCESS(read_image_2_tensor_as_gray_dfeat(bgr_mat, input_tensors.data()), "read_image_2_tensor_as_nv12 failed");
    }

    // ==================================================== run infer =====================================================
    // Step4: run inference
    auto t1 = std::chrono::steady_clock::now();
    hbUCPTaskHandle_t task_handle = nullptr;
    hbDNNTensor *output = output_tensors.data();
    {
        // make sure memory data is flushed to DDR before inference
        for (int i = 0; i < input_count; i++)
        {
            hbUCPMemFlush(&input_tensors[i].sysMem, HB_SYS_MEM_CACHE_CLEAN);
        }

        hbUCPSchedParam infer_ctrl_param;
        HB_UCP_INITIALIZE_SCHED_PARAM(&infer_ctrl_param);
        HB_CHECK_SUCCESS(dnn_infer(&task_handle, &output, input_tensors.data(), dnn_handle_df, &infer_ctrl_param), "dnn_infer failed");
        // wait task done
        HB_CHECK_SUCCESS(hbUCPWaitTaskDone(task_handle, 0), "hbUCPWaitTaskDone failed");
    }
    auto t2 = std::chrono::steady_clock::now();
    auto time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count() * 1000;
    std::cout << "\033[31m" << "=> dfeat infer Time: " << time_used << "ms\033[0m" << std::endl;

    // ==================================================== do postprocess with output data ===============================
    // Step5: do postprocess with output data
    {
        // make sure CPU read data from DDR before using output tensor data
        for (int i = 0; i < output_count; i++)
        {
            hbUCPMemFlush(&output_tensors[i].sysMem, HB_SYS_MEM_CACHE_INVALIDATE);
        }
        std::cout << "=> output_count: " << output_count << std::endl;

        int *shape = output->properties.validShape.dimensionSize;
        int tensor_len = shape[0] * shape[1] * shape[2] * shape[3];
        std::cout << "=> tensor 0 len " << tensor_len << ", shape " << shape[0] << " * " << shape[1] << " * " << shape[2] << " *  " << shape[3] << std::endl;

        int *shape_1 = output_tensors[1].properties.validShape.dimensionSize;
        int tensor_len_1 = shape_1[0] * shape_1[1] * shape_1[2] * shape_1[3];
        std::cout << "=> tensor 1 len " << tensor_len_1 << ", shape " << shape_1[0] << " * " << shape_1[1] << " * " << shape_1[2] << " *  " << shape_1[3] << std::endl;

        auto semi = reinterpret_cast<int8_t *>(output_tensors[0].sysMem.virAddr);
        auto desc = reinterpret_cast<int8_t *>(output_tensors[1].sysMem.virAddr);

        // Extracting KeyPoints
        int res_count = 0;
        float scale_semi = *output_tensors[0].properties.scale.scaleData;
        std::cout << "=> scale_semi: " << scale_semi << std::endl;
        std::vector<KeyPoint> keypoints_all;
        for (int i = 0; i < tensor_len; i++)
        {
            float tmp_dequant = static_cast<float>(semi[i]) * scale_semi;

            if (tmp_dequant >= point_th)
            {
                res_count++;
                int kp_x = i % 640;
                int kp_y = i / 640;

                KeyPoint cur_kp;
                cur_kp.x = kp_x;
                cur_kp.y = kp_y;
                cur_kp.score = tmp_dequant;
                keypoints_all.emplace_back(cur_kp);
            }
        }
        float threshold = 0.0;
        std::vector<cv::Point2f> nmsKeypoints = applyNMS_grid(keypoints_all, threshold, windowSize);
        
        auto t3 = std::chrono::steady_clock::now();
        auto time_used_2 = std::chrono::duration_cast<std::chrono::duration<double>>(t3 - t2).count() * 1000;
        std::cout << "\033[31m" << "=> dfeat postprocess nms Time: " << time_used_2 << "ms\033[0m" << std::endl;

        // Parsing Descriptors
        float scale_desc = *output_tensors[1].properties.scale.scaleData;
        std::cout << "=> scale_desc: " << scale_desc << std::endl;
        int desc_dim = 256;
        Eigen::MatrixXf matMatrixXd(nmsKeypoints.size(), desc_dim);
        for (size_t i = 0; i < nmsKeypoints.size(); i++)
        {
            int cur_x = nmsKeypoints[i].x;
            int cur_y = nmsKeypoints[i].y;

            Eigen::VectorXf cur_desc = Eigen::VectorXf::Zero(desc_dim);

            for (int j = 0; j < desc_dim; j++)
            {
                cur_desc[j] = static_cast<float>(desc[(cur_y * 640 + cur_x) * desc_dim + j]) * scale_desc;
            }

            float norm_sqrt = cur_desc.norm();

            Eigen::VectorXf cur_desc_normalized = cur_desc / norm_sqrt;

            matMatrixXd.row(i) = cur_desc_normalized;
        }

        auto t4 = std::chrono::steady_clock::now();
        auto time_used_3 = std::chrono::duration_cast<std::chrono::duration<double>>(t4 - t3).count() * 1000;
        std::cout << "\033[31m" << "=> dfeat postprocess desc Time: " << time_used_3 << "ms\033[0m" << std::endl;

        Eigen::MatrixXd desc_double(matMatrixXd.cast<double>());

        extractor_result.first = nmsKeypoints;
        extractor_result.second = desc_double;
    }

    // ==================================================== release resources =============================================
    // Step6: release resources
    {
        // release task handle
        HB_CHECK_SUCCESS(hbUCPReleaseTask(task_handle), "hbUCPReleaseTask failed");
        // free input mem
        for (int i = 0; i < input_count; i++)
        {
            HB_CHECK_SUCCESS(hbUCPFree(&(input_tensors[i].sysMem)), "hbUCPFree failed");
        }
        // free output mem
        for (int i = 0; i < output_count; i++)
        {
            HB_CHECK_SUCCESS(hbUCPFree(&(output_tensors[i].sysMem)), "hbUCPFree failed");
        }
    }

    return 0;
}

// ==================================================== LightGlue Infer ===============================================
hbDNNPackedHandle_t packed_dnn_handle_lg;
hbUCPTaskHandle_t dnn_handle_lg;
const char **model_name_list_lg;
bool flag_init_lg = false;

float match_score_th = 0.1f;

std::vector<cv::Point2f> NormalizeKeypoints(std::vector<cv::Point2f> kpts, int h, int w)
{
    cv::Size size(w, h);
    cv::Point2f shift(static_cast<float>(w) / 2, static_cast<float>(h) / 2);
    float scale = static_cast<float>((std::max)(w, h)) / 2;

    std::vector<cv::Point2f> normalizedKpts;
    for (const cv::Point2f &kpt : kpts)
    {
        cv::Point2f normalizedKpt = (kpt - shift) / scale;
        normalizedKpts.push_back(normalizedKpt);
    }

    return normalizedKpts;
}

int infer_s100_lg(std::vector<cv::Point2f> &keypoint_1, Eigen::MatrixXd &desc_1, std::vector<cv::Point2f> &keypoint_2, Eigen::MatrixXd &desc_2, std::vector<cv::Point2f> &match_kp_1, std::vector<cv::Point2f> &match_kp_2)
{
    // ==================================================== load model ====================================================
    if (flag_init_lg == false)
    {
        flag_init_lg = true;

        std::string model_name = "../lg.hbm";

        auto modelFileName = model_name.c_str();
        int model_count = 0;
        // Step1: get model handle
        {
            HB_CHECK_SUCCESS(hbDNNInitializeFromFiles(&packed_dnn_handle_lg, &modelFileName, 1), "hbDNNInitializeFromFiles failed");
            HB_CHECK_SUCCESS(hbDNNGetModelNameList(&model_name_list_lg, &model_count, packed_dnn_handle_lg), "hbDNNGetModelNameList failed");
            HB_CHECK_SUCCESS(hbDNNGetModelHandle(&dnn_handle_lg, packed_dnn_handle_lg, model_name_list_lg[0]), "hbDNNGetModelHandle failed");
        }
        // Show how to get dnn version
        std::cout << "=> Load Moldel:" << model_name << " success, DNN runtime version: " << hbDNNGetVersion() << std::endl;
    }

    // ==================================================== prepare input and output tensor ===============================
    // Step2: prepare input and output tensor
    std::vector<hbDNNTensor> input_tensors;
    std::vector<hbDNNTensor> output_tensors;
    int input_count = 0;
    int output_count = 0;
    {
        HB_CHECK_SUCCESS(hbDNNGetInputCount(&input_count, dnn_handle_lg), "hbDNNGetInputCount failed");
        HB_CHECK_SUCCESS(hbDNNGetOutputCount(&output_count, dnn_handle_lg), "hbDNNGetOutputCount failed");
        input_tensors.resize(input_count);
        output_tensors.resize(output_count);
        prepare_tensor(input_tensors.data(), output_tensors.data(), dnn_handle_lg);
    }

    // ==================================================== set input data to input tensor ================================
    // Step3: set input data to input tensor
    {
        auto kpts1 = NormalizeKeypoints(keypoint_1, img_height, img_width);
        auto kpts2 = NormalizeKeypoints(keypoint_2, img_height, img_width);
        std::cout << "=> kpts1 size: " << kpts1.size() << ", kpts2 size: " << kpts2.size() << std::endl;
        float *kpts1_data = new float[kpts1.size() * 2];
        float *kpts2_data = new float[kpts2.size() * 2];
        for (size_t i = 0; i < kpts1.size(); ++i)
        {
            kpts1_data[i * 2] = kpts1[i].x;
            kpts1_data[i * 2 + 1] = kpts1[i].y;
        }
        for (size_t i = 0; i < kpts2.size(); ++i)
        {
            kpts2_data[i * 2] = kpts2[i].x;
            kpts2_data[i * 2 + 1] = kpts2[i].y;
        }

        Eigen::MatrixXf desc_1_float = desc_1.cast<float>();
        Eigen::MatrixXf desc_2_float = desc_2.cast<float>();
        Eigen::MatrixXf desc_1_float_trans = desc_1_float.transpose();
        Eigen::MatrixXf desc_2_float_trans = desc_2_float.transpose();
        float *desc1 = desc_1_float_trans.data();
        float *desc2 = desc_2_float_trans.data();

        int *kp1_shape = input_tensors[0].properties.validShape.dimensionSize;
        int kp1_tensor_len = kp1_shape[0] * kp1_shape[1] * kp1_shape[2];
        std::cout << "=> kp1 tensor len " << kp1_tensor_len << ", shape " << kp1_shape[0] << " * " << kp1_shape[1] << " *  " << kp1_shape[2] << std::endl;

        int *kp2_shape = input_tensors[1].properties.validShape.dimensionSize;
        int kp2_tensor_len = kp2_shape[0] * kp2_shape[1] * kp2_shape[2];
        std::cout << "=> kp2 tensor len " << kp2_tensor_len << ", shape " << kp2_shape[0] << " * " << kp2_shape[1] << " *  " << kp2_shape[2] << std::endl;

        int *desc1_shape = input_tensors[2].properties.validShape.dimensionSize;
        int desc1_tensor_len = desc1_shape[0] * desc1_shape[1] * desc1_shape[2];
        std::cout << "=> desc1 tensor len " << desc1_tensor_len << ", shape " << desc1_shape[0] << " * " << desc1_shape[1] << " *  " << desc1_shape[2] << std::endl;

        int *desc2_shape = input_tensors[3].properties.validShape.dimensionSize;
        int desc2_tensor_len = desc2_shape[0] * desc2_shape[1] * desc2_shape[2];
        std::cout << "=> desc2 tensor len " << desc2_tensor_len << ", shape " << desc2_shape[0] << " * " << desc2_shape[1] << " *  " << desc2_shape[2] << std::endl;

        auto data_kp1 = input_tensors[0].sysMem.virAddr;
        memcpy(data_kp1, kpts1_data, kp1_tensor_len * sizeof(float));

        auto data_kp2 = input_tensors[1].sysMem.virAddr;
        memcpy(data_kp2, kpts2_data, kp2_tensor_len * sizeof(float));

        auto data_desc1 = input_tensors[2].sysMem.virAddr;
        memcpy(data_desc1, desc1, desc1_tensor_len * sizeof(float));

        auto data_desc2 = input_tensors[3].sysMem.virAddr;
        memcpy(data_desc2, desc2, desc2_tensor_len * sizeof(float));
    }

    // ==================================================== run infer =====================================================
    // Step4: run inference
    auto t1 = std::chrono::steady_clock::now();
    hbUCPTaskHandle_t task_handle = nullptr;
    hbDNNTensor *output = output_tensors.data();
    {
        // make sure memory data is flushed to DDR before inference
        for (int i = 0; i < input_count; i++)
        {
            hbUCPMemFlush(&input_tensors[i].sysMem, HB_SYS_MEM_CACHE_CLEAN);
        }

        hbUCPSchedParam infer_ctrl_param;
        HB_UCP_INITIALIZE_SCHED_PARAM(&infer_ctrl_param);
        HB_CHECK_SUCCESS(dnn_infer(&task_handle, &output, input_tensors.data(), dnn_handle_lg, &infer_ctrl_param), "dnn_infer failed");
        // wait task done
        HB_CHECK_SUCCESS(hbUCPWaitTaskDone(task_handle, 0), "hbUCPWaitTaskDone failed");
    }
    auto t2 = std::chrono::steady_clock::now();
    auto time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count() * 1000;
    std::cout << "\033[31m" << "=> lightglue infer Time: " << time_used << "ms\033[0m" << std::endl;

    // ==================================================== do postprocess with output data ===============================
    // Step5: do postprocess with output data
    {
        // make sure CPU read data from DDR before using output tensor data
        for (int i = 0; i < output_count; i++)
        {
            hbUCPMemFlush(&output_tensors[i].sysMem, HB_SYS_MEM_CACHE_INVALIDATE);
        }

        int *shape = output->properties.validShape.dimensionSize;
        int tensor_len = shape[0] * shape[1];
        std::cout << "=> lg tensor 0 len " << tensor_len << ", shape " << shape[0] << " * " << shape[1] << std::endl;

        int *shape_1 = output_tensors[1].properties.validShape.dimensionSize;
        int tensor_len_1 = shape_1[0];
        std::cout << "=> lg tensor 1 len " << tensor_len_1 << ", shape " << shape_1[0] << std::endl;

        auto matches = reinterpret_cast<int64_t *>(output_tensors[0].sysMem.virAddr);
        auto scores = reinterpret_cast<float *>(output_tensors[1].sysMem.virAddr);

        for (int i = 0; i < tensor_len_1; i++)
        {
            if (scores[i] > match_score_th)
            {
                match_kp_1.push_back(keypoint_1[matches[i * 2]]);
                match_kp_2.push_back(keypoint_2[matches[i * 2 + 1]]);
            }
        }
    }

    // ==================================================== release resources =============================================
    // Step6: release resources
    {
        // release task handle
        HB_CHECK_SUCCESS(hbUCPReleaseTask(task_handle), "hbUCPReleaseTask failed");
        // free input mem
        for (int i = 0; i < input_count; i++)
        {
            HB_CHECK_SUCCESS(hbUCPFree(&(input_tensors[i].sysMem)), "hbUCPFree failed");
        }
        // free output mem
        for (int i = 0; i < output_count; i++)
        {
            HB_CHECK_SUCCESS(hbUCPFree(&(output_tensors[i].sysMem)), "hbUCPFree failed");
        }
    }

    return 0;
}


int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << "./DFMatch_S100 input_image_dir output_image_dir\n";
        return 1;
    }

    std::string img_folder_name = argv[1];
    std::string img_match_vis_name = argv[2];
    if (!fs::exists(img_match_vis_name))
    {
        if (fs::create_directories(img_match_vis_name))
        {
            std::cout << "=> create result dir: " << img_match_vis_name << std::endl;\
        }
        else
        {
            std::cerr << "=> create result dir failed: " << img_match_vis_name << std::endl;
            return 1;
        }
    }

    std::cout << "-+ ================== Start ====================" << std::endl;
    for (int i = 0; i < 50; i++)
    {
        std::cout << "-+ ================== DFMatch ==================" << std::endl;
        std::string img_name_1 = img_folder_name + std::to_string(i) + ".png";
        std::cout << img_name_1 << std::endl;

        std::string img_name_2 = img_folder_name + std::to_string(i + 1) + ".png";
        std::cout << img_name_2 << std::endl;

        cv::Mat bgr_mat_1 = cv::imread(img_name_1, cv::IMREAD_COLOR);
        cv::Mat bgr_mat_2 = cv::imread(img_name_2, cv::IMREAD_COLOR);
        if (bgr_mat_1.empty())
        {
            std::cout << "=> image file " << img_name_1 << " not exist!" << std::endl;
            return -1;
        }
        if (bgr_mat_2.empty())
        {
            std::cout << "=> image file " << img_name_1 << " not exist!" << std::endl;
            return -1;
        }
        if (bgr_mat_1.cols != 640 || bgr_mat_1.rows != 480)
        {
            std::cout << "=> image " << img_name_1 << " size is not [640x480]!" << std::endl;
            return -1;
        }
        if (bgr_mat_2.cols != 640 || bgr_mat_2.rows != 480)
        {
            std::cout << "=> image " << img_name_1 << " size is not [640x480]!" << std::endl;
            return -1;
        }

        std::pair<std::vector<cv::Point2f>, Eigen::MatrixXd> dfeat_result_1;
        infer_s100_dfeat(bgr_mat_1, dfeat_result_1);

        std::pair<std::vector<cv::Point2f>, Eigen::MatrixXd> dfeat_result_2;
        infer_s100_dfeat(bgr_mat_2, dfeat_result_2);

        std::cout << "-+ ================== LightGlue ================" << std::endl;
        std::vector<cv::Point2f> match_pt_1, match_pt_2;
        infer_s100_lg(dfeat_result_1.first, dfeat_result_1.second, dfeat_result_2.first, dfeat_result_2.second, match_pt_1, match_pt_2);

        cv::Mat img_hstack;
        cv::hconcat(bgr_mat_1, bgr_mat_2, img_hstack);

        std::vector<cv::Point2f> points1 = dfeat_result_1.first;
        std::vector<cv::Point2f> points2 = dfeat_result_2.first;
        for (const auto &point : points1)
        {
            cv::Point point_img1;
            point_img1.x = (int)point.x;
            point_img1.y = (int)point.y;
            cv::circle(img_hstack, point_img1, 2, cv::Scalar(0, 255, 0), -1);
        }
        for (const auto &point : points2)
        {
            cv::Point point_img2;
            point_img2.x = (int)point.x + 640;
            point_img2.y = (int)point.y;
            cv::circle(img_hstack, point_img2, 2, cv::Scalar(0, 255, 0), -1);
        }
        for (size_t j = 0; j < match_pt_1.size(); j++)
        {
            cv::Point point_img1;
            cv::Point point_img2;

            point_img1.x = (int)match_pt_1[j].x;
            point_img1.y = (int)match_pt_1[j].y;

            point_img2.x = (int)match_pt_2[j].x + 640;
            point_img2.y = (int)match_pt_2[j].y;

            cv::circle(img_hstack, point_img1, 2, cv::Scalar(255, 0, 0), -1);
            cv::circle(img_hstack, point_img2, 2, cv::Scalar(255, 0, 0), -1);
            cv::line(img_hstack, point_img1, point_img2, cv::Scalar(0, 0, 255), 1);
        }

        std::string img_match_vis_path = img_match_vis_name + "match__" + std::to_string(i) + "__" + std::to_string(i + 1) + ".png";
        std::cout << "=> save result to: " << img_match_vis_path << std::endl;
        cv::imwrite(img_match_vis_path, img_hstack);
        // break;
    }

    std::cout << "-+ ================== End ======================" << std::endl;
    // release model
    HB_CHECK_SUCCESS(hbDNNRelease(packed_dnn_handle_df), "hbDNNRelease failed");
    HB_CHECK_SUCCESS(hbDNNRelease(packed_dnn_handle_lg), "hbDNNRelease failed");

    return 0;
}