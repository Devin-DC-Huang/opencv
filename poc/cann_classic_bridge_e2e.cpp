// e2e harness for the "restore the classic engine" PoC on OpenCV 5.x.
//
// Question being answered: with the classic (pre-5.x) engine path restored,
// does the CANN back-end become reachable again -- and does the model really
// execute on the NPU?
//
// Engine value 3 == the PoC classic engine.  cv::dnn::EngineType in 5.x is
//   { ENGINE_AUTO=0, ENGINE_OPENCV=1, ENGINE_ORT=2 }
// ENGINE_CLASSIC was removed by #29341 ("disable_engine_classic"), so the PoC
// uses 3 as a private value.
//
// Both runs below use the SAME importer (the classic one); only the back-end
// differs, so a mismatch cannot be blamed on a different model parser.
//
// Each back-end is run twice: the FIRST forward() also pays the one-off
// initialisation (for CANN that is the GE graph -> OM compile, tens of
// seconds), the SECOND one is pure inference.

#include <opencv2/core.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

static const int ENGINE_CLASSIC_POAC = 3;

struct Result
{
    double first_ms = 0;
    double second_ms = 0;
    int top1 = -1;
    double score = 0;
    cv::Mat out;
};

// cv::dnn::softmax() is not part of the 5.x public API any more, so do it here.
static void top1Of(const cv::Mat& out, int& top1, double& score)
{
    cv::Mat flat = out.reshape(1, 1);
    double mx = 0;
    cv::Point idx;
    cv::minMaxLoc(flat, nullptr, &mx, nullptr, &idx);

    const float* p = flat.ptr<float>();
    double sum = 0;
    for (int i = 0; i < flat.cols; ++i)
        sum += std::exp((double)p[i] - mx);

    top1 = idx.x;
    score = 1.0 / sum;
}

static cv::Mat makeInput()
{
    // A synthetic image: we compare CPU vs NPU outputs of the *same* net, so
    // the content does not matter, only that both back-ends see identical bits.
    cv::Mat img(224, 224, CV_8UC3);
    cv::RNG rng(20261104);
    rng.fill(img, cv::RNG::UNIFORM, cv::Scalar::all(0), cv::Scalar::all(255));
    return cv::dnn::blobFromImage(img, 1.0 / 255.0, cv::Size(224, 224), cv::Scalar(), true, false);
}

static Result run(const std::string& model, int backend, int target, const char* tag)
{
    printf("---- %s ----\n", tag);
    fflush(stdout);

    cv::dnn::Net net = cv::dnn::readNetFromONNX(model, ENGINE_CLASSIC_POAC);
    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    cv::Mat blob = makeInput();
    net.setInput(blob);

    auto t0 = std::chrono::steady_clock::now();
    cv::Mat out = net.forward();
    auto t1 = std::chrono::steady_clock::now();

    auto t2 = std::chrono::steady_clock::now();
    cv::Mat out2 = net.forward();
    auto t3 = std::chrono::steady_clock::now();

    Result r;
    r.first_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    r.second_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    r.out = out;
    top1Of(out, r.top1, r.score);

    double diff = -1;
    if (out.size == out2.size && out.type() == out2.type())
        diff = cv::norm(out, out2, cv::NORM_INF);

    printf("RESULT[%s] first_ms=%.2f second_ms=%.2f top1=%d score=%.6f out=%dx%dx%dx%d rerun_diff=%.3g\n",
           tag, r.first_ms, r.second_ms, r.top1, r.score,
           out.dims > 0 ? out.size[0] : 0,
           out.dims > 1 ? out.size[1] : 0,
           out.dims > 2 ? out.size[2] : 0,
           out.dims > 3 ? out.size[3] : 0,
           diff);
    fflush(stdout);
    return r;
}

int main(int argc, char** argv)
{
    const std::string model = (argc > 1) ? argv[1] : "/tmp/mobilenetv2-12.onnx";

    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_INFO);

    printf("HARNESS_START model=%s\n", model.c_str());
    fflush(stdout);

    Result cpu = run(model, cv::dnn::DNN_BACKEND_OPENCV, cv::dnn::DNN_TARGET_CPU, "CPU");
    Result npu = run(model, cv::dnn::DNN_BACKEND_CANN, cv::dnn::DNN_TARGET_NPU, "CANN_NPU");

    double maxdiff = -1;
    if (!cpu.out.empty() && !npu.out.empty() && cpu.out.size == npu.out.size && cpu.out.type() == npu.out.type())
        maxdiff = cv::norm(cpu.out, npu.out, cv::NORM_INF);

    printf("SUMMARY cpu_first_ms=%.2f cpu_second_ms=%.2f cann_first_ms=%.2f cann_second_ms=%.2f "
           "cpu_top1=%d cann_top1=%d score_delta=%.6f max_abs_diff=%.6g speedup_second=%.2fx\n",
           cpu.first_ms, cpu.second_ms, npu.first_ms, npu.second_ms,
           cpu.top1, npu.top1, std::fabs(cpu.score - npu.score), maxdiff,
           npu.second_ms > 0 ? cpu.second_ms / npu.second_ms : 0.0);
    fflush(stdout);

    const bool same = (cpu.top1 == npu.top1) && (std::fabs(cpu.score - npu.score) < 1e-3);
    printf("VERDICT %s\n", same ? "MATCH" : "MISMATCH");
    return same ? 0 : 2;
}
