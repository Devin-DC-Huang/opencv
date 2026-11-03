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

#include <opencv2/core.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>

static const int ENGINE_CLASSIC_POAC = 3;

struct Result
{
    double ms = 0;
    int top1 = -1;
    double score = 0;
};

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

    Result r;
    r.ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    cv::Mat probs = cv::dnn::softmax(out);
    cv::Point idx;
    double mx = 0;
    cv::minMaxLoc(probs.reshape(1, 1), nullptr, &mx, nullptr, &idx);
    r.top1 = idx.x;
    r.score = mx;

    printf("RESULT[%s] ms=%.2f top1=%d score=%.6f out=%dx%dx%dx%d\n",
           tag, r.ms, r.top1, r.score,
           out.dims > 0 ? out.size[0] : 0,
           out.dims > 1 ? out.size[1] : 0,
           out.dims > 2 ? out.size[2] : 0,
           out.dims > 3 ? out.size[3] : 0);
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

    printf("SUMMARY cpu_ms=%.2f cann_ms=%.2f cpu_top1=%d cann_top1=%d score_delta=%.6f\n",
           cpu.ms, npu.ms, cpu.top1, npu.top1, std::fabs(cpu.score - npu.score));
    fflush(stdout);

    const bool same = (cpu.top1 == npu.top1) && (std::fabs(cpu.score - npu.score) < 1e-3);
    printf("VERDICT %s\n", same ? "MATCH" : "MISMATCH");
    return same ? 0 : 2;
}
