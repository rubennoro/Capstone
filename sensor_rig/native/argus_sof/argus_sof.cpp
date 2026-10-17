// argus_sof: libargus capture helper for sensor_rig (backend "argus_sof"). NOT BUILT / NOT RUN YET (needs a Jetson
// with the Jetson Multimedia API; see README.md).
//
// Why: nvarguscamerasrc -> appsink stamps are buffer-arrival times (G2 / F8). This helper reads the sensor
// start-of-frame stamp from the capture metadata, ICaptureMetadata::getSensorTimestamp(), per frame.
//
// Follows the Jetson Multimedia API Argus samples (JetPack 6, /usr/src/jetson_multimedia_api):
//   argus/samples/userAutoExposure  - ICaptureMetadata (sensor timestamp, exposure, analog gain) per frame
//   argus/samples/cudaHistogram, samples/09_argus_camera_jpeg - EGLStream::FrameConsumer, IImageNativeBuffer ->
//                                     NvBufSurface (dmabuf), NvBufSurfaceMap for CPU access
//   argus/samples/multiSensor       - one CameraDevice per sensor id
//
// Protocol (stdout, one line per message, flushed):
//   READY <width> <height> NV12 <slots> <slot_bytes> <shm_name>
//   F <seq> <sensor_ts_ns> <exposure_ns> <analog_gain> <slot>
//   E <message>                       (fatal; the process exits non-zero afterwards)
// Shared memory /dev/shm/<shm_name>: 32-byte header "ARGSOF01", u32 slots, u32 slot_bytes, u32 width, u32 height,
// u64 reserved; then `slots` slots of [u64 seq_begin][u64 seq_end][NV12 payload: Y plane w*h, then UV w*h/2,
// tightly packed]. Writer: seq_begin = seq, copy, seq_end = seq (release fences). The reader accepts a slot only if
// seq_end == seq_begin == the announced seq (otherwise it was overwritten: counted as an overrun).
// The sensor timestamp domain is CLOCK_MONOTONIC_RAW-based on L4T (verify with the LED test); sensor_rig converts it
// to CLOCK_MONOTONIC (clock.RawToMono).

#include <Argus/Argus.h>
#include <EGLStream/EGLStream.h>
#include <EGLStream/NV/ImageNativeBuffer.h>
#include <nvbufsurface.h>

#include <atomic>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/mman.h>
#include <unistd.h>

using namespace Argus;

static std::atomic<bool> g_stop(false);
static void on_signal(int) { g_stop = true; }

struct Opts {
    int sensor_id = 0, mode = -1, width = 1640, height = 1232, slots = 8;
    double fps = 30.0, gain_lo = 1.0, gain_hi = 8.0;
    uint64_t exp_lo = 100000, exp_hi = 4000000;
    bool ae_lock = false, awb_lock = false;
    std::string shm = "sensor_rig_argus_sof";
};

static void fail(const char* msg) {
    std::printf("E %s\n", msg);
    std::fflush(stdout);
    std::exit(2);
}

static Opts parse(int argc, char** argv) {
    Opts o;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string k = argv[i], v = argv[i + 1];
        if (k == "--sensor-id") o.sensor_id = std::atoi(v.c_str());
        else if (k == "--mode") o.mode = std::atoi(v.c_str());
        else if (k == "--width") o.width = std::atoi(v.c_str());
        else if (k == "--height") o.height = std::atoi(v.c_str());
        else if (k == "--fps") o.fps = std::atof(v.c_str());
        else if (k == "--exposure-ns") std::sscanf(v.c_str(), "%lu,%lu", &o.exp_lo, &o.exp_hi);
        else if (k == "--gain") std::sscanf(v.c_str(), "%lf,%lf", &o.gain_lo, &o.gain_hi);
        else if (k == "--ae-lock") o.ae_lock = v == "1";
        else if (k == "--awb-lock") o.awb_lock = v == "1";
        else if (k == "--slots") o.slots = std::atoi(v.c_str());
        else if (k == "--shm-name") o.shm = v;
        else fail(("unknown option " + k).c_str());
    }
    return o;
}

int main(int argc, char** argv) {
    Opts o = parse(argc, argv);
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    UniqueObj<CameraProvider> provider(CameraProvider::create());
    ICameraProvider* iProvider = interface_cast<ICameraProvider>(provider);
    if (!iProvider) fail("cannot create the Argus CameraProvider (is nvargus-daemon running?)");
    std::vector<CameraDevice*> devices;
    iProvider->getCameraDevices(&devices);
    if (o.sensor_id >= (int)devices.size()) fail("sensor id not found");
    CameraDevice* device = devices[o.sensor_id];
    ICameraProperties* iProps = interface_cast<ICameraProperties>(device);
    std::vector<SensorMode*> modes;
    iProps->getAllSensorModes(&modes);
    if (o.mode >= (int)modes.size()) fail("sensor mode not found");

    UniqueObj<CaptureSession> session(iProvider->createCaptureSession(device));
    ICaptureSession* iSession = interface_cast<ICaptureSession>(session);
    if (!iSession) fail("cannot create a capture session (camera busy?)");

    UniqueObj<OutputStreamSettings> streamSettings(iSession->createOutputStreamSettings(STREAM_TYPE_EGL));
    IEGLOutputStreamSettings* iEgl = interface_cast<IEGLOutputStreamSettings>(streamSettings);
    iEgl->setPixelFormat(PIXEL_FMT_YCbCr_420_888);
    iEgl->setResolution(Size2D<uint32_t>(o.width, o.height));
    iEgl->setMetadataEnable(true);
    UniqueObj<OutputStream> stream(iSession->createOutputStream(streamSettings.get()));
    UniqueObj<EGLStream::FrameConsumer> consumer(EGLStream::FrameConsumer::create(stream.get()));
    EGLStream::IFrameConsumer* iConsumer = interface_cast<EGLStream::IFrameConsumer>(consumer);
    if (!iConsumer) fail("cannot create the EGLStream frame consumer");

    UniqueObj<Request> request(iSession->createRequest());
    IRequest* iRequest = interface_cast<IRequest>(request);
    iRequest->enableOutputStream(stream.get());
    ISourceSettings* iSource = interface_cast<ISourceSettings>(iRequest->getSourceSettings());
    if (o.mode >= 0) iSource->setSensorMode(modes[o.mode]);
    uint64_t frame_ns = (uint64_t)(1e9 / o.fps);
    iSource->setFrameDurationRange(Range<uint64_t>(frame_ns));
    iSource->setExposureTimeRange(Range<uint64_t>(o.exp_lo, o.exp_hi));
    iSource->setGainRange(Range<float>((float)o.gain_lo, (float)o.gain_hi));
    IAutoControlSettings* iAc = interface_cast<IAutoControlSettings>(iRequest->getAutoControlSettings());
    iAc->setAeLock(o.ae_lock);
    iAc->setAwbLock(o.awb_lock);
    iAc->setIspDigitalGainRange(Range<float>(1.0f, 1.0f));

    // shared memory ring
    const size_t payload = (size_t)o.width * o.height * 3 / 2;
    const size_t slot_bytes = 16 + payload;
    const size_t total = 32 + slot_bytes * o.slots;
    int fd = shm_open(("/" + o.shm).c_str(), O_CREAT | O_RDWR, 0600);
    if (fd < 0 || ftruncate(fd, total) != 0) fail("cannot create the shared memory ring");
    uint8_t* shm = (uint8_t*)mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (shm == MAP_FAILED) fail("cannot map the shared memory ring");
    std::memcpy(shm, "ARGSOF01", 8);
    uint32_t hdr[4] = {(uint32_t)o.slots, (uint32_t)slot_bytes, (uint32_t)o.width, (uint32_t)o.height};
    std::memcpy(shm + 8, hdr, sizeof(hdr));

    if (iSession->repeat(request.get()) != STATUS_OK) fail("cannot start the repeating capture request");
    std::printf("READY %d %d NV12 %d %zu %s\n", o.width, o.height, o.slots, slot_bytes, o.shm.c_str());
    std::fflush(stdout);

    int dmabuf = -1;
    while (!g_stop) {
        Argus::Status st;
        UniqueObj<EGLStream::Frame> frame(iConsumer->acquireFrame(2000000000ULL /* 2 s */, &st));
        if (st == STATUS_TIMEOUT) continue;
        EGLStream::IFrame* iFrame = interface_cast<EGLStream::IFrame>(frame);
        if (!iFrame) fail("acquireFrame failed (camera stopped?)");
        EGLStream::IArgusCaptureMetadata* iMeta = interface_cast<EGLStream::IArgusCaptureMetadata>(frame);
        ICaptureMetadata* md = iMeta ? interface_cast<ICaptureMetadata>(iMeta->getMetadata()) : nullptr;
        if (!md) fail("no capture metadata (setMetadataEnable)");
        const uint64_t seq = iFrame->getNumber();
        const uint64_t sensor_ts = md->getSensorTimestamp();
        const uint64_t exposure = md->getSensorExposureTime();
        const float gain = md->getSensorAnalogGain();

        EGLStream::NV::IImageNativeBuffer* iNative =
            interface_cast<EGLStream::NV::IImageNativeBuffer>(iFrame->getImage());
        if (!iNative) fail("IImageNativeBuffer not supported");
        if (dmabuf < 0) {
            dmabuf = iNative->createNvBuffer(Size2D<uint32_t>(o.width, o.height), NVBUF_COLOR_FORMAT_NV12,
                                             NVBUF_LAYOUT_PITCH);
            if (dmabuf < 0) fail("createNvBuffer failed");
        } else if (iNative->copyToNvBuffer(dmabuf) != STATUS_OK) {
            fail("copyToNvBuffer failed");
        }
        NvBufSurface* surf = nullptr;
        if (NvBufSurfaceFromFd(dmabuf, (void**)&surf) != 0) fail("NvBufSurfaceFromFd failed");
        NvBufSurfaceMap(surf, 0, -1, NVBUF_MAP_READ);
        NvBufSurfaceSyncForCpu(surf, 0, -1);
        const int slot = (int)(seq % (uint64_t)o.slots);
        uint8_t* s = shm + 32 + (size_t)slot * slot_bytes;
        std::memcpy(s, &seq, 8);
        std::atomic_thread_fence(std::memory_order_release);
        uint8_t* dst = s + 16;
        const NvBufSurfaceParams& p = surf->surfaceList[0];
        for (int plane = 0; plane < 2; ++plane) {  // Y, then interleaved UV; strip the pitch padding
            const int rows = plane == 0 ? o.height : o.height / 2;
            const uint8_t* src = (const uint8_t*)p.mappedAddr.addr[plane];
            for (int r = 0; r < rows; ++r) {
                std::memcpy(dst, src + (size_t)r * p.planeParams.pitch[plane], o.width);
                dst += o.width;
            }
        }
        NvBufSurfaceUnMap(surf, 0, -1);
        std::atomic_thread_fence(std::memory_order_release);
        std::memcpy(s + 8, &seq, 8);
        std::printf("F %lu %lu %lu %.4f %d\n", seq, sensor_ts, exposure, gain, slot);
        std::fflush(stdout);
    }
    iSession->stopRepeat();
    iSession->waitForIdle();
    if (dmabuf >= 0) NvBufSurf::NvDestroy(dmabuf);
    munmap(shm, total);
    shm_unlink(("/" + o.shm).c_str());
    return 0;
}
