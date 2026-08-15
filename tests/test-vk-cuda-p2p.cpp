// Standalone Vulkan<->CUDA P2P probe (no ggml dependency).
//
// Usage: test-vk-cuda-p2p [devA] [devB]
//
// Picks two NVIDIA physical devices (default: the first two) and checks whether a real
// device-to-device copy is reachable by exporting Vulkan DEVICE_LOCAL buffers through
// VK_KHR_external_memory_fd (OPAQUE_FD) and importing them into CUDA, then copying with
// cuMemcpyPeer. Prints the measured P2P bandwidth in both directions.
//
// Build (system vulkan dev package + CUDA driver headers):
//   c++ -std=c++20 -O2 test-vk-cuda-p2p.cpp -o test-vk-cuda-p2p \
//       -lvulkan -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcuda
//
// Expected result on two NVIDIA RTX PRO 4000 Blackwell: ~44 GB/s in each direction
// (PCIe P2P; there is no NVLink on these parts).
//
#include <vulkan/vulkan.hpp>
#include <cuda.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <unistd.h>
#include <chrono>

static vk::Instance g_inst;

static uint32_t find_qfam(vk::PhysicalDevice d, vk::QueueFlags want) {
    auto qp = d.getQueueFamilyProperties();
    for (uint32_t i = 0; i < qp.size(); i++) if ((qp[i].queueFlags & want) == want) return i;
    return ~0u;
}

static uint32_t memtype(vk::PhysicalDevice phys, uint32_t bits, vk::MemoryPropertyFlags want) {
    auto mp = phys.getMemoryProperties();
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
    return ~0u;
}

#define CU_CHK(call) do { CUresult r = (call); if (r != CUDA_SUCCESS) { const char* s = ""; cuGetErrorString(r, &s); fprintf(stderr, "CUDA error %s at %d\n", s, __LINE__); return 1; } } while(0)

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    vk::ApplicationInfo app{"vkcuda-p2p", 1};
    g_inst = vk::createInstance(vk::InstanceCreateInfo{{}, &app});
    auto phys = g_inst.enumeratePhysicalDevices();

    int ai = 0, bi = 1;
    if (argc > 1) ai = atoi(argv[1]);
    if (argc > 2) bi = atoi(argv[2]);
    printf("A = [%d] %s\nB = [%d] %s\n", ai, (const char*) phys[ai].getProperties().deviceName,
           bi, (const char*) phys[bi].getProperties().deviceName);

    std::vector<const char*> exts = {
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
        VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME,
        VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME,
    };
    auto mkdev = [&](int i) {
        uint32_t q = find_qfam(phys[i], vk::QueueFlagBits::eCompute | vk::QueueFlagBits::eTransfer);
        float prio = 1.0f;
        vk::DeviceQueueCreateInfo qci{{}, q, 1, &prio};
        return phys[i].createDevice(vk::DeviceCreateInfo{{}, 1, &qci, 0, nullptr, (uint32_t)exts.size(), exts.data()});
    };
    vk::Device dev0 = mkdev(ai), dev1 = mkdev(bi);

    const size_t N = 256ull * 1024 * 1024; // 256 MB
    vk::BufferUsageFlags usage = vk::BufferUsageFlagBits::eTransferSrc |
                                 vk::BufferUsageFlagBits::eTransferDst |
                                 vk::BufferUsageFlagBits::eStorageBuffer;
    auto ht = vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd;

    auto make_export_buf = [&](vk::Device& dev, vk::PhysicalDevice& pd, int* out_fd) {
        vk::ExternalMemoryBufferCreateInfo ebci{ht};
        vk::Buffer b = dev.createBuffer(vk::BufferCreateInfo{{}, N, usage, vk::SharingMode::eExclusive, 0, nullptr, &ebci});
        auto req = dev.getBufferMemoryRequirements(b);
        uint32_t t = memtype(pd, req.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
        vk::MemoryDedicatedAllocateInfo ded{{}, b, nullptr};
        vk::ExportMemoryAllocateInfo emai{ht, &ded};
        vk::DeviceMemory m = dev.allocateMemory(vk::MemoryAllocateInfo{req.size, t, &emai});
        dev.bindBufferMemory(b, m, 0);
        auto pfn = (PFN_vkGetMemoryFdKHR) dev.getProcAddr("vkGetMemoryFdKHR");
        VkMemoryGetFdInfoKHR gi{};
        gi.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
        gi.memory = (VkDeviceMemory) m;
        gi.handleType = (VkExternalMemoryHandleTypeFlagBits) ht;
        int fd = -1;
        pfn((VkDevice) dev, &gi, &fd);
        *out_fd = fd;
        return m;
    };

    int fd0, fd1;
    auto m0 = make_export_buf(dev0, phys[ai], &fd0);
    auto m1 = make_export_buf(dev1, phys[bi], &fd1);

    CU_CHK(cuInit(0));
    CUdevice cu0, cu1;
    CU_CHK(cuDeviceGet(&cu0, ai));
    CU_CHK(cuDeviceGet(&cu1, bi));
    CUcontext ctx0, ctx1;
    CU_CHK(cuDevicePrimaryCtxRetain(&ctx0, cu0));
    CU_CHK(cuDevicePrimaryCtxRetain(&ctx1, cu1));

    CUexternalMemory ext0, ext1;
    CUdeviceptr p0 = 0, p1 = 0;
    {
        CU_CHK(cuCtxSetCurrent(ctx0));
        CUDA_EXTERNAL_MEMORY_HANDLE_DESC hd{};
        hd.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
        hd.handle.fd = fd0;
        hd.size = N;
        hd.flags = 0;
        CU_CHK(cuImportExternalMemory(&ext0, &hd));
        CUDA_EXTERNAL_MEMORY_BUFFER_DESC bd{};
        bd.offset = 0; bd.size = N; bd.flags = 0;
        CU_CHK(cuExternalMemoryGetMappedBuffer(&p0, ext0, &bd));
    }
    {
        CU_CHK(cuCtxSetCurrent(ctx1));
        CUDA_EXTERNAL_MEMORY_HANDLE_DESC hd{};
        hd.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
        hd.handle.fd = fd1;
        hd.size = N;
        hd.flags = 0;
        CU_CHK(cuImportExternalMemory(&ext1, &hd));
        CUDA_EXTERNAL_MEMORY_BUFFER_DESC bd{};
        bd.offset = 0; bd.size = N; bd.flags = 0;
        CU_CHK(cuExternalMemoryGetMappedBuffer(&p1, ext1, &bd));
    }

    // correctness round-trip first
    std::vector<float> src(N / sizeof(float));
    for (size_t i = 0; i < src.size(); i++) src[i] = (float)(i + 1);
    CU_CHK(cuCtxSetCurrent(ctx0));
    CU_CHK(cuMemcpyHtoD(p0, src.data(), N));
    CU_CHK(cuMemcpyPeer(p1, ctx1, p0, ctx0, N));
    std::vector<float> dst(N / sizeof(float));
    CU_CHK(cuCtxSetCurrent(ctx1));
    CU_CHK(cuMemcpyDtoH(dst.data(), p1, N));
    printf("round-trip: %s\n", memcmp(src.data(), dst.data(), N) == 0 ? "OK" : "MISMATCH");

    // warmup
    CU_CHK(cuMemcpyPeer(p1, ctx1, p0, ctx0, N));
    CU_CHK(cuCtxSetCurrent(ctx1)); CU_CHK(cuCtxSynchronize());
    CU_CHK(cuCtxSetCurrent(ctx0)); CU_CHK(cuCtxSynchronize());

    const int iters = 20;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; i++) CU_CHK(cuMemcpyPeer(p1, ctx1, p0, ctx0, N));
    CU_CHK(cuCtxSetCurrent(ctx1)); CU_CHK(cuCtxSynchronize());
    auto t1 = std::chrono::steady_clock::now();
    double dt = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / 1e9;
    printf("P2P dev%d->dev%d: %.1f GB/s (%.3f ms/copy)\n", ai, bi, N * iters / dt / 1e9, dt / iters * 1e3);

    t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; i++) CU_CHK(cuMemcpyPeer(p0, ctx0, p1, ctx1, N));
    CU_CHK(cuCtxSetCurrent(ctx0)); CU_CHK(cuCtxSynchronize());
    t1 = std::chrono::steady_clock::now();
    dt = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / 1e9;
    printf("P2P dev%d->dev%d: %.1f GB/s (%.3f ms/copy)\n", bi, ai, N * iters / dt / 1e9, dt / iters * 1e3);

    return 0;
}
