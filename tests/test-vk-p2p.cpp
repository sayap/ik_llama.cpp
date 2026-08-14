// Standalone Vulkan cross-device P2P test (no ggml dependency).
//
// Usage: test-vk-p2p [devA] [devB]
//
// Picks two physical devices (default: the first two NVIDIA devices, or the two
// indices given on the command line) and exercises the mechanisms we need for a
// P2P all-reduce:
//
//   1. shared HOST buffer via VK_EXT_external_memory_dma_buf + vkCmdCopyBuffer
//   2. cross-device semaphore sync via VK_KHR_external_semaphore_fd
//        - SYNC_FD  (the cross-vendor handle type)
//        - OPAQUE_FD (vendor-specific)
//   3. (informational) DEVICE_LOCAL memory import via DMA_BUF, which is the
//      "real" GPU P2P but is technically a spec violation (VUID-00644)
//
// Build (system vulkan dev package):
//   c++ -std=c++20 -O2 test-vk-p2p.cpp -o test-vk-p2p -lvulkan
//
#include <vulkan/vulkan.hpp>
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <poll.h>
#include <thread>
#include <atomic>
#include <chrono>
#include <set>
#include <unistd.h>

static constexpr uint64_t kTimeoutNs = 2000000000ull; // 2s per wait

static vk::Instance g_inst;
static std::atomic<bool> g_had_timeout{false};

static void semaphore_caps(vk::PhysicalDevice phys, vk::ExternalSemaphoreHandleTypeFlagBits ht, bool& importable, bool& exportable) {
    importable = exportable = false;
    auto pfn = (PFN_vkGetPhysicalDeviceExternalSemaphorePropertiesKHR) g_inst.getProcAddr("vkGetPhysicalDeviceExternalSemaphorePropertiesKHR");
    if (!pfn) return;
    VkPhysicalDeviceExternalSemaphoreInfoKHR info{};
    info.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO_KHR;
    info.handleType = (VkExternalSemaphoreHandleTypeFlagBits) ht;
    VkExternalSemaphorePropertiesKHR props{};
    props.sType = VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES_KHR;
    pfn((VkPhysicalDevice) phys, &info, &props);
    importable = (props.externalSemaphoreFeatures & VK_EXTERNAL_SEMAPHORE_FEATURE_IMPORTABLE_BIT) != 0;
    exportable = (props.externalSemaphoreFeatures & VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT) != 0;
}

static uint32_t find_qfam(vk::PhysicalDevice d, vk::QueueFlags want) {
    auto qp = d.getQueueFamilyProperties();
    for (uint32_t i = 0; i < qp.size(); i++) {
        if ((qp[i].queueFlags & want) == want) return i;
    }
    return ~0u;
}

static std::set<std::string> available_device_exts(vk::PhysicalDevice phys) {
    std::set<std::string> s;
    for (auto& e : phys.enumerateDeviceExtensionProperties()) s.insert((const char*) e.extensionName);
    return s;
}

// enabled-extension bookkeeping shared across the two devices
static bool g_have_ext_memory_fd = false;
static bool g_have_ext_semaphore_fd = false;
static bool g_have_ext_dma_buf = false;

static vk::Device mkdev(vk::PhysicalDevice phys) {
    uint32_t q = find_qfam(phys, vk::QueueFlagBits::eCompute | vk::QueueFlagBits::eTransfer);
    float prio = 1.0f;
    vk::DeviceQueueCreateInfo qci{{}, q, 1, &prio};

    auto avail = available_device_exts(phys);
    std::vector<const char*> wanted = {
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
        VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME,
        VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME,
        VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME,
    };
    std::vector<const char*> exts;
    for (auto w : wanted) {
        if (avail.count(w)) {
            exts.push_back(w);
        } else {
            printf("  note: extension not present on %s: %s\n", (const char*) phys.getProperties().deviceName, w);
        }
    }

    g_have_ext_memory_fd    |= avail.count(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME) > 0;
    g_have_ext_semaphore_fd |= avail.count(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME) > 0;
    g_have_ext_dma_buf      |= avail.count(VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME) > 0;

    vk::DeviceCreateInfo dci{{}, 1, &qci, 0, nullptr, (uint32_t)exts.size(), exts.data()};
    return phys.createDevice(dci);
}

static uint32_t memtype(vk::PhysicalDevice phys, uint32_t bits, vk::MemoryPropertyFlags want) {
    auto mp = phys.getMemoryProperties();
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
    }
    return ~0u;
}

static vk::CommandBuffer one_cb(vk::Device d, vk::CommandPool p) {
    return d.allocateCommandBuffers(vk::CommandBufferAllocateInfo{p, vk::CommandBufferLevel::ePrimary, 1})[0];
}

// submit a command buffer, optionally waiting/signaling a semaphore; returns fence wait result
static vk::Result submit_wait(vk::Device d, vk::Queue q, vk::CommandBuffer cb,
                              vk::Semaphore wait = {}, vk::PipelineStageFlags wait_stage = {},
                              vk::Semaphore signal = {}) {
    cb.end();
    vk::Fence f = d.createFence(vk::FenceCreateInfo{});
    vk::SubmitInfo si{
        wait ? 1u : 0u, wait ? &wait : nullptr, wait ? &wait_stage : nullptr,
        1, &cb,
        signal ? 1u : 0u, signal ? &signal : nullptr,
    };
    q.submit(si, f);
    vk::Result r = d.waitForFences(f, VK_TRUE, kTimeoutNs);
    d.destroyFence(f);
    return r;
}

// vkQueueSubmit can block forever on some broken semaphore imports, so run it on a
// worker thread and give up after a timeout.
static vk::Result submit_wait_timeout(vk::Device d, vk::Queue q, vk::CommandBuffer cb,
                                      vk::Semaphore wait, vk::PipelineStageFlags wait_stage,
                                      vk::Semaphore signal) {
    std::atomic<bool> done{false};
    std::atomic<int> result{(int) vk::Result::eErrorUnknown};
    std::thread t([&] {
        result = (int) submit_wait(d, q, cb, wait, wait_stage, signal);
        done = true;
    });
    for (int i = 0; i < 500 && !done; i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (!done) {
        g_had_timeout = true;
        t.detach();
        return vk::Result::eTimeout;
    }
    t.join();
    return (vk::Result) result.load();
}

static int export_memory(vk::Device d, vk::DeviceMemory mem, vk::ExternalMemoryHandleTypeFlagBits ht) {
    auto pfn = (PFN_vkGetMemoryFdKHR) d.getProcAddr("vkGetMemoryFdKHR");
    if (!pfn) return -1;
    VkMemoryGetFdInfoKHR gi{};
    gi.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    gi.memory = (VkDeviceMemory) mem;
    gi.handleType = (VkExternalMemoryHandleTypeFlagBits) ht;
    int fd = -1;
    VkResult r = pfn((VkDevice) d, &gi, &fd);
    return (r == VK_SUCCESS) ? fd : -1;
}

static int export_semaphore(vk::Device d, vk::Semaphore sem, vk::ExternalSemaphoreHandleTypeFlagBits ht) {
    auto pfn = (PFN_vkGetSemaphoreFdKHR) d.getProcAddr("vkGetSemaphoreFdKHR");
    if (!pfn) return -1;
    VkSemaphoreGetFdInfoKHR gi{};
    gi.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
    gi.semaphore = (VkSemaphore) sem;
    gi.handleType = (VkExternalSemaphoreHandleTypeFlagBits) ht;
    int fd = -1;
    VkResult r = pfn((VkDevice) d, &gi, &fd);
    return (r == VK_SUCCESS) ? fd : -1;
}

struct Dev {
    int idx = -1;
    vk::PhysicalDevice phys;
    vk::Device dev;
    uint32_t qfam;
    vk::Queue queue;
    vk::CommandPool pool;
    const char* name;
};

static vk::BufferUsageFlags kUsage =
    vk::BufferUsageFlagBits::eTransferSrc |
    vk::BufferUsageFlagBits::eTransferDst |
    vk::BufferUsageFlagBits::eStorageBuffer;

static vk::ExternalMemoryHandleTypeFlagBits mem_handle_type() {
    return g_have_ext_dma_buf ? vk::ExternalMemoryHandleTypeFlagBits::eDmaBufEXT
                             : vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd;
}

// ---- test 1: shared HOST buffer via external memory fd ----
static bool test_shared_host_buffer(Dev& a, Dev& b) {
    const size_t N = 4096 * sizeof(float);
    const auto ht = mem_handle_type();

    // allocate + export a host-visible buffer on A
    vk::ExternalMemoryBufferCreateInfo ebci{ht};
    vk::Buffer shA = a.dev.createBuffer(vk::BufferCreateInfo{{}, N, kUsage, vk::SharingMode::eExclusive, 0, nullptr, &ebci});
    auto reqA = a.dev.getBufferMemoryRequirements(shA);
    uint32_t tA = memtype(a.phys, reqA.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    vk::MemoryDedicatedAllocateInfo dedA{{}, shA, nullptr};
    vk::ExportMemoryAllocateInfo emai{ht, &dedA};
    vk::DeviceMemory shMemA = a.dev.allocateMemory(vk::MemoryAllocateInfo{reqA.size, tA, &emai});
    a.dev.bindBufferMemory(shA, shMemA, 0);
    int fd = export_memory(a.dev, shMemA, ht);
    if (fd < 0) { printf("  [1] export memory failed\n"); return false; }

    // import on B
    vk::ExternalMemoryBufferCreateInfo ebciB{ht};
    vk::Buffer shB = b.dev.createBuffer(vk::BufferCreateInfo{{}, N, kUsage, vk::SharingMode::eExclusive, 0, nullptr, &ebciB});
    auto reqB = b.dev.getBufferMemoryRequirements(shB);
    uint32_t tB = memtype(b.phys, reqB.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible);
    vk::MemoryDedicatedAllocateInfo dedB{{}, shB, nullptr};
    vk::ImportMemoryFdInfoKHR imfi{ht, fd, &dedB};
    vk::DeviceMemory shMemB = b.dev.allocateMemory(vk::MemoryAllocateInfo{reqB.size, tB, &imfi});
    b.dev.bindBufferMemory(shB, shMemB, 0);

    // device-local buffers on both
    auto devbuf = [&](Dev& d) {
        vk::Buffer buf = d.dev.createBuffer(vk::BufferCreateInfo{{}, N, kUsage});
        auto req = d.dev.getBufferMemoryRequirements(buf);
        uint32_t t = memtype(d.phys, req.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
        vk::DeviceMemory mem = d.dev.allocateMemory(vk::MemoryAllocateInfo{req.size, t});
        d.dev.bindBufferMemory(buf, mem, 0);
        return std::make_pair(buf, mem);
    };
    auto [bufA, memA] = devbuf(a);
    auto [bufB, memB] = devbuf(b);

    // upload known data to bufA
    std::vector<float> src(N / sizeof(float));
    for (size_t i = 0; i < src.size(); i++) src[i] = (float)(i + 1);
    {
        vk::Buffer up = a.dev.createBuffer(vk::BufferCreateInfo{{}, N, vk::BufferUsageFlagBits::eTransferSrc});
        auto req = a.dev.getBufferMemoryRequirements(up);
        uint32_t t = memtype(a.phys, req.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible);
        vk::DeviceMemory m = a.dev.allocateMemory(vk::MemoryAllocateInfo{req.size, t});
        a.dev.bindBufferMemory(up, m, 0);
        void* p = a.dev.mapMemory(m, 0, N); memcpy(p, src.data(), N); a.dev.unmapMemory(m);
        vk::CommandBuffer cb = one_cb(a.dev, a.pool);
        cb.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        vk::BufferCopy bc{0, 0, N}; cb.copyBuffer(up, bufA, 1, &bc);
        submit_wait(a.dev, a.queue, cb);
        a.dev.freeCommandBuffers(a.pool, cb);
        a.dev.destroyBuffer(up); a.dev.freeMemory(m);
    }

    // A: bufA -> shared, host-fence; B: shared -> bufB, host-fence
    {
        vk::CommandBuffer cb = one_cb(a.dev, a.pool);
        cb.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        vk::BufferCopy bc{0, 0, N}; cb.copyBuffer(bufA, shA, 1, &bc);
        if (submit_wait(a.dev, a.queue, cb) != vk::Result::eSuccess) { printf("  [1] A copy failed\n"); return false; }
        a.dev.freeCommandBuffers(a.pool, cb);
    }
    {
        vk::CommandBuffer cb = one_cb(b.dev, b.pool);
        cb.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        vk::BufferCopy bc{0, 0, N}; cb.copyBuffer(shB, bufB, 1, &bc);
        if (submit_wait(b.dev, b.queue, cb) != vk::Result::eSuccess) { printf("  [1] B copy failed\n"); return false; }
        b.dev.freeCommandBuffers(b.pool, cb);
    }

    // read back bufB
    std::vector<float> dst(N / sizeof(float));
    {
        vk::Buffer dn = b.dev.createBuffer(vk::BufferCreateInfo{{}, N, vk::BufferUsageFlagBits::eTransferDst});
        auto req = b.dev.getBufferMemoryRequirements(dn);
        uint32_t t = memtype(b.phys, req.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible);
        vk::DeviceMemory m = b.dev.allocateMemory(vk::MemoryAllocateInfo{req.size, t});
        b.dev.bindBufferMemory(dn, m, 0);
        vk::CommandBuffer cb = one_cb(b.dev, b.pool);
        cb.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        vk::BufferCopy bc{0, 0, N}; cb.copyBuffer(bufB, dn, 1, &bc);
        submit_wait(b.dev, b.queue, cb);
        b.dev.freeCommandBuffers(b.pool, cb);
        void* p = b.dev.mapMemory(m, 0, N); memcpy(dst.data(), p, N); b.dev.unmapMemory(m);
        b.dev.destroyBuffer(dn); b.dev.freeMemory(m);
    }

    bool ok = memcmp(src.data(), dst.data(), N) == 0;
    printf("  [1] shared HOST buffer (%s): %s\n", ht == vk::ExternalMemoryHandleTypeFlagBits::eDmaBufEXT ? "DMA_BUF" : "OPAQUE_FD", ok ? "OK" : "MISMATCH");
    return ok;
}

// ---- test 2/3: cross-device semaphore ----
static bool test_semaphore(Dev& a, Dev& b, vk::ExternalSemaphoreHandleTypeFlagBits ht, const char* name) {
    bool impA=false, expA=false, impB=false, expB=false;
    semaphore_caps(a.phys, ht, impA, expA);
    semaphore_caps(b.phys, ht, impB, expB);
    if (!expA || !impB) {
        printf("  [%s] unsupported (A import=%d export=%d, B import=%d export=%d)\n", name, impA, expA, impB, expB);
        return false;
    }

    vk::Semaphore semA;
    try {
        // create + pre-signal an exportable binary semaphore on A
        vk::ExportSemaphoreCreateInfo esci{ht};
        semA = a.dev.createSemaphore(vk::SemaphoreCreateInfo{{}, &esci});
        vk::CommandBuffer cb = one_cb(a.dev, a.pool);
        cb.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        if (submit_wait(a.dev, a.queue, cb, {}, {}, semA) != vk::Result::eSuccess) { printf("  [%s] pre-signal failed\n", name); return false; }
        a.dev.freeCommandBuffers(a.pool, cb);
    } catch (const vk::SystemError& e) {
        printf("  [%s] create/export failed: %s\n", name, e.what());
        return false;
    }

    int fd = export_semaphore(a.dev, semA, ht);
    if (fd < 0) { printf("  [%s] export failed\n", name); return false; }

    vk::Semaphore semB;
    try {
        vk::ImportSemaphoreFdInfoKHR isfi{{}, vk::SemaphoreImportFlagBits::eTemporary, ht, fd};
        semB = b.dev.createSemaphore(vk::SemaphoreCreateInfo{{}, &isfi});
    } catch (const vk::SystemError& e) {
        printf("  [%s] import failed: %s\n", name, e.what());
        return false;
    }

    // B: wait on the imported semaphore (empty submit)
    vk::CommandBuffer cb = one_cb(b.dev, b.pool);
    cb.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    vk::PipelineStageFlags stage = vk::PipelineStageFlagBits::eAllCommands;
    vk::Result r = submit_wait_timeout(b.dev, b.queue, cb, semB, stage, {});
    if (r != vk::Result::eTimeout) b.dev.freeCommandBuffers(b.pool, cb);

    bool ok = (r == vk::Result::eSuccess);
    printf("  [%s] semaphore sync: %s (wait result = %d)\n", name, ok ? "OK" : (r == vk::Result::eTimeout ? "HANG/TIMEOUT" : "FAIL"), (int)r);
    return ok;
}

// ---- test 4 (informational): device-local memory import ----
static void test_device_local_p2p(Dev& a, Dev& b) {
    const size_t N = 4096 * sizeof(float);
    const auto ht = mem_handle_type();

    vk::ExternalMemoryBufferCreateInfo ebci{ht};
    vk::Buffer bufA = a.dev.createBuffer(vk::BufferCreateInfo{{}, N, kUsage, vk::SharingMode::eExclusive, 0, nullptr, &ebci});
    auto reqA = a.dev.getBufferMemoryRequirements(bufA);
    uint32_t tA = memtype(a.phys, reqA.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
    vk::MemoryDedicatedAllocateInfo dedA{{}, bufA, nullptr};
    vk::ExportMemoryAllocateInfo emai{ht, &dedA};
    vk::DeviceMemory memA = a.dev.allocateMemory(vk::MemoryAllocateInfo{reqA.size, tA, &emai});
    a.dev.bindBufferMemory(bufA, memA, 0);
    int fd = export_memory(a.dev, memA, ht);
    if (fd < 0) { printf("  [4] device-local memory export failed\n"); return; }

    vk::ExternalMemoryBufferCreateInfo ebciB{ht};
    vk::Buffer bufB = b.dev.createBuffer(vk::BufferCreateInfo{{}, N, kUsage, vk::SharingMode::eExclusive, 0, nullptr, &ebciB});
    auto reqB = b.dev.getBufferMemoryRequirements(bufB);
    uint32_t tB = memtype(b.phys, reqB.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
    vk::MemoryDedicatedAllocateInfo dedB{{}, bufB, nullptr};
    vk::ImportMemoryFdInfoKHR imfi{ht, fd, &dedB};
    try {
        vk::DeviceMemory memB = b.dev.allocateMemory(vk::MemoryAllocateInfo{reqB.size, tB, &imfi});
        b.dev.bindBufferMemory(bufB, memB, 0);
        printf("  [4] device-local memory import (%s): OK (spec-nonconforming)\n", ht == vk::ExternalMemoryHandleTypeFlagBits::eDmaBufEXT ? "DMA_BUF" : "OPAQUE_FD");
    } catch (const vk::SystemError& e) {
        printf("  [4] device-local memory import (%s): FAILED (%s)\n", ht == vk::ExternalMemoryHandleTypeFlagBits::eDmaBufEXT ? "DMA_BUF" : "OPAQUE_FD", e.what());
    }
}

int main(int argc, char** argv) {
    vk::ApplicationInfo app{"test-vk-p2p", 1};
    std::vector<const char*> iexts = { VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME };
    g_inst = vk::createInstance(vk::InstanceCreateInfo{{}, &app, 0, nullptr, (uint32_t)iexts.size(), iexts.data()});

    auto phys = g_inst.enumeratePhysicalDevices();
    std::vector<int> nvidia;
    for (int i = 0; i < (int)phys.size(); i++) {
        if (phys[i].getProperties().vendorID == 0x10de) nvidia.push_back(i);
    }
    if (nvidia.empty()) { printf("no NVIDIA devices found\n"); return 1; }

    if (nvidia.size() < 2 && argc < 3) {
        printf("found %zu NVIDIA device(s); pass two physical-device indices explicitly:\n", nvidia.size());
        for (int i = 0; i < (int) phys.size(); i++) {
            printf("  [%d] %s\n", i, (const char*) phys[i].getProperties().deviceName);
        }
        printf("usage: test-vk-p2p <devA> <devB>\n");
        return 1;
    }
    int ai = nvidia[0], bi = nvidia[1];
    if (argc > 1) ai = atoi(argv[1]);
    if (argc > 2) bi = atoi(argv[2]);

    printf("A = [%d] %s\nB = [%d] %s\n", ai, (const char*)phys[ai].getProperties().deviceName,
           bi, (const char*)phys[bi].getProperties().deviceName);

    Dev a; a.idx = ai; a.phys = phys[ai]; a.dev = mkdev(phys[ai]);
    Dev b; b.idx = bi; b.phys = phys[bi]; b.dev = mkdev(phys[bi]);
    a.qfam = find_qfam(phys[ai], vk::QueueFlagBits::eCompute | vk::QueueFlagBits::eTransfer);
    b.qfam = find_qfam(phys[bi], vk::QueueFlagBits::eCompute | vk::QueueFlagBits::eTransfer);
    a.queue = a.dev.getQueue(a.qfam, 0);
    b.queue = b.dev.getQueue(b.qfam, 0);
    a.pool = a.dev.createCommandPool(vk::CommandPoolCreateInfo{{}, a.qfam});
    b.pool = b.dev.createCommandPool(vk::CommandPoolCreateInfo{{}, b.qfam});

    printf("enabled: external_memory_fd=%d external_semaphore_fd=%d external_memory_dma_buf=%d\n",
           g_have_ext_memory_fd, g_have_ext_semaphore_fd, g_have_ext_dma_buf);

    bool ok = true;
    if (g_have_ext_memory_fd) {
        ok &= test_shared_host_buffer(a, b);
        test_device_local_p2p(a, b);
    } else {
        printf("  [skip] memory tests (no external_memory_fd)\n");
    }
    if (g_have_ext_semaphore_fd) {
        ok &= test_semaphore(a, b, vk::ExternalSemaphoreHandleTypeFlagBits::eSyncFd, "SYNC_FD");
        ok &= test_semaphore(a, b, vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd, "OPAQUE_FD");
    } else {
        printf("  [skip] semaphore tests (no external_semaphore_fd)\n");
    }

    printf("SUMMARY: %s\n", ok ? "PASS" : "FAIL");

    // A semaphore wait that hangs leaves a worker thread stuck inside vkQueueSubmit,
    // which makes vkDestroyDevice block at process exit. Skip clean shutdown in that
    // case so the test doesn't hang the shell.
    if (g_had_timeout) {
        fflush(stdout);
        fflush(stderr);
        _exit(ok ? 0 : 1);
    }
    return ok ? 0 : 1;
}
