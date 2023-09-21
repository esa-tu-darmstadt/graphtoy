#include "gt_buffer_tests.hpp"
#include "graphtoy.hpp"

#include <array>

#include <boost/assert.hpp>

using namespace graphtoy;

struct RawBufferWriteThenReadKernel: GtKernelBase {
    RawBufferWriteThenReadKernel(GtContext *ctx, GtBuffer<int> *buf): GtKernelBase(ctx), m_buf(buf) {}

    GtBuffer<int> *m_buf;
    int m_readVal = 0;

    GtKernelCoro kernelMain() override {
        co_await m_buf->write(1);
        m_readVal = co_await m_buf->read();
    }
};

struct RawBufferReadThenWriteKernel: GtKernelBase {
    RawBufferReadThenWriteKernel(GtContext *ctx, GtBuffer<int> *buf): GtKernelBase(ctx), m_buf(buf) {}

    GtBuffer<int> *m_buf;
    int m_readVal = 0;

    GtKernelCoro kernelMain() override {
        m_readVal = co_await m_buf->read();
        co_await m_buf->write(m_readVal + 1);
    }
};

struct RawBufferWriteWriteReadReadKernel: GtKernelBase {
    RawBufferWriteWriteReadReadKernel(GtContext *ctx, GtBuffer<int> *buf): GtKernelBase(ctx), m_buf(buf) {}

    GtBuffer<int> *m_buf;
    int m_readVal1 = 0;
    int m_readVal2 = 0;

    GtKernelCoro kernelMain() override {
        co_await m_buf->write(10);
        co_await m_buf->write(20);
        m_readVal1 = co_await m_buf->read();
        m_readVal2 = co_await m_buf->read();
    }
};

/**
 * Test the handling of multiple concurrent readers on a single GtBuffer.
 * 
 * This test relies on the implementation details of GtContext.
 * In particular, it relies on the fact that GtContext executes work items in a LIFO fashion.
 */
static void testGtBufferReadReadiness() {
    GtContext ctx{};
    GtBuffer<int> buf{&ctx, 1};

    // Order matters! Last kernel gets executed first.
    const auto& wrKern = ctx.addKernel<RawBufferWriteThenReadKernel>(&buf);
    const auto& rwKern = ctx.addKernel<RawBufferReadThenWriteKernel>(&buf);

    // 1. rwKern tries to read from the buffer, and blocks. -> Context switch
    // 2. wrKern writes 1 into the buffer. rwKern gets readied and claims the element in the buffer.
    // 3. wrKern tries to read from the buffer, and blocks, because the element is claimed. -> Context switch
    // 4. rwKern reads 1 from the buffer.
    // 5. rwKern writes 2 to the buffer and terminates.
    // 6. wrKern reads 2 from the buffer and terminates.
    ctx.runToCompletion();

    BOOST_ASSERT(rwKern.m_readVal == 1);
    BOOST_ASSERT(wrKern.m_readVal == 2);
}

/**
 * Test the handling of multiple concurrent writers on a single GtBuffer.
 * 
 * Relies on the same GtContext implementation details as testGtBufferReadReadiness().
 */
static void testGtBufferWriteReadiness() {
    GtContext ctx{};
    GtBuffer<int> buf{&ctx, 1};

    // Order matters! Last kernel gets executed first.
    const auto& rwKern = ctx.addKernel<RawBufferReadThenWriteKernel>(&buf);
    const auto& wwrrKern = ctx.addKernel<RawBufferWriteWriteReadReadKernel>(&buf);

    // 1. wwrrKern writes 10 into the buffer.
    // 2. wwrrKern tries to write to the buffer, and blocks. -> Context switch
    // 3. rwKern reads 10 from the buffer. wwrrKern gets readied and claims the empty slot in the buffer.
    // 4. rwKern tries to write to the buffer, and blocks, because the empty slot is already claimed. -> Context switch
    // 5. wwrrKern writes 20 to the buffer.
    // 6. wwrrKern reads 20 from the buffer. rwKern gets readied and claims the empty slot in the buffer.
    // 7. wwrrKern tries to read from the buffer, and blocks. -> Context switch
    // 8. rwKern writes 11 to the buffer and terminates.
    // 9. wwrrKern reads 11 from the buffer and terminates.
    ctx.runToCompletion();

    BOOST_ASSERT(rwKern.m_readVal == 10);
    BOOST_ASSERT(wwrrKern.m_readVal1 == 20);
    BOOST_ASSERT(wwrrKern.m_readVal2 == 11);
}

struct RawBufferWriteKernel: GtKernelBase {
    RawBufferWriteKernel(GtContext *ctx, GtBuffer<int> *buf, std::span<const int> values): GtKernelBase(ctx), m_buf(buf), m_values(values) {}

    GtBuffer<int> *m_buf;
    std::span<const int> m_values;

    GtKernelCoro kernelMain() override {
        for (int v: m_values)
            co_await m_buf->write(v);
    }
};

struct RawBufferReadKernel: GtKernelBase {
    RawBufferReadKernel(GtContext *ctx, GtBuffer<int> *buf): GtKernelBase(ctx), m_buf(buf) {}

    GtBuffer<int> *m_buf;
    std::vector<int> m_data;

    GtKernelCoro kernelMain() override {
        while (true)
            m_data.push_back(co_await m_buf->read());
    }
};

struct RawBufferReadOnceKernel: GtKernelBase {
    RawBufferReadOnceKernel(GtContext *ctx, GtBuffer<int> *buf): GtKernelBase(ctx), m_buf(buf) {}

    GtBuffer<int> *m_buf;
    int m_data = 0;

    GtKernelCoro kernelMain() override {
        m_data = co_await m_buf->read();
    }
};

/**
 * Test GtBuffer under high write pressure: Many kernels are blocked waiting to write, and only one kernel
 * eventually drains the backlog.
 * 
 * Again, relies on GtContext and GtBuffer internals:
 * GtContext executes kernels LIFO, and GtBuffer readies waiters LIFO.
 */
static void testGtBufferWriteWaitQueue() {
    GtContext ctx{};
    GtBuffer<int> buf{&ctx, 1};

    const auto testData = std::to_array({1, 2, 4, 19, 82347, 42, 1337, 999, 35, -1, 3});

    // Last kernel gets executed first
    const auto& readKernel = ctx.addKernel<RawBufferReadKernel>(&buf);
    for (const int& val: testData) {
        ctx.addKernel<RawBufferWriteKernel>(&buf, std::span<const int>(&val, size_t(1)));
    }

    // 1. Writer writing the last value does so, and terminates
    // 2. Writer writing the 2nd-to-last value wants to write, and blocks
    // 3. ...
    // 4. Writer writing the first value wants to write, and blocks
    // 5. Reader reads the last value, unblocks most-recently blocked writer
    // 6. Reader wants to read, and blocks
    // 7. Writer writes the first value, readies reader, and terminates
    // 8. Reader reads the first value, unblocks most-recently blocked waiter
    // 9. Reader wants to read, and blocks
    // 10. Writer writes the second value, readies reader, and terminates
    // 11. ...
    // 12. Reader reads the 2nd-to-last value
    // 13. Reader wants to read, and blocks
    // 14. No kernel is ready anymore, execution of the graph terminates
    // => Values read in order {N-1, 0, 1, 2, ..., N-2}
    ctx.runToCompletion();

    const auto& readData = readKernel.m_data;
    BOOST_ASSERT(readData.size() == testData.size());

    for (size_t i = 0; i < testData.size(); ++i) {
        size_t readDataIndex = i + 1;
        if (readDataIndex >= readData.size())
            readDataIndex -= readData.size();

        BOOST_ASSERT(readData.at(readDataIndex) == testData[i]);
    }
}


/**
 * Test GtBuffer under high read pressure: Many kernels are blocked waiting to read, and only one kernel
 * eventually provides data.
 * 
 * Again, relies on GtContext and GtBuffer internals:
 * GtContext executes kernels LIFO, and GtBuffer readies waiters LIFO.
 */
static void testGtBufferReadWaitQueue(bool largeBuffer) {
    GtContext ctx{};

    const auto testData = std::to_array({923, 37, 7324, 183, 193, 23, 59, 129});

    GtBuffer<int> buf{&ctx, largeBuffer ? testData.size() : 1};

    // Last kernel gets executed first
    ctx.addKernel<RawBufferWriteKernel>(&buf, testData);
    std::array<const RawBufferReadOnceKernel *, std::tuple_size_v<decltype(testData)>> readKernels{};
    for (auto& readKernel: readKernels) {
        readKernel = &ctx.addKernel<RawBufferReadOnceKernel>(&buf);
    }

    // SMALL BUFFER CASE:
    // 1. All readers get launched, in reverse order, and block trying to read from the buffer
    // 2. The writer gets launched, writes item 0, then blocks
    // 3. Most recently blocked reader (0) gets unblocked and reads, then terminates
    // 4. Writer writes item 1, blocks again
    // 5. Next reader (1) gets unblocked and reads, terminates
    // 6. ...
    // => Items get read in true order

    // LARGE BUFFER CASE:
    // 1. All readers get launched, in reverse order, and block trying to read from the buffer
    // 2. The writer gets launched, writes item 0, unblocks most recently blocked reader (0)
    // 3. Writer writes item 1, unblocks most recently blocked reader (1)
    // 4. ...
    // 5. All readers are now unblocked, but none has executed yet
    // 6. Writer terminates
    // 7. Most recently unblocked reader (last one) executes, reads item 0
    // 8. Most recently unblocked reader (2nd-to-last one) executes, reads item 1
    // 9. ...
    // => Items get read in reverse order

    ctx.runToCompletion();

    for (size_t i = 0; i < testData.size(); ++i) {
        const auto dataIndex = largeBuffer ? (testData.size() - 1 - i) : i;
        BOOST_ASSERT(testData[dataIndex] == readKernels.at(i)->m_data);
    }
}


void runGraphtoyBufferTests() {
    testGtBufferReadReadiness();
    testGtBufferWriteReadiness();
    testGtBufferWriteWaitQueue();
    testGtBufferReadWaitQueue(/* largeBuffer = */ false);
    testGtBufferReadWaitQueue(/* largeBuffer = */ true);
}
