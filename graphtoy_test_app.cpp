#include "graphtoy.hpp"

#include <utility>
#include <array>

#include <stdexcept>
#include <functional>

#include <boost/assert.hpp>

#include "gt_buffer_tests.hpp"

using namespace graphtoy;

// === EXAMPLE === ADD CONSECUTIVE INTS FROM AN INPUT STREAM ===

/**
 * This kernel takes an input stream containing single int values, and emits an output stream also containing ints.
 * For each two ints consumed, it adds them and then emits the sum.
 */
struct AdderKernel: GtKernelBase {
    // I/O streams have to be created in the kernel constructor using addIoStream<T>().
    // You can optionally pass a desired FIFO depth to it. If none is given, it defaults to 1.
    explicit AdderKernel(GtContext *ctx): GtKernelBase(ctx), m_input(addIoStream<int>()), m_output(addIoStream<int>()) {}

    GtKernelIoStream<int> *m_input;
    GtKernelIoStream<int> *m_output;

private:

    // This is our kernel function. It's the code running on the simulated AIE tile.
    GtKernelCoro kernelMain() override {

        // Most kernels should run in an infinite loop. Graphtoy will terminate the coroutine gracefully when
        // the graph has no more input data.
        while (true) {

            // Read our input values from the input stream.
            // Note: All blocking operations always use co_await.
            // This allows Graphtoy to suspend the kernel and run a different one if the operation would block.
            // As a result, we get concurrent kernel execution without having to use threads.
            int a = co_await m_input->read();
            int b = co_await m_input->read();

            // Write our result.
            // Forgetting the co_await on a stream write call is an easy mistake to make.
            co_await m_output->write(a + b);
        }
    }
};

// This exercises the AdderKernel from above.
// It uses a template just so we can re-use the same test function for another similar kernel later.
template<typename Kernel = AdderKernel>
static void testBasicAdderGraph() {

    // Generate some test data: Inputs and expected results.
    std::vector<int> data = {1, 2, 3, 4, 5, 6};
    std::vector<int> expectedResult{};

    for (size_t i = 0; i < data.size(); i += 2) {
        expectedResult.push_back(data.at(i) + data.at(i + 1));
    }

    // The first thing you have to do when you want to invoke a graph is to construct a GtContext.
    // A context should only live for a single graph invocation.
    // Building a context and adding kernels to it is a cheap operation.
    GtContext ctx{};

    // Next, we instantiate our kernels. GtContext::addKernel invokes the kernel constructor for you,
    // and passes the context as the first constructor argument. You can pass more arguments if needed.
    auto& gmioSource = ctx.addKernel<GtMemStreamSource<int>>(data);     // Behaves as gm2aie_nb
    auto& adder = ctx.addKernel<Kernel>();                              // Our kernel under test
    auto& gmioSink = ctx.addKernel<GtMemStreamSink<int>>();             // Behaves as aie2gm_nb

    // Establish the stream connections between our kernels. Type safety is enforced at compile time.
    // You can specify multiple destinations if you want, which results in a broadcast arrangement,
    // just like on the real AI engines.
    ctx.connect(gmioSource.output(), adder.m_input);
    ctx.connect(adder.m_output, gmioSink.input());

    // Execute our graph. This runs as long as there's any kernel that can run (has available input data).
    // It terminates once all kernels are blocked because there's no more input data.
    // This ensures complete processing of all input data, even though there are no explicit termination conditions.
    // Graphtoy will print warnings in case the kernels are unable to process all input data.
    ctx.runToCompletion();

    // Finally, our processed data is just sitting in the sink.
    BOOST_ASSERT(gmioSink.data() == expectedResult);
}


// === EXAMPLE === BLOCKING KERNEL SUBROUTINES ===

struct AdderKernelWithSubroutine: GtKernelBase {
    explicit AdderKernelWithSubroutine(GtContext *ctx): GtKernelBase(ctx), m_input(addIoStream<int>()), m_output(addIoStream<int>()) {}

    GtKernelIoStream<int> *m_input;
    GtKernelIoStream<int> *m_output;

private:

    // If you want to call a subroutine from a kernel, but that subroutine may block on an I/O stream,
    // you can use GtSubroutineCoro<ReturnType> to turn your subroutine into a (suspendable) coroutine.
    // Subroutines don't necessarily have to be member functions of your kernel, but they can be.
    // If you want to share a subroutine between kernels, just make it a free function.
    GtSubroutineCoro<std::array<int, 2>> getTwoInts() {
        std::array<int, 2> result = {};

        for (auto& entry: result) {
            // This can (and will) block the calling kernel!
            entry = co_await m_input->read();
        }

        // Note the use of co_return here.
        co_return result;
    }

    GtKernelCoro kernelMain() override {
        while (true) {
            // We have to co_await this since it's a coroutine and can suspend.
            auto values = co_await getTwoInts();

            co_await m_output->write(values[0] + values[1]);
        }
    }
};


// === EXAMPLE === PACKET SPLIT / MERGE ===

// This kernel consumes a stream of int pairs, performs a binary operation on them, and emits single ints.
template<typename Fn>
struct BinaryMathKernelSplitMerge: GtKernelBase {
    explicit BinaryMathKernelSplitMerge(GtContext *ctx): GtKernelBase(ctx) {}

    using DataT = std::pair<int, int>;

    // Input packets are still tagged with the destination when they arrive from the splitter.
    using InputT = GtSplitterPacket<DataT>;

    // We'll pass the tag right through, so the packets can be identified after the stream merger.
    using OutputT = GtSplitterPacket<int>;

    GtKernelIoStream<InputT> *m_input = addIoStream<InputT>();
    GtKernelIoStream<OutputT> *m_output = addIoStream<OutputT>();

private:
    GtKernelCoro kernelMain() override {
        Fn fn{};

        while (true) {
            const InputT data = co_await m_input->read();
            const OutputT result = {
                .m_destination = data.m_destination,
                .m_data = fn(data.m_data.first, data.m_data.second)
            };
            co_await m_output->write(result);
        }
    }
};

using BinaryAddKernelSplitMerge = BinaryMathKernelSplitMerge<std::plus<>>;
using BinaryMultiplyKernelSplitMerge = BinaryMathKernelSplitMerge<std::multiplies<>>;

// Demonstrates the use of packet split/merge kernels. A stream of integer pairs is fed into a graph,
// and depending on a tag associated with each pair, it's sent to either an adder kernel, or a multiplier kernel.
// Afterwards, the split streams are merged into one again.
static void testPacketSplitMerge() {
    GtContext ctx{};

    using DataT = BinaryAddKernelSplitMerge::DataT;
    using InputT = BinaryAddKernelSplitMerge::InputT;
    using OutputT = BinaryAddKernelSplitMerge::OutputT;

    static constexpr size_t s_numOpKernels = 2;

    // The data to send into the graph, tagged with where to send it (0 = add, 1 = multiply).
    const auto inputData = std::to_array({
        InputT{.m_destination = 0, .m_data = {1, 2}},
        InputT{.m_destination = 0, .m_data = {3, 4}},
        InputT{.m_destination = 1, .m_data = {5, 3}},
        InputT{.m_destination = 1, .m_data = {6, 3}},
        InputT{.m_destination = 1, .m_data = {2, 5}},
        InputT{.m_destination = 0, .m_data = {9, 1}},
        InputT{.m_destination = 1, .m_data = {13, 5}},
        InputT{.m_destination = 1, .m_data = {4, 5}},
        InputT{.m_destination = 0, .m_data = {1, 4}},
        InputT{.m_destination = 0, .m_data = {3, 4}},
        InputT{.m_destination = 1, .m_data = {5, 3}},
        InputT{.m_destination = 1, .m_data = {3, 6}}
    });

    // Split the data apart so we can check all the add and multiply results separately.
    // This is needed because the packet merger doesn't guarantee the order in which packets
    // are merged into the output stream.
    std::array<std::vector<const InputT *>, s_numOpKernels> inputsSortedByDestination{};
    for (auto& elem: inputData) {
        inputsSortedByDestination[elem.m_destination].push_back(&elem);
    }

    // Our two binary op kernels.
    auto& adder = ctx.addKernel<BinaryAddKernelSplitMerge>();
    auto& multiplier = ctx.addKernel<BinaryMultiplyKernelSplitMerge>();

    // Packet splitter and merger.
    auto& splitter = ctx.addKernel<GtPacketStreamSplitter<DataT>>(s_numOpKernels);
    auto& merger = ctx.addKernel<GtPacketStreamMerger<OutputT>>(s_numOpKernels);

    // Input and output GMIO streams.
    auto& source = ctx.addKernel<GtMemStreamSource<InputT>>(inputData);
    auto& sink = ctx.addKernel<GtMemStreamSink<OutputT>>();

    // Connect it all up...
    ctx.connect(source.output(), splitter.input());
    ctx.connect(splitter.output(0), adder.m_input);
    ctx.connect(splitter.output(1), multiplier.m_input);
    ctx.connect(adder.m_output, merger.input(0));
    ctx.connect(multiplier.m_output, merger.input(1));
    ctx.connect(merger.output(), sink.input());

    //                   dest=0
    //                   .----------> adder ----------.
    //                   |                            v
    // source ------> splitter                      merger -----> sink
    //                   |                            ^
    //                   '--------> multiplier -------'
    //                   dest=1

    // Run the graph!
    ctx.runToCompletion();

    // Now check the results. We have to keep an index per destination, because the merger only guarantees
    // that packets from one source remain in order, but packets from different sources may be interleaved
    // arbitrarily.
    std::array<size_t, s_numOpKernels> indices = {0};

    for (const auto& resultElem: sink.data()) {
        const size_t dest = resultElem.m_destination;
        BOOST_ASSERT(dest < s_numOpKernels);
        auto& testDataIdx = indices[dest];

        const auto testData = inputsSortedByDestination[dest].at(testDataIdx);
        const int a = testData->m_data.first;
        const int b = testData->m_data.second;

        int expectedResult = 0;
        switch (dest) {
            case 0: expectedResult = a + b; break;
            case 1: expectedResult = a * b; break;
        }

        BOOST_ASSERT(resultElem.m_data == expectedResult);

        ++testDataIdx;
    }
}


// === EXAMPLE === INPUT WINDOW OVERLAP ===

// This kernel adds all consecutive number triples in the input stream using an overlap window.
// It is essentially a FIR filter with the coefficients {1, 1, 1}.
struct FirFilterKernelWithOverlapWindow: GtKernelBase {
    explicit FirFilterKernelWithOverlapWindow(GtContext *ctx): GtKernelBase(ctx) {}

    GtKernelIoStream<int> *m_input = addIoStream<int>();
    GtKernelIoStream<int> *m_output = addIoStream<int>();

private:
    GtKernelCoro kernelMain() override {
        // 3-element window with 2 elements being kept between iterations (overlap).
        static constexpr size_t s_winsz = 3;
        static constexpr size_t s_overlap = s_winsz - 1;

        // Instantiate the window itself. It's sufficient to do this in the coroutine, it doesn't have
        // to be a member variable of the kernel class.
        GtOverlapInputWindow<int> inputWindow{m_input, s_winsz, s_overlap};

        while (true) {
            // A GtOverlapInputWindow must be advanced before the first access.
            // Since this can block (it's a GtSubroutineCoro), await it.
            co_await inputWindow.advance();

            // Sum up all elements in the window.
            int result = 0;
            for (size_t i = 0; i < s_winsz; ++i)
                result += inputWindow.at(i);

            co_await m_output->write(result);
        }
    }
};

template<size_t FirSz, size_t N>
static constexpr std::array<int, N + 1 - FirSz> firFilter(const std::array<int, N>& data) {
    std::array<int, N + 1 - FirSz> result{};

    for (size_t i = 0; i < result.size(); ++i)
        for (size_t j = 0; j < FirSz; ++j)
            result[i] += data[i + j];

    return result;
}

// The kernel using the overlap window behaves just like any other kernel.
// Instantiate it, hook it up to a stream source and sink, and run some data through it.
static void testFirFilterWithOverlapWindow() {
    GtContext ctx{};

    static constexpr auto testData = std::to_array({9, 138, 4, 84, 27, 52, 284, 73, 192, 5744, 194, 1, 0, 4});
    static constexpr auto expectedResult = firFilter<3>(testData);

    auto& source = ctx.addKernel<GtMemStreamSource<int>>(testData);
    auto& sink = ctx.addKernel<GtMemStreamSink<int>>();
    auto& fir = ctx.addKernel<FirFilterKernelWithOverlapWindow>();

    ctx.connect(source.output(), fir.m_input);
    ctx.connect(fir.m_output, sink.input());

    ctx.runToCompletion();

    const auto& graphResult = sink.data();
    BOOST_ASSERT(graphResult.size() == expectedResult.size());

    for (size_t i = 0; i < expectedResult.size(); ++i) {
        BOOST_ASSERT(graphResult.at(i) == expectedResult[i]);
    }
}


int main() {
    runGraphtoyBufferTests();

    testBasicAdderGraph();
    testBasicAdderGraph<AdderKernelWithSubroutine>();

    testPacketSplitMerge();

    testFirFilterWithOverlapWindow();

    return 0;
}
