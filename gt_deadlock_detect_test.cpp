#include "graphtoy.hpp"

#include <array>

using namespace graphtoy;

struct DerpKernel: GtKernelBase {
    explicit DerpKernel(GtContext *ctx): GtKernelBase(ctx) {}

    GtKernelIoStream<int> *m_input = addIoStream<int>();
    GtKernelIoStream<size_t> *m_input2 = addIoStream<size_t>();

private:
    GtKernelCoro kernelMain() override {
        co_return; // :D
    }
};

template<typename T>
struct SourceKernel: GtKernelBase {
    explicit SourceKernel(GtContext *ctx): GtKernelBase(ctx) {}

    GtKernelIoStream<T> *m_output = addIoStream<T>();

private:
    GtKernelCoro kernelMain() override {
        T val = 0;

        while (true) {
            co_await m_output->write(++val);
        }
    }
};

static constexpr size_t g_numSources = 3;

int main() {
    GtContext ctx{};

    std::array<SourceKernel<int> *, g_numSources> sources;
    for (auto& src: sources) {
        src = &ctx.addKernel<SourceKernel<int>>();
    }

    auto& merger = ctx.addKernel<GtPacketStreamMerger<int>>(g_numSources);
    auto& anotherSource = ctx.addKernel<SourceKernel<size_t>>();
    auto& derp = ctx.addKernel<DerpKernel>();

    for (size_t i = 0; i < sources.size(); ++i) {
        ctx.connect(sources[i]->m_output, merger.input(i));
    }

    ctx.connect(merger.output(), derp.m_input);
    ctx.connect(anotherSource.m_output, derp.m_input2);

    // This will warn about a deadlock
    ctx.runToCompletion();

    return 0;
}
