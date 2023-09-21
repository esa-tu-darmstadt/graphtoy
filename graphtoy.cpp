#include "graphtoy.hpp"

#include <iostream>
#include <algorithm>
#include <functional>


struct graphtoy::GtStuckBufferInfo {
    size_t m_ioStreamIndex;
    const std::type_info *m_dataType;
    bool m_isOutputFromKernel;
    size_t m_fifoCapacity;
    size_t m_fifoInUseEntries;
};

static void printAsyncStackTrace(const std::vector<std::source_location>& trace, size_t indentation = 0) {
    std::string indentStr(indentation, ' ');

    size_t i = 1;

    for (const auto& sourceLoc: trace) {
        std::cerr << indentStr << "#" << i << " in " << sourceLoc.function_name()
                  << " (" << sourceLoc.file_name() << ", line " << sourceLoc.line() << ", col " << sourceLoc.column() << ")\n";
        ++i;
    }

    if (trace.empty()) {
        std::cerr << indentStr << "(Coroutine has already returned)\n";
    }
}

void graphtoy::printCoroException(std::exception_ptr exptr, const std::vector<std::source_location>& trace) {
    std::cerr << "FATAL: Graphtoy caught an exception while executing a kernel.\n";

    try {
        std::rethrow_exception(exptr);
    } catch (const std::exception& ex) {
        std::cerr << "Exception type: " << typeid(ex).name() << "\n"
                  << "        what(): " << ex.what() << "\n";
    } catch (...) {
        std::cerr << "(Exception is not derived from std::exception)\n";
    }

    std::cerr << "Coroutine frame trace, deepest last:\n";
    printAsyncStackTrace(trace, 2);
    std::cerr << std::endl;
}

void graphtoy::GtContext::runToCompletion() {
    BOOST_ASSERT(m_runningCoros.empty());

    m_runningCoros.reserve(m_kernels.size());
    m_workItems.reserve(m_kernels.size());

    for (auto& kernel: m_kernels) {
        std::coroutine_handle<> coro = kernel.startUp();
        m_runningCoros.push_back(coro);
        m_workItems.push_back(coro);
    }

    while (!m_workItems.empty()) {
        std::coroutine_handle<> workItem = m_workItems.back();
        m_workItems.pop_back();
        workItem.resume();
    }

    bool printDeadlockHint = false;

    size_t i = 0;
    for (const auto& kernel: m_kernels) {
        if (auto stuckBuffers = kernel.checkForStuckBuffers(); !stuckBuffers.empty()) {
            std::cerr << "WARNING: Graphtoy detected data stuck in one or more I/O buffers of kernel " << i << " (an instance of " << typeid(kernel).name() << ").\n";

            bool anyOutputsBlocked = false;
            for (const auto& buf: stuckBuffers) {
                anyOutputsBlocked |= buf.m_isOutputFromKernel && (buf.m_fifoInUseEntries >= buf.m_fifoCapacity);

                std::cerr << "=> " << (buf.m_isOutputFromKernel ? "Output" : "Input") << " stream " << buf.m_ioStreamIndex
                          << " (of type " << buf.m_dataType->name() << ") contains unprocessed data: "
                          << buf.m_fifoInUseEntries << " of " << buf.m_fifoCapacity << " FIFO entries in use.\n";
            }

            if (!kernel.suppressStuckBufferBacktraceForThisKernel()) {
                std::cerr << "=> Currently active coroutine frames in this kernel (frame trace), deepest last:\n";
                printAsyncStackTrace(kernel.getAsyncStackTrace(), 3);
            } else {
                std::cerr << "=> (Backtrace is suppressed for this kernel)\n";
            }

            if (!anyOutputsBlocked) {
                std::cerr << "=> Note: This kernel is likely the cause of the deadlock because none of its output streams are blocked.\n";
            }

            std::cerr << std::endl;

            printDeadlockHint = true;
        }

        ++i;
    }

    if (printDeadlockHint) {
        std::cerr << "WARNING: The previous warnings indicate a buffer back-pressure deadlock in your graph.\n"
                     "This means that one or more kernels still have data to process, but can't do so because they're\n"
                     "stuck waiting for one of their other I/O streams to become available for reading or writing.\n"
                     "Graph execution will be terminated because no kernel can run anymore.\n"
                     "Please review the architecture of your graph for potential deadlocks using the information above.\n"
                     "Continuing anyway, but the results of this graph invocation are incomplete.\n" << std::endl;
    }
}

graphtoy::GtContext::~GtContext() {
    for (auto& coro: m_runningCoros) {
        coro.destroy();
    }

    while (!m_kernels.empty()) {
        auto *kernel = &m_kernels.back();
        m_kernels.pop_back();
        delete kernel;
    }
}

graphtoy::GtKernelCoro graphtoy::GtKernelBase::startUp() {
    BOOST_ASSERT(!m_coro);

    for (auto& stream: m_ioStreams) {
        BOOST_ASSERT_MSG(stream.isConnectedToAxiStreamNetwork(), "Unconnected kernel I/O stream!");
    }

    GtKernelCoro coro = kernelMain();
    m_coro = &coro.promise();
    return coro;
}

std::vector<graphtoy::GtStuckBufferInfo> graphtoy::GtKernelBase::checkForStuckBuffers() const {
    std::vector<GtStuckBufferInfo> result{};

    if (ignoreThisKernelInStuckBufferChecks())
        return result;

    using BeginFn = uintptr_t (GtKernelBase::*)() const;
    using DerefFn = const GtKernelIoStreamBase * (GtKernelBase::*)(uintptr_t) const;
    using AdvanceFn = uintptr_t (GtKernelBase::*)(uintptr_t) const;

    const auto checkStreams = [this, &result](BeginFn begin, DerefFn deref, AdvanceFn advance) {
        size_t i = 0;

        for (auto iter = (this->*begin)(); true; iter = (this->*advance)(iter), ++i) {
            const auto *stream = (this->*deref)(iter);
            if (!stream) break;

            if (const auto inUseEntries = stream->fifoInUseEntries()) {
                result.emplace_back(GtStuckBufferInfo{
                    .m_ioStreamIndex = i,
                    .m_dataType = &stream->getDataType(),
                    .m_isOutputFromKernel = stream->isOutputFromKernel(),
                    .m_fifoCapacity = stream->fifoCapacity(),
                    .m_fifoInUseEntries = inUseEntries
                });
            }
        }
    };

    // Check all of our own streams for stuck buffers
    checkStreams(&GtKernelBase::beginStreamList, &GtKernelBase::derefIoStreamIterator, &GtKernelBase::advanceIoStreamIterator);

    // Also check any additional streams that the kernel implementation wants us to check
    checkStreams(&GtKernelBase::beginAdditionalIoStreamList, &GtKernelBase::derefAdditionalIoStreamIterator, &GtKernelBase::advanceAdditionalIoStreamIterator);

    return result;
}

std::vector<std::source_location> graphtoy::GtKernelBase::getAsyncStackTrace() const {
    BOOST_ASSERT(m_coro);
    return m_coro->getAsyncStackTrace();
}


uintptr_t graphtoy::GtKernelBase::beginStreamList() const {
    const GtKernelIoStreamBase *stream = m_ioStreams.empty() ? nullptr : &m_ioStreams.front();
    return uintptr_t(stream);
}

const graphtoy::GtKernelIoStreamBase * graphtoy::GtKernelBase::derefIoStreamIterator(uintptr_t iter) const {
    return (const GtKernelIoStreamBase *)(iter);
}

uintptr_t graphtoy::GtKernelBase::advanceIoStreamIterator(uintptr_t rawIter) const {
    auto *stream = (const GtKernelIoStreamBase *)(rawIter);
    if (!stream) return uintptr_t(nullptr);

    auto iter = m_ioStreams.iterator_to(*stream);
    ++iter;

    if (iter == m_ioStreams.end()) return uintptr_t(nullptr);

    stream = std::to_address(iter);
    return uintptr_t(stream);
}

graphtoy::GtKernelBase::~GtKernelBase() {
    while (!m_ioStreams.empty()) {
        auto *stream = &m_ioStreams.back();
        m_ioStreams.pop_back();
        delete stream;
    }
}
