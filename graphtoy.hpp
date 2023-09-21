#pragma once

#include <cstdlib>
#include <coroutine>
#include <optional>
#include <memory>
#include <vector>
#include <span>
#include <type_traits>
#include <typeinfo>
#include <source_location>
#include <deque>
#include <exception>
#include <bit>

#include <boost/assert.hpp>
#include <boost/circular_buffer.hpp>
#include <boost/container/small_vector.hpp>
#include <boost/intrusive/list.hpp>

namespace graphtoy {

void printCoroException(std::exception_ptr exptr, const std::vector<std::source_location>& trace);

struct GtContext;

struct GtKernelCoroPromise;
struct GtKernelBase;

template<typename T>
struct GtKernelIoStream;

template<typename T>
struct GtStreamTransferKernel;

template<typename T>
using GtSmallVector = boost::container::small_vector<T, 1>;


struct GtContextBase {
    // Internal function, you shouldn't call this from user code.
    void addWorkItem(std::coroutine_handle<> h) {
        m_workItems.push_back(h);
    }

private:
    // Pending work items (coroutines that are ready to run and should be resumed).
    std::vector<std::coroutine_handle<>> m_workItems{};

    GtContextBase() = default;
    GtContextBase(const GtContextBase&) = delete;
    GtContextBase& operator= (const GtContextBase&) = delete;
    ~GtContextBase() = default;

    friend struct GtContext;
};


/**
 * This buffer class is the core of Graphtoy: It allows passing data between kernels and arranges for
 * suspension and resumption of kernel coroutines when they have to wait for input data or space in their
 * output buffers.
 *
 * Implements a multi-producer, multi-consumer bounded FIFO.
 *
 * User code shouldn't have to instantiate this class directly; use GtKernelIoStream instead (via addIoStream).
 */
template<typename T>
struct GtBuffer final {
    explicit GtBuffer(GtContextBase *context, size_t fifoCapacity = 1): m_context(context), m_fifo(fifoCapacity) {
        BOOST_ASSERT(m_fifo.capacity() > 0);
        BOOST_ASSERT(m_fifo.empty());
    }
    ~GtBuffer() = default;

    GtBuffer(const GtBuffer&) = delete;
    GtBuffer& operator= (const GtBuffer&) = delete;

    // Read an item from this buffer; block if none is available.
    auto read() {
        struct GtBufferReadAwaiter {
            GtBuffer *m_self;

            // Returns true if an item can be read immediately (no suspension needed), false otherwise (must suspend).
            bool await_ready() {
                const bool ready = !m_self->empty();

                // If we're going to read immediately, mark our read as in-flight until we complete it.
                if (ready)
                    ++m_self->m_inFlightReads;

                return ready;
            }

            void await_suspend(std::coroutine_handle<> h) {
                m_self->m_readWaiters.push_back(h);
            }

            T await_resume() {
                // Item is available now, grab it.
                return m_self->yoink();
            }
        };

        return GtBufferReadAwaiter{this};
    }

    auto write(T val) {
        struct GtBufferWriteAwaiter {
            GtBuffer *m_self;
            T m_value;

            // Returns true if an item can be written immediately (no suspension needed), false otherwise (must suspend).
            bool await_ready() {
                const bool ready = !m_self->full();

                // If we're going to write immediately, mark our write as in-flight until we complete it.
                if (ready)
                    ++m_self->m_inFlightWrites;

                return ready;
            }

            void await_suspend(std::coroutine_handle<> h) {
                m_self->m_writeWaiters.push_back(h);
            }

            void await_resume() {
                // There's space in the buffer now, write our item.
                m_self->yeet(std::move(m_value));
            }
        };

        return GtBufferWriteAwaiter{this, std::move(val)};
    }

    // Returns the maximum number of items in the FIFO.
    size_t fifoCapacity() const {
        return m_fifo.capacity();
    }

    // Returns the number of items currently present in the FIFO, ignoring in-flight reads and writes.
    size_t fifoInUseEntries() const {
        return m_fifo.size();
    }

    // Returns the number of items in the FIFO that can still be claimed by a read.
    size_t fifoReadClaimableEntries() const {
        BOOST_ASSERT(m_fifo.size() >= m_inFlightReads);
        return m_fifo.size() - m_inFlightReads;
    }

    // Returns the number of FIFO slots that are currently occupied, including ones claimed by in-flight writes.
    size_t fifoWriteClaimedEntries() const {
        return m_fifo.size() + m_inFlightWrites;
    }

    // Returns true if an incoming write must block. (All FIFO slots claimed by writes)
    bool full() const {
        return fifoWriteClaimedEntries() >= fifoCapacity();
    }

    // Returns true if an incoming read must block. (No unclaimed items in the FIFO)
    bool empty() const {
        return fifoReadClaimableEntries() == 0;
    }

private:
    GtContextBase *m_context;

    // Backing storage.
    boost::circular_buffer<T> m_fifo;

    // Number of reads that have already claimed an element from the FIFO, but haven't yet gotten around to
    // removing that element from the FIFO.
    size_t m_inFlightReads = 0;

    // Number of writes that have already claimed a slot in the FIFO, but not yet written their data into it.
    size_t m_inFlightWrites = 0;

    // List of coroutines blocked trying to read and write, respectively.
    GtSmallVector<std::coroutine_handle<>> m_readWaiters = {};
    GtSmallVector<std::coroutine_handle<>> m_writeWaiters = {};

    // Checks if there are any waiters in the respective list, and if yes, readies one.
    void readyWaiter(decltype(m_readWaiters)& waiters) {
        if (!waiters.empty()) {
            m_context->addWorkItem(waiters.back());
            waiters.pop_back();

            if (&waiters == &m_readWaiters) {
                // We're readying a reader and telling it to grab an item from the FIFO, so we should mark
                // that item as claimed now (= add an in-flight read).
                ++m_inFlightReads;
                BOOST_ASSERT(m_fifo.size() >= m_inFlightReads);

            } else if (&waiters == &m_writeWaiters) {
                // We're reading a writer and telling it that it can place an item into the FIFO, so we should
                // mark another slot as claimed now (= add an in-flight write).
                ++m_inFlightWrites;
                BOOST_ASSERT(fifoWriteClaimedEntries() <= fifoCapacity());

            } else {
                std::abort(); // Unreachable
            }
        }
    }

    // Removes an item from the FIFO and returns it, thereby completing an in-flight read.
    // Readies one write waiter, if there is any.
    T yoink() {
        BOOST_ASSERT(!m_fifo.empty());
        BOOST_ASSERT(m_inFlightReads);
        T result = std::move(m_fifo.front());
        m_fifo.pop_front();
        --m_inFlightReads;

        BOOST_ASSERT(!full());
        readyWaiter(m_writeWaiters);

        return result;
    }

    // Adds an item to the FIFO, thereby completing an in-flight write.
    // Readies one reader, if there is any.
    void yeet(T val) {
        BOOST_ASSERT(!m_fifo.full());
        BOOST_ASSERT(m_inFlightWrites);
        m_fifo.push_back(std::move(val));
        --m_inFlightWrites;

        BOOST_ASSERT(!empty());
        readyWaiter(m_readWaiters);
    }
};


/**
 * Represents one coroutine frame on a "call stack" of nested coroutines.
 * Implements a linked list using shared_ptrs to represent the entire call stack.
 * Users should not have to interact with this class.
 */
struct GtAsyncStackTraceEntry {
    // Next deeper nested frame.
    std::shared_ptr<GtAsyncStackTraceEntry> m_next = nullptr;

    // Root frame (the kernel at the root of this call stack).
    GtAsyncStackTraceEntry *m_root = this;

    // Source location where the coroutine frame represented by this stack trace entry is currently suspended.
    std::source_location m_suspendLocation{};

    // True if the coroutine has already returned.
    bool m_done = false;

    // Builds and returns a stack trace (from this stack entry and deeper).
    std::vector<std::source_location> getAsyncStackTrace() const {
        if (m_done)
            return {};

        std::vector<std::source_location> result{};

        for (const GtAsyncStackTraceEntry *entry = this; entry != nullptr; entry = entry->m_next.get()) {
            result.push_back(entry->m_suspendLocation);
        }

        return result;
    }
};


/**
 * Mixin base class that provides asynchronous stack tracing capabilities to any coroutine promises that
 * derive from it. Also provides the GtContextBase to invoked sub-coroutines if they need it.
 *
 * Implements initial_suspend, final_suspend, unhandled_exception, and await_transform.
 * Deriving promises should either not implement these themselves, or call the base class version
 * in their own implementation if they decide to override them.
 *
 * Users should not have to interact with this class.
 */
struct GtAsyncStackTracerPromiseMixin {
    GtContextBase *m_context;
    std::shared_ptr<GtAsyncStackTraceEntry> m_stackTraceEntry = std::make_shared<GtAsyncStackTraceEntry>();

private:
    template<typename T>
    static constexpr bool s_isTraceableAwaitable = requires (const T& t) {
        { t.getAsyncStackTraceEntry() } -> std::same_as<std::shared_ptr<GtAsyncStackTraceEntry>>;
    };

    template<typename T>
    static constexpr bool s_awaitableNeedsContext = requires (T& t, GtContextBase *ctx) {
        { t.setGtContext(ctx) } -> std::same_as<void>;
    };

    template<typename T>
    void extendStackTraceChain(const T& awaitable) {
        m_stackTraceEntry->m_next = awaitable.getAsyncStackTraceEntry();
        m_stackTraceEntry->m_next->m_root = m_stackTraceEntry->m_root;
    }

    template<typename T>
    void giveContextToAwaitable(T& awaitable) {
        awaitable.setGtContext(m_context);
    }

public:
    explicit GtAsyncStackTracerPromiseMixin(GtContextBase *context): m_context(context) {}

    // Snoops on any sub-coroutines (and other awaitables) that get invoked and adds them to the
    // stack trace, if possible. Also provides the GtContextBase to the invoked awaitable if requested.
    template<typename T>
    auto&& await_transform(T&& awaitable, std::source_location loc = std::source_location::current()) {
        BOOST_ASSERT(m_context);

        m_stackTraceEntry->m_suspendLocation = std::move(loc);

        if constexpr (s_isTraceableAwaitable<T>) {
            extendStackTraceChain(awaitable);
        } else {
            m_stackTraceEntry->m_next = nullptr;
        }

        if constexpr (s_awaitableNeedsContext<T>) {
            giveContextToAwaitable(awaitable);
        }

        return std::forward<T>(awaitable);
    }

    std::suspend_always initial_suspend() noexcept { return {}; }

    std::suspend_always final_suspend() noexcept {
        m_stackTraceEntry->m_next = nullptr;
        m_stackTraceEntry->m_done = true;
        return {};
    }

    std::vector<std::source_location> getAsyncStackTrace() const {
        return m_stackTraceEntry->getAsyncStackTrace();
    }

    // Print an async stack trace when an exception is thrown, then terminate the program.
    void unhandled_exception(std::source_location loc = std::source_location::current()) {
        m_stackTraceEntry->m_suspendLocation = std::move(loc);
        m_stackTraceEntry->m_next = nullptr;

        printCoroException(std::current_exception(), m_stackTraceEntry->m_root->getAsyncStackTrace());
        std::abort();
    }
};


/**
 * Mixin base class to store return values of sub-coroutines.
 *
 * Users should not have to interact with this class.
 */
template<typename RetVal>
struct GtCoroReturnValueStorageMixin {
    std::optional<RetVal> m_returnValue = std::nullopt;

    void return_value(RetVal v) {
        BOOST_ASSERT(!m_returnValue.has_value());
        m_returnValue = std::move(v);
    }

    RetVal getReturnValue() {
        BOOST_ASSERT(m_returnValue.has_value());
        return std::move(m_returnValue.value());
    }
};

template<>
struct GtCoroReturnValueStorageMixin<void> {
    void return_void() {}
    void getReturnValue() {}
};

template<typename RetVal>
struct GtSubroutineCoro;


/**
 * Promise for sub-coroutines that can be invoked from kernels (and from other sub-coroutines).
 */
template<typename RetVal>
struct GtSubroutineCoroPromise final: GtAsyncStackTracerPromiseMixin, GtCoroReturnValueStorageMixin<RetVal> {
    std::coroutine_handle<> m_waiter = nullptr;

    GtSubroutineCoroPromise(): GtAsyncStackTracerPromiseMixin(nullptr) {}

    GtSubroutineCoro<RetVal> get_return_object();

    auto final_suspend() noexcept {
        // Resume the coroutine that's waiting on us when we finish execution.
        // This essentially implements a synchronous return.
        BOOST_ASSERT(m_context && m_waiter);
        m_context->addWorkItem(m_waiter);
        m_waiter = nullptr;
        return GtAsyncStackTracerPromiseMixin::final_suspend();
    }

    void setGtContext(GtContextBase *ctx) {
        BOOST_ASSERT(!m_context);
        m_context = ctx;
    }
};


/**
 * Coroutine handle (and return value) of sub-coroutines.
 * This is what the user co_awaits when they call a sub-coroutine.
 * RAII handle; destroying this also destroys the coroutine.
 */
template<typename RetVal>
struct GtSubroutineCoro final: std::coroutine_handle<GtSubroutineCoroPromise<RetVal>> {
    using promise_type = GtSubroutineCoroPromise<RetVal>;
    explicit GtSubroutineCoro(std::coroutine_handle<promise_type> h): std::coroutine_handle<promise_type>(h) {}

    GtSubroutineCoro(const GtSubroutineCoro&) = delete;
    GtSubroutineCoro& operator= (const GtSubroutineCoro&) = delete;

    ~GtSubroutineCoro() {
        this->destroy();
    }

    // Allow this class to be used with co_await.
    auto operator co_await() {
        struct SubroutineAwaiter {
            promise_type *m_promise;

            // Sub-coroutines always have to do something, they are never ready immediately.
            bool await_ready() { return false; }

            void await_suspend(std::coroutine_handle<> waiter) {
                // Store the calling coroutine so we can resume it when we're done.
                GtContextBase *ctx = m_promise->m_context;
                BOOST_ASSERT(ctx);
                BOOST_ASSERT(!m_promise->m_waiter);
                m_promise->m_waiter = waiter;

                // Now that the caller is suspended, we can resume ourselves and invoke the user code.
                ctx->addWorkItem(std::coroutine_handle<promise_type>::from_promise(*m_promise));
            }

            auto await_resume() {
                // The result of the co_await expression is whatever was co_returned by the sub-coroutine.
                return m_promise->getReturnValue();
            }
        };

        return SubroutineAwaiter{.m_promise = &this->promise()};
    }

    // Request the GtContextBase from GtAsyncStackTracerPromiseMixin.
    void setGtContext(GtContextBase *ctx) {
        this->promise().setGtContext(ctx);
    }

    // Allow GtAsyncStackTracerPromiseMixin to do coroutine frame traces into this sub-coroutine.
    auto getAsyncStackTraceEntry() const {
        return this->promise().m_stackTraceEntry;
    }
};

template<typename RetVal>
GtSubroutineCoro<RetVal> GtSubroutineCoroPromise<RetVal>::get_return_object() {
    return GtSubroutineCoro{std::coroutine_handle<GtSubroutineCoroPromise>::from_promise(*this)};
}


/**
 * GtKernelCoro is just a regular dumb coroutine_handle, with no added logic.
 *
 * Users shouldn't interact with this class, apart from returning it from their kernelMain.
 */
struct GtKernelCoro final: std::coroutine_handle<GtKernelCoroPromise> {
    using promise_type = GtKernelCoroPromise;
    explicit GtKernelCoro(std::coroutine_handle<promise_type> h): std::coroutine_handle<promise_type>(h) {}
};


/**
 * Almost all of the logic of the kernel coroutine promise is handled in GtAsyncStackTracerPromiseMixin,
 * so this is rather bare-bones.
 *
 * Users shouldn't have to interact with this class.
 */
struct GtKernelCoroPromise final: GtAsyncStackTracerPromiseMixin {
    template<typename Self>
    explicit GtKernelCoroPromise(Self& self): GtAsyncStackTracerPromiseMixin(self.m_context) {}

    GtKernelCoro get_return_object() {
        return GtKernelCoro{std::coroutine_handle<GtKernelCoroPromise>::from_promise(*this)};
    }

    void return_void() {}
};


/**
 * Common functions that have to be available on all GtKernelIoStreams regardless of their type.
 * Used for error checking and to generate deadlock debug output.
 */
struct GtKernelIoStreamBase: boost::intrusive::list_base_hook<> {
    virtual ~GtKernelIoStreamBase() = default;
    virtual bool isConnectedToAxiStreamNetwork() const = 0;
    virtual const std::type_info& getDataType() const = 0;
    virtual bool isOutputFromKernel() const = 0;
    virtual size_t fifoCapacity() const = 0;
    virtual size_t fifoInUseEntries() const = 0;
};


/**
 * Represents an input or output stream of a graph kernel.
 * Should be created via GtKernelBase::addIoStream.
 *
 * This is basically just a thin wrapper around a GtBuffer.
 *
 * Kernels can read and write data to these streams by co_awaiting the read and write functions, respectively.
 */
template<typename T>
struct GtKernelIoStream final: GtKernelIoStreamBase {
    // Write a value to this output stream. Don't forget to co_await the return value!!
    auto write(T val) {
        BOOST_ASSERT(!m_isTransferDest);
        return m_buf.write(std::move(val));
    }

    // Read a value from this input stream. Don't forget to co_await the return value!!
    auto read() {
        BOOST_ASSERT(!m_isTransferSource);
        return m_buf.read();
    }

    bool isConnectedToAxiStreamNetwork() const override {
        return m_isTransferSource || m_isTransferDest;
    }

    size_t fifoCapacity() const override {
        return m_buf.fifoCapacity();
    }

    size_t fifoInUseEntries() const override {
        return m_buf.fifoInUseEntries();
    }

    const std::type_info& getDataType() const override {
        return typeid(T);
    }

    bool isOutputFromKernel() const override {
        return m_isTransferSource;
    }

    virtual ~GtKernelIoStream() override = default;

private:
    GtBuffer<T> m_buf{};

    // Remember whether we've been connected to the AXI stream network as a source or destination.
    bool m_isTransferSource = false;
    bool m_isTransferDest = false;

    GtKernelIoStream(GtContextBase *context, size_t fifoCapacity): m_buf(context, fifoCapacity) {}

    friend struct GtStreamTransferKernel<T>;
    friend struct GtKernelBase;
};

struct GtStuckBufferInfo;

/**
 * Base class for graph kernels. User kernels should derive from this (directly or indirectly).
 */
struct GtKernelBase: boost::intrusive::list_base_hook<> {
    virtual ~GtKernelBase();

    GtKernelBase(const GtKernelBase&) = delete;
    GtKernelBase& operator= (const GtKernelBase&) = delete;

protected:
    explicit GtKernelBase(GtContextBase *context): m_context(context) {}

    /**
     * Add an input or output stream to this kernel with the given FIFO size.
     * You should typically store the return value of this function in a member variable of your kernel.
     *
     * Graphtoy doesn't distinguish between input and output at stream creation time; instead,
     * the usage of the stream decides whether it will be considered an input or output.
     */
    template<typename T>
    GtKernelIoStream<T> * addIoStream(size_t fifoCapacity = 1) {
        auto *stream = new GtKernelIoStream<T>(m_context, fifoCapacity);
        m_ioStreams.push_back(*stream);
        return stream;
    }

private:
    GtContextBase *m_context;
    boost::intrusive::list<GtKernelIoStreamBase> m_ioStreams = {};
    const GtKernelCoroPromise *m_coro = nullptr;

    // Entry point into the kernel; users must override this and place their kernel logic in this function.
    virtual GtKernelCoro kernelMain() = 0;

    // Launch the kernel; to be invoked by GtContext only.
    GtKernelCoro startUp();

    // Returns any I/O streams that still have data stuck in their FIFOs.
    std::vector<GtStuckBufferInfo> checkForStuckBuffers() const;

    // Returns a coroutine frame trace of this kernel's running coroutine.
    std::vector<std::source_location> getAsyncStackTrace() const;

    uintptr_t beginStreamList() const;
    const GtKernelIoStreamBase * derefIoStreamIterator(uintptr_t iter) const;
    uintptr_t advanceIoStreamIterator(uintptr_t iter) const;

    // Used by GtPacketStreamMerger to make the deadlock debugging logic check its child kernels for stuck I/O buffers too.
    virtual uintptr_t beginAdditionalIoStreamList() const { return 0; }
    virtual const GtKernelIoStreamBase * derefAdditionalIoStreamIterator(uintptr_t) const { return nullptr; }
    virtual uintptr_t advanceAdditionalIoStreamIterator(uintptr_t) const { return 0; }

    // Used by GtPacketStreamMerger to make the deadlock debugging logic suppress implementation details that can't cause
    // a deadlock (no cycles) and would otherwise confuse users.
    virtual bool ignoreThisKernelInStuckBufferChecks() const { return false; }
    virtual bool suppressStuckBufferBacktraceForThisKernel() const { return false; }

    friend struct GtKernelCoroPromise;
    friend struct GtContext;
};


/**
 * The main class of Graphtoy. Holds all the kernels and connections between them.
 * This should be the first thing to instantiate when you want to run a graph.
 */
struct GtContext final: GtContextBase {
    // Execute the graph until it finishes processing all input data (or deadlocks).
    void runToCompletion();

    GtContext() = default;

    GtContext(const GtContext&) = delete;
    GtContext& operator= (const GtContext&) = delete;
    
    ~GtContext();

    /**
     * Create a new kernel and add it to this graph.
     * You can pass additional arguments, which will get passed on to the kernel constructor.
     */
    template<typename Kernel, typename... Args>
    Kernel& addKernel(Args&&... args) {
        auto *kernel = new Kernel(this, std::forward<Args>(args)...);
        m_kernels.push_back(*kernel);
        return *kernel;
    }

    // Connects the given output stream to the given input streams in a broadcast arrangement.
    template<typename T>
    void connect(GtKernelIoStream<T> *source, GtSmallVector<GtKernelIoStream<T> *> dests);

    // Connects the given output stream (source) to the given input stream (dest).
    template<typename T>
    void connect(GtKernelIoStream<T> *source, GtKernelIoStream<T> *dest) {
        connect(source, GtSmallVector<GtKernelIoStream<T> *>{dest});
    }

private:
    boost::intrusive::list<GtKernelBase> m_kernels = {};

    // All coroutines that have been launched and not yet destroyed.
    std::vector<std::coroutine_handle<>> m_runningCoros = {};
};


/**
 * Simple kernel that copies data from one I/O stream into another.
 * This is the implementation behind GtContext::connect().
 *
 * Users should not have to interact with this class.
 */
template<typename T>
struct GtStreamTransferKernel final: GtKernelBase {
    GtStreamTransferKernel(GtContext *context, GtKernelIoStream<T> *source, GtSmallVector<GtKernelIoStream<T> *> dests):
        GtKernelBase(context), m_source(source), m_dests(std::move(dests))
    {
        // If one of the asserts below fires, the user tried to connect an I/O stream multiple times.

        BOOST_ASSERT(!source->m_isTransferSource);
        source->m_isTransferSource = true;

        for (auto& dest: m_dests) {
            BOOST_ASSERT(!dest->m_isTransferDest);
            dest->m_isTransferDest = true;
        }
    }

private:
    GtKernelIoStream<T> *m_source;
    GtSmallVector<GtKernelIoStream<T> *> m_dests;

    GtKernelCoro kernelMain() override {
        while (true) {
            T val = co_await m_source->m_buf.read();
            for (auto& dest: m_dests) {
                co_await dest->m_buf.write(val);
            }
        }
    }
};

template<typename T>
void GtContext::connect(GtKernelIoStream<T> *source, GtSmallVector<GtKernelIoStream<T> *> dests) {
    addKernel<GtStreamTransferKernel<T>>(source, std::move(dests));
}


[[maybe_unused]] static size_t idealFifoSizeFor(size_t numElems) {
    return size_t(1) << size_t(std::bit_width(numElems) / 2);
}

/**
 * Reads data from a buffer and sends it into the graph.
 *
 * Does not take ownership of the data! Whatever data you pass in must only be destroyed when the graph has terminated!
 */
template<typename T, typename OutT = T>
struct GtMemStreamSource final: GtKernelBase {
    GtMemStreamSource(GtContext *context, std::span<const T> data, size_t fifoSize = 0):
        GtKernelBase(context), m_data(data), m_outputStream(addIoStream<OutT>(fifoSize > 0 ? fifoSize : idealFifoSizeFor(data.size()))) {}

    GtKernelIoStream<OutT> * output() {
        return m_outputStream;
    }

private:
    std::span<const T> m_data{};
    GtKernelIoStream<OutT> *m_outputStream;

    GtKernelCoro kernelMain() override {
        for (const auto& elem: m_data) {
            co_await m_outputStream->write(elem);
        }
    }
};


/**
 * Reads data from the graph and writes it into a vector.
 *
 * You can std::move() the data() out of the GtMemStreamSink if you want to save a copy.
 */
template<typename T>
struct GtMemStreamSink: GtKernelBase {
    explicit GtMemStreamSink(GtContext *context, size_t fifoSize = 1):
        GtKernelBase(context), m_inputStream(addIoStream<T>(fifoSize)) {}

    GtKernelIoStream<T> * input() {
        return m_inputStream;
    }

    std::vector<T>& data() {
        return m_data;
    }

    const std::vector<T>& data() const {
        return m_data;
    }

private:
    std::vector<T> m_data{};
    GtKernelIoStream<T> *m_inputStream;

    GtKernelCoro kernelMain() override {
        while (true) {
            m_data.push_back(co_await m_inputStream->read());
        }
    }
};


/**
 * Simulates a HLS kernel running on the PL (FPGA) that takes a stream of addresses (array indices) from
 * the graph, reads the corresponding entries from an array, and streams the data back into the graph.
 *
 * This is necessary because the AIE array does not have random memory access to the DRAM at all.
 */
template<typename T, typename Addr = uint32_t>
struct GtFpgaDmaMemReader: GtKernelBase {
    static_assert(std::is_unsigned_v<Addr>);

    GtFpgaDmaMemReader(GtContext *context, std::span<const T> mem, size_t fifoSize = 1):
        GtKernelBase(context), m_mem(mem), m_addrStream(addIoStream<Addr>(fifoSize)), m_dataStream(addIoStream<T>(fifoSize)) {}

    GtKernelIoStream<Addr> * input() {
        return m_addrStream;
    }

    GtKernelIoStream<T> * output() {
        return m_dataStream;
    }

private:
    std::span<const T> m_mem;
    GtKernelIoStream<Addr> *m_addrStream;
    GtKernelIoStream<T> *m_dataStream;
    
    GtKernelCoro kernelMain() override {
        while (true) {
            const Addr a = co_await m_addrStream->read();
            BOOST_ASSERT(a < m_mem.size());
            co_await m_dataStream->write(m_mem[a]);
        }
    }
};


/**
 * Input type of the GtPacketStreamSplitter. Data tagged with destination.
 */
template<typename T>
struct GtSplitterPacket {
    size_t m_destination;
    T m_data;
};


/**
 * Stream splitter: Takes an input stream of tagged data items, and routes them to one of
 * a pre-determined number of outputs. The output to route to is indicated by the destination
 * tag of the data.
 */
template<typename T>
struct GtPacketStreamSplitter: GtKernelBase {
    GtPacketStreamSplitter(GtContext *ctx, size_t numOutputStreams):
        GtKernelBase(ctx), m_inputStream(addIoStream<GtSplitterPacket<T>>())
    {
        m_outputStreams.resize(numOutputStreams);
        for (auto& os: m_outputStreams) {
            os = addIoStream<GtSplitterPacket<T>>();
        }
    }

    auto * input() {
        return m_inputStream;
    }

    auto * output(size_t idx) {
        return m_outputStreams.at(idx);
    }

private:
    GtKernelIoStream<GtSplitterPacket<T>> *m_inputStream;
    std::vector<GtKernelIoStream<GtSplitterPacket<T>> *> m_outputStreams{};

    GtKernelCoro kernelMain() override {
        while (true) {
            auto packet = co_await m_inputStream->read();
            const auto dest = packet.m_destination;
            BOOST_ASSERT(dest < m_outputStreams.size());
            co_await m_outputStreams.at(dest)->write(packet);
        }
    }
};


/**
 * Stream merger: Takes a number of input streams and merges their data items onto a single
 * output stream. The order of items from a single input stream is preserved; however, items
 * from different input streams will get interleaved in an unspecified way.
 */
template<typename T>
struct GtPacketStreamMerger: GtKernelBase {
    GtPacketStreamMerger(GtContext *ctx, size_t numInputStreams):
        GtKernelBase(ctx), m_outputStream(addIoStream<T>())
    {
        m_inputStreams.resize(numInputStreams);
        for (auto& is: m_inputStreams) {
            is = ctx->addKernel<MergerKernel>(this).m_inputStream;
        }
    }

    auto * input(size_t idx) {
        return m_inputStreams.at(idx);
    }

    auto * output() {
        return m_outputStream;
    }

private:
    GtKernelIoStream<T> *m_outputStream;
    std::vector<GtKernelIoStream<T> *> m_inputStreams{};

    struct MergerKernel: GtKernelBase {
        MergerKernel(GtContext *ctx, GtPacketStreamMerger *parent): GtKernelBase(ctx), m_inputStream(addIoStream<T>()), m_parent(parent) {}

        GtKernelIoStream<T> *m_inputStream;
        GtPacketStreamMerger *m_parent;

        GtKernelCoro kernelMain() override {
            while (true) {
                co_await m_parent->m_outputStream->write(co_await m_inputStream->read());
            }
        }

        bool ignoreThisKernelInStuckBufferChecks() const override { return true; }
    };

    GtKernelCoro kernelMain() override {
        // All work is done by the sub-kernels
        co_return;
    }

    uintptr_t beginAdditionalIoStreamList() const override { return 0; }

    const GtKernelIoStreamBase * derefAdditionalIoStreamIterator(uintptr_t iter) const override {
        return iter < m_inputStreams.size() ? m_inputStreams.at(iter) : nullptr;
    }

    uintptr_t advanceAdditionalIoStreamIterator(uintptr_t iter) const override { return iter + 1; }

    bool suppressStuckBufferBacktraceForThisKernel() const override { return true; }
};


/**
 * Equivalent to the input window overlap option of the AI engines.
 * Maintains an array of data items (the window) with random access.
 *
 * Every time the window is advance()d, the window is filled with new data,
 * except for the overlap, which is just shifted down in the window and kept.
 *
 * It is valid to use this class with an overlap of 0.
 * This will cause it to replace the entire window contents on every advance().
 */
template<typename T>
struct GtOverlapInputWindow {
    GtOverlapInputWindow(GtKernelIoStream<T> *stream, size_t windowSize, size_t windowOverlap): m_stream(stream), m_windowSize(windowSize), m_overlap(windowOverlap) {}

    // Access the data stored in the window.
    const T& at(size_t index) const { return m_storage.at(index); }

    // Read new data from the input stream into the window.
    // You must call this once before you want to access the window for the first time.
    // Don't forget to co_await the return value!!
    GtSubroutineCoro<void> advance() {
        if (m_overlap == 0)
            m_storage.clear();

        while (m_storage.size() > m_overlap)
            m_storage.pop_front();

        while (m_storage.size() < m_windowSize)
            m_storage.push_back(co_await m_stream->read());

        co_return;
    }

private:
    std::deque<T> m_storage{};
    GtKernelIoStream<T> *m_stream;
    size_t m_windowSize;
    size_t m_overlap;
};

}
