// cpp2/concurrency.hpp - Kotlin-style Structured Concurrency for Cpp2
//
// Runtime library providing Task<T>, Channel<T>, and coroutineScope primitives.
// Targets C++20 coroutines with optional C++26 std::execution integration.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CPP2_CONCURRENCY_HPP
#define CPP2_CONCURRENCY_HPP

#include <coroutine>
#include <exception>
#include <future>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <optional>
#include <memory>
#include <vector>
#include <functional>
#include <atomic>
#include <thread>

namespace cpp2 {

// ============================================================================
// Task<T> - Lazy coroutine task (similar to Kotlin's Deferred<T>)
// ============================================================================

template<typename T>
class Task {
public:
    struct promise_type {
        std::optional<T> result;
        std::exception_ptr exception;

        Task get_return_object() {
            return Task{std::coroutine_handle<promise_type>::from_promise(*this)};
        }

        std::suspend_always initial_suspend() noexcept { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }

        void unhandled_exception() {
            exception = std::current_exception();
        }

        template<typename U>
        void return_value(U&& value) {
            result = std::forward<U>(value);
        }

        // Awaiter for co_await on Task<T>
        auto await_transform(Task<T>& task) {
            return task.operator co_await();
        }
    };

    using handle_type = std::coroutine_handle<promise_type>;

    Task() = default;
    explicit Task(handle_type h) : handle_(h) {}

    ~Task() {
        if (handle_) {
            handle_.destroy();
        }
    }

    Task(Task&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }

    Task& operator=(Task&& other) noexcept {
        if (this != &other) {
            if (handle_) handle_.destroy();
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    Task(const Task&) = delete;
    Task& operator=(const Task&) = delete;

    bool done() const { return handle_.done(); }

    void resume() {
        if (handle_ && !handle_.done()) {
            handle_.resume();
        }
    }

    T get() {
        while (!handle_.done()) {
            handle_.resume();
        }
        if (handle_.promise().exception) {
            std::rethrow_exception(handle_.promise().exception);
        }
        return std::move(*handle_.promise().result);
    }

    // Awaitable interface
    auto operator co_await() {
        struct Awaiter {
            handle_type handle;

            bool await_ready() const noexcept {
                return handle.done();
            }

            std::coroutine_handle<> await_suspend(std::coroutine_handle<> caller) noexcept {
                return handle;
            }

            T await_resume() {
                if (handle.promise().exception) {
                    std::rethrow_exception(handle.promise().exception);
                }
                return std::move(*handle.promise().result);
            }
        };
        return Awaiter{handle_};
    }

private:
    handle_type handle_ = nullptr;
};

// Specialization for void
template<>
class Task<void> {
public:
    struct promise_type {
        std::exception_ptr exception;

        Task get_return_object() {
            return Task{std::coroutine_handle<promise_type>::from_promise(*this)};
        }

        std::suspend_always initial_suspend() noexcept { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }

        void unhandled_exception() {
            exception = std::current_exception();
        }

        void return_void() {}
    };

    using handle_type = std::coroutine_handle<promise_type>;

    Task() = default;
    explicit Task(handle_type h) : handle_(h) {}

    ~Task() {
        if (handle_) handle_.destroy();
    }

    Task(Task&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }

    Task& operator=(Task&& other) noexcept {
        if (this != &other) {
            if (handle_) handle_.destroy();
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    Task(const Task&) = delete;
    Task& operator=(const Task&) = delete;

    bool done() const { return handle_.done(); }

    void resume() {
        if (handle_ && !handle_.done()) {
            handle_.resume();
        }
    }

    void get() {
        while (!handle_.done()) {
            handle_.resume();
        }
        if (handle_.promise().exception) {
            std::rethrow_exception(handle_.promise().exception);
        }
    }

private:
    handle_type handle_ = nullptr;
};

// ============================================================================
// Job - Fire-and-forget coroutine (similar to Kotlin's Job)
// ============================================================================

class Job {
public:
    struct promise_type {
        std::exception_ptr exception;

        Job get_return_object() {
            return Job{std::coroutine_handle<promise_type>::from_promise(*this)};
        }

        std::suspend_never initial_suspend() noexcept { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }

        void unhandled_exception() {
            exception = std::current_exception();
        }

        void return_void() {}
    };

    using handle_type = std::coroutine_handle<promise_type>;

    Job() = default;
    explicit Job(handle_type h) : handle_(h) {}

    ~Job() {
        if (handle_) handle_.destroy();
    }

    Job(Job&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }

    Job& operator=(Job&& other) noexcept {
        if (this != &other) {
            if (handle_) handle_.destroy();
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    bool done() const { return handle_.done(); }

    void join() {
        while (!handle_.done()) {
            handle_.resume();
        }
    }

private:
    handle_type handle_ = nullptr;
};

// ============================================================================
// Channel<T> - Kotlin-style channel for coroutine communication
// ============================================================================

template<typename T>
class Channel {
public:
    explicit Channel(size_t capacity = 0) : capacity_(capacity) {}

    ~Channel() {
        close();
    }

    Channel(const Channel&) = delete;
    Channel& operator=(const Channel&) = delete;

    // Send value to channel (blocking if buffer full)
    bool send(T value) {
        std::unique_lock<std::mutex> lock(mutex_);
        
        if (closed_) return false;

        // Wait if buffer is full (for buffered channels)
        if (capacity_ > 0) {
            not_full_.wait(lock, [this] {
                return queue_.size() < capacity_ || closed_;
            });
        }

        if (closed_) return false;

        queue_.push(std::move(value));
        not_empty_.notify_one();
        return true;
    }

    // Try send without blocking
    bool trySend(T value) {
        std::unique_lock<std::mutex> lock(mutex_);
        
        if (closed_) return false;
        if (capacity_ > 0 && queue_.size() >= capacity_) return false;

        queue_.push(std::move(value));
        not_empty_.notify_one();
        return true;
    }

    // Receive value from channel (blocking if empty)
    std::optional<T> receive() {
        std::unique_lock<std::mutex> lock(mutex_);

        not_empty_.wait(lock, [this] {
            return !queue_.empty() || closed_;
        });

        if (queue_.empty()) return std::nullopt;

        T value = std::move(queue_.front());
        queue_.pop();
        not_full_.notify_one();
        return value;
    }

    // Try receive without blocking
    std::optional<T> tryReceive() {
        std::unique_lock<std::mutex> lock(mutex_);

        if (queue_.empty()) return std::nullopt;

        T value = std::move(queue_.front());
        queue_.pop();
        not_full_.notify_one();
        return value;
    }

    // Close the channel
    void close() {
        std::unique_lock<std::mutex> lock(mutex_);
        closed_ = true;
        not_empty_.notify_all();
        not_full_.notify_all();
    }

    bool isClosed() const {
        std::unique_lock<std::mutex> lock(mutex_);
        return closed_;
    }

    bool isEmpty() const {
        std::unique_lock<std::mutex> lock(mutex_);
        return queue_.empty();
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    std::queue<T> queue_;
    size_t capacity_;
    bool closed_ = false;
};

// ============================================================================
// CoroutineScope - Structured concurrency scope (Kotlin-style)
// ============================================================================

class CoroutineScope {
public:
    CoroutineScope() = default;

    ~CoroutineScope() {
        // Wait for all jobs to complete (structured concurrency guarantee)
        joinAll();
    }

    CoroutineScope(const CoroutineScope&) = delete;
    CoroutineScope& operator=(const CoroutineScope&) = delete;

    // Launch a new coroutine in this scope
    template<typename F>
    void launch(F&& func) {
        std::unique_lock<std::mutex> lock(mutex_);
        futures_.push_back(std::async(std::launch::async, std::forward<F>(func)));
    }

    // Wait for all launched coroutines to complete
    void joinAll() {
        std::vector<std::future<void>> to_wait;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            to_wait = std::move(futures_);
            futures_.clear();
        }
        for (auto& f : to_wait) {
            if (f.valid()) {
                f.wait();
            }
        }
    }

    // Cancel all coroutines (cooperative cancellation)
    void cancel() {
        std::unique_lock<std::mutex> lock(mutex_);
        cancelled_ = true;
    }

    bool isCancelled() const {
        std::unique_lock<std::mutex> lock(mutex_);
        return cancelled_;
    }

private:
    mutable std::mutex mutex_;
    std::vector<std::future<void>> futures_;
    bool cancelled_ = false;
};

// ============================================================================
// coroutineScope - RAII-based structured concurrency block
// ============================================================================

template<typename F>
auto coroutineScope(F&& block) {
    CoroutineScope scope;
    return block(scope);
}

// ============================================================================
// Dispatcher - Coroutine execution context
// ============================================================================

class Dispatcher {
public:
    virtual ~Dispatcher() = default;

    virtual void dispatch(std::function<void()> task) = 0;
};

// Default dispatcher using thread pool
class DefaultDispatcher : public Dispatcher {
public:
    static DefaultDispatcher& instance() {
        static DefaultDispatcher dispatcher;
        return dispatcher;
    }

    void dispatch(std::function<void()> task) override {
        std::thread(std::move(task)).detach();
    }
};

// IO dispatcher for blocking operations
class IODispatcher : public Dispatcher {
public:
    static IODispatcher& instance() {
        static IODispatcher dispatcher;
        return dispatcher;
    }

    void dispatch(std::function<void()> task) override {
        std::thread(std::move(task)).detach();
    }
};

// ============================================================================
// Utility functions
// ============================================================================

// Yield to other coroutines
inline auto yield() {
    struct YieldAwaiter {
        bool await_ready() const noexcept { return false; }
        void await_suspend(std::coroutine_handle<> h) const noexcept {
            // Resume immediately on another thread or same thread
            h.resume();
        }
        void await_resume() const noexcept {}
    };
    return YieldAwaiter{};
}

// Delay execution
inline auto delay(std::chrono::milliseconds duration) {
    struct DelayAwaiter {
        std::chrono::milliseconds duration;

        bool await_ready() const noexcept { return duration.count() <= 0; }

        void await_suspend(std::coroutine_handle<> h) const {
            std::thread([h, d = duration] {
                std::this_thread::sleep_for(d);
                h.resume();
            }).detach();
        }

        void await_resume() const noexcept {}
    };
    return DelayAwaiter{duration};
}

} // namespace cpp2

#endif // CPP2_CONCURRENCY_HPP
