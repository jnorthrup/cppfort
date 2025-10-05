#pragma once

#include "rbcursive_combinators.h"

#include <cstdint>
#include <deque>
#include <functional>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace cppfort::ir {

using ByteSpan = std::span<const std::uint8_t>;

// --------------------------------------------------------------------------------------
// Result types mirroring Litebike continuation.rs
// --------------------------------------------------------------------------------------

enum class StreamFeedKind {
    Ok,
    DataAdded,
    AlreadyComplete,
    Error,
};

struct StreamFeedResult {
    StreamFeedKind kind;
    std::size_t bytes = 0;
    std::optional<ParseError> error{};

    static StreamFeedResult ok(std::size_t appended) {
        return {StreamFeedKind::Ok, appended, std::nullopt};
    }

    static StreamFeedResult dataAdded(std::size_t appended) {
        return {StreamFeedKind::DataAdded, appended, std::nullopt};
    }

    static StreamFeedResult alreadyComplete() {
        return {StreamFeedKind::AlreadyComplete, 0, std::nullopt};
    }

    static StreamFeedResult error(ParseError err) {
        return {StreamFeedKind::Error, 0, err};
    }
};

enum class StreamParseKind {
    Complete,
    NeedMoreData,
    AlreadyComplete,
    Error,
};

struct StreamParseResult {
    StreamParseKind kind;
    std::size_t metric = 0;  // Consumed bytes or buffered length
    std::optional<ParseError> error{};

    static StreamParseResult complete(std::size_t consumed) {
        return {StreamParseKind::Complete, consumed, std::nullopt};
    }

    static StreamParseResult needMore(std::size_t buffered) {
        return {StreamParseKind::NeedMoreData, buffered, std::nullopt};
    }

    static StreamParseResult alreadyComplete() {
        return {StreamParseKind::AlreadyComplete, 0, std::nullopt};
    }

    static StreamParseResult error(ParseError err, std::size_t consumed = 0) {
        return {StreamParseKind::Error, consumed, err};
    }
};

// --------------------------------------------------------------------------------------
// StreamParser - continuation-aware streaming parser
// --------------------------------------------------------------------------------------

template <typename T>
class StreamParser {
public:
    explicit StreamParser(std::size_t maxBufferSize)
        : maxBufferSize_(maxBufferSize), state_(std::monostate{}) {}

    StreamFeedResult feed(ByteSpan data) {
        if (buffer_.size() + data.size() > maxBufferSize_) {
            state_ = ParseError::InvalidLength;
            return StreamFeedResult::error(ParseError::InvalidLength);
        }

        for (auto byte : data) {
            buffer_.push_back(byte);
        }

        if (std::holds_alternative<T>(state_)) {
            return StreamFeedResult::alreadyComplete();
        }
        if (std::holds_alternative<ParseError>(state_)) {
            return StreamFeedResult::error(std::get<ParseError>(state_));
        }

        return StreamFeedResult::ok(data.size());
    }

    template <typename Parser>
    StreamParseResult tryParse(const Parser& parser) {
        using Value = typename Parser::value_type;
        static_assert(std::is_same_v<Value, T>, "Parser value type must match StreamParser type");

        if (std::holds_alternative<T>(state_)) {
            return StreamParseResult::alreadyComplete();
        }
        if (std::holds_alternative<ParseError>(state_)) {
            return StreamParseResult::error(std::get<ParseError>(state_));
        }

        std::vector<std::uint8_t> data(buffer_.begin(), buffer_.end());
        ByteSpan span{data.data(), data.size()};
        auto result = parser.parse(span);

        if (auto complete = result.intoComplete()) {
            auto [value, consumed] = *complete;
            discard(consumed);
            state_ = std::move(value);
            return StreamParseResult::complete(consumed);
        }

        if (result.kind() == ParseResult<Value, ParseError>::Kind::Incomplete) {
            return StreamParseResult::needMore(buffer_.size());
        }

        if (auto err = result.intoError()) {
            auto [error, consumed] = *err;
            discard(consumed);
            state_ = error;
            return StreamParseResult::error(error, consumed);
        }

        return StreamParseResult::error(ParseError::InvalidInput);
    }

    std::optional<T> takeResult() {
        if (auto ptr = std::get_if<T>(&state_)) {
            std::optional<T> out{std::move(*ptr)};
            state_ = std::monostate{};
            return out;
        }
        return std::nullopt;
    }

    bool isComplete() const {
        return std::holds_alternative<T>(state_);
    }

    bool isError() const {
        return std::holds_alternative<ParseError>(state_);
    }

    std::size_t bufferSize() const {
        return buffer_.size();
    }

    void reset() {
        buffer_.clear();
        state_ = std::monostate{};
    }

    std::vector<std::uint8_t> peekBuffer() const {
        return std::vector<std::uint8_t>(buffer_.begin(), buffer_.end());
    }

private:
    void discard(std::size_t count) {
        for (std::size_t i = 0; i < count && !buffer_.empty(); ++i) {
            buffer_.pop_front();
        }
    }

    std::deque<std::uint8_t> buffer_;
    std::size_t maxBufferSize_;
    std::variant<std::monostate, T, ParseError> state_;
};

// --------------------------------------------------------------------------------------
// MultiStreamParser - tries multiple parsers in sequence
// --------------------------------------------------------------------------------------

enum class MultiParseKind {
    Success,
    NeedMoreData,
    BufferFull,
    TooManyAttempts,
};

template <typename T>
struct MultiParseResult {
    MultiParseKind kind;
    std::optional<T> result{};
    std::size_t parserIndex = 0;
    std::size_t consumed = 0;
    std::size_t remaining = 0;
    std::size_t bufferSize = 0;
    std::size_t attempts = 0;

    static MultiParseResult success(T value, std::size_t index,
                                    std::size_t consumedBytes,
                                    std::size_t remainingBytes) {
        MultiParseResult out;
        out.kind = MultiParseKind::Success;
        out.result = std::move(value);
        out.parserIndex = index;
        out.consumed = consumedBytes;
        out.remaining = remainingBytes;
        return out;
    }

    static MultiParseResult needMore(std::size_t bufferLen, std::size_t attemptCount) {
        MultiParseResult out;
        out.kind = MultiParseKind::NeedMoreData;
        out.bufferSize = bufferLen;
        out.attempts = attemptCount;
        return out;
    }

    static MultiParseResult bufferFull() {
        MultiParseResult out;
        out.kind = MultiParseKind::BufferFull;
        return out;
    }

    static MultiParseResult tooManyAttempts() {
        MultiParseResult out;
        out.kind = MultiParseKind::TooManyAttempts;
        return out;
    }
};

template <typename T>
class MultiStreamParser {
public:
    MultiStreamParser(std::size_t maxBufferSize, std::size_t maxAttempts)
        : maxBufferSize_(maxBufferSize), attempts_(0), maxAttempts_(maxAttempts) {}

    template <typename Parser>
    MultiParseResult<T> feedAndTry(ByteSpan data,
                                   std::span<const std::reference_wrapper<const Parser>> parsers) {
        using Value = typename Parser::value_type;
        static_assert(std::is_same_v<Value, T>, "Parser value type must match MultiStreamParser type");

        if (buffer_.size() + data.size() > maxBufferSize_) {
            return MultiParseResult<T>::bufferFull();
        }

        for (auto byte : data) {
            buffer_.push_back(byte);
        }

        ++attempts_;
        if (attempts_ > maxAttempts_) {
            return MultiParseResult<T>::tooManyAttempts();
        }

        std::vector<std::uint8_t> copy(buffer_.begin(), buffer_.end());
        ByteSpan span{copy.data(), copy.size()};

        for (std::size_t index = 0; index < parsers.size(); ++index) {
            const auto& parser = parsers[index].get();
            auto result = parser.parse(span);

            if (auto complete = result.intoComplete()) {
                auto [value, consumed] = *complete;
                discard(consumed);
                return MultiParseResult<T>::success(std::move(value), index, consumed, buffer_.size());
            }

            if (result.kind() == ParseResult<Value, ParseError>::Kind::Incomplete) {
                continue;
            }

            if (auto err = result.intoError()) {
                auto [_, consumed] = *err;
                discard(consumed);
                continue;
            }
        }

        return MultiParseResult<T>::needMore(buffer_.size(), attempts_);
    }

    void reset() {
        buffer_.clear();
        attempts_ = 0;
    }

private:
    void discard(std::size_t count) {
        for (std::size_t i = 0; i < count && !buffer_.empty(); ++i) {
            buffer_.pop_front();
        }
    }

    std::deque<std::uint8_t> buffer_;
    std::size_t maxBufferSize_;
    std::size_t attempts_;
    std::size_t maxAttempts_;
};

// --------------------------------------------------------------------------------------
// ParseContinuation - function-based continuation state machine
// --------------------------------------------------------------------------------------

template <typename T>
struct ContinuationResult {
    enum class Kind { Complete, Continue, Error };

    Kind kind;
    std::optional<T> result{};
    std::optional<ParseError> error{};
    std::size_t consumed = 0;

    static ContinuationResult complete(T value, std::size_t consumedBytes) {
        return {Kind::Complete, std::optional<T>(std::move(value)), std::nullopt, consumedBytes};
    }

    static ContinuationResult cont(std::size_t consumedBytes) {
        return {Kind::Continue, std::nullopt, std::nullopt, consumedBytes};
    }

    static ContinuationResult error(ParseError err, std::size_t consumedBytes) {
        return {Kind::Error, std::nullopt, err, consumedBytes};
    }
};

template <typename T>
class ParseContinuation {
public:
    using StateFn = std::function<ContinuationResult<T>(ByteSpan)>;

    ParseContinuation() = default;

    explicit ParseContinuation(StateFn fn)
        : stateFn_(std::move(fn)) {}

    template <typename Fn>
    static ParseContinuation create(Fn&& fn) {
        return ParseContinuation(StateFn(std::forward<Fn>(fn)));
    }

    ContinuationResult<T> continueWith(ByteSpan data) {
        if (!stateFn_) {
            return ContinuationResult<T>::error(ParseError::InvalidInput, 0);
        }
        return stateFn_(data);
    }

private:
    StateFn stateFn_;
};

}  // namespace cppfort::ir 