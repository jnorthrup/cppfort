#pragma once

#include <deque>
#include <vector>
#include <memory>
#include <functional>
#include <variant>
#include <optional>
#include "orbit_ring.h"

namespace cppfort::stage0 {

// Forward declarations
template<typename T>
class StreamParser;

template<typename T>
class MultiStreamParser;

template<typename T>
class ParseContinuation;

// Parser result types
enum class ParseResultType {
    Complete,
    Incomplete,
    Error
};

template<typename T>
struct ParseResult {
    enum class Type { Complete, Incomplete, Error };
    Type type;
    std::optional<T> result;
    std::optional<std::string> error_msg;
    size_t consumed;

    static ParseResult complete(T val, size_t consumed_bytes) {
        return {Type::Complete, std::move(val), std::nullopt, consumed_bytes};
    }

    static ParseResult incomplete(size_t consumed_bytes) {
        return {Type::Incomplete, std::nullopt, std::nullopt, consumed_bytes};
    }

    static ParseResult error(std::string msg, size_t consumed_bytes) {
        return {Type::Error, std::nullopt, std::move(msg), consumed_bytes};
    }
};

// Parser interface
template<typename T>
class Parser {
public:
    virtual ~Parser() = default;
    virtual ParseResult<T> parse(const std::vector<uint8_t>& data) const = 0;
};

// Stream parser states
template<typename T>
struct StreamState {
    enum class Type { Ready, Complete, Error };
    Type type;
    std::optional<T> result;
    std::optional<std::string> error;

    StreamState() : type(Type::Ready) {}
    StreamState(T val) : type(Type::Complete), result(std::move(val)) {}
    StreamState(std::string err) : type(Type::Error), error(std::move(err)) {}
};

// Stream parser result types
template<typename T>
struct StreamParseResult {
    enum class Type { Complete, NeedMoreData, AlreadyComplete, Error };
    Type type;
    size_t consumed;
    std::optional<std::string> error_msg;

    static StreamParseResult complete(size_t consumed_bytes) {
        return {Type::Complete, consumed_bytes, std::nullopt};
    }

    static StreamParseResult need_more_data(size_t buffer_size) {
        return {Type::NeedMoreData, buffer_size, std::nullopt};
    }

    static StreamParseResult already_complete() {
        return {Type::AlreadyComplete, 0, std::nullopt};
    }

    static StreamParseResult error(std::string err) {
        return {Type::Error, 0, std::move(err)};
    }
};

// Stream feed result types
struct StreamFeedResult {
    enum class Type { Ok, DataAdded, AlreadyComplete, Error };
    Type type;
    size_t added;
    std::optional<std::string> error_msg;

    static StreamFeedResult ok() {
        return {Type::Ok, 0, std::nullopt};
    }

    static StreamFeedResult data_added(size_t bytes) {
        return {Type::DataAdded, bytes, std::nullopt};
    }

    static StreamFeedResult already_complete() {
        return {Type::AlreadyComplete, 0, std::nullopt};
    }

    static StreamFeedResult error(std::string err) {
        return {Type::Error, 0, std::move(err)};
    }
};

// Continuation result types
template<typename T>
struct ContinuationResult {
    enum class Type { Complete, Continue, Error };
    Type type;
    std::optional<T> result;
    std::optional<std::string> error_msg;

    static ContinuationResult complete(T val) {
        return {Type::Complete, std::move(val), std::nullopt};
    }

    static ContinuationResult continue_parsing() {
        return {Type::Continue, std::nullopt, std::nullopt};
    }

    static ContinuationResult error(std::string err) {
        return {Type::Error, std::nullopt, std::move(err)};
    }
};

// Multi-parser result types
template<typename T>
struct MultiParseResult {
    enum class Type { Success, NeedMoreData, BufferFull, TooManyAttempts };
    Type type;
    std::optional<T> result;
    size_t parser_index;
    size_t consumed;
    size_t remaining;
    size_t buffer_size;
    size_t attempts;

    static MultiParseResult success(T val, size_t index, size_t consumed_bytes, size_t remaining_bytes) {
        return {Type::Success, std::move(val), index, consumed_bytes, remaining_bytes, 0, 0};
    }

    static MultiParseResult need_more_data(size_t buffer_sz, size_t attempt_count) {
        return {Type::NeedMoreData, std::nullopt, 0, 0, 0, buffer_sz, attempt_count};
    }

    static MultiParseResult buffer_full() {
        return {Type::BufferFull, std::nullopt, 0, 0, 0, 0, 0};
    }

    static MultiParseResult too_many_attempts() {
        return {Type::TooManyAttempts, std::nullopt, 0, 0, 0, 0, 0};
    }
};

} // namespace cppfort::stage0