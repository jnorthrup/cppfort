#include "stream_parser.h"
#include <algorithm>
#include <iterator>

namespace cppfort::stage0 {

// StreamParser implementation
template<typename T>
class StreamParser {
private:
    std::deque<uint8_t> buffer;
    size_t max_buffer_size;
    StreamState<T> state;

public:
    StreamParser(size_t max_size)
        : max_buffer_size(max_size), state() {
        buffer.reserve(1024);
    }

    StreamFeedResult feed(const std::vector<uint8_t>& data) {
        // Check buffer size limit
        if (buffer.size() + data.size() > max_buffer_size) {
            state = StreamState<T>(std::string("Buffer size limit exceeded"));
            return StreamFeedResult::error("Buffer size limit exceeded");
        }

        // Add data to buffer
        std::copy(data.begin(), data.end(), std::back_inserter(buffer));

        if (state.type == StreamState<T>::Type::Complete) {
            return StreamFeedResult::already_complete();
        } else if (state.type == StreamState<T>::Type::Error) {
            return StreamFeedResult::error(*state.error);
        }

        return StreamFeedResult::ok();
    }

    template<typename ParserType>
    StreamParseResult<T> try_parse(const ParserType& parser) {
        if (state.type == StreamState<T>::Type::Complete) {
            return StreamParseResult<T>::already_complete();
        } else if (state.type == StreamState<T>::Type::Error) {
            return StreamParseResult<T>::error(*state.error);
        }

        // Convert buffer to vector for parsing
        std::vector<uint8_t> data(buffer.begin(), buffer.end());

        auto result = parser.parse(data);
        switch (result.type) {
            case ParseResultType::Complete: {
                // Remove consumed bytes from buffer
                for (size_t i = 0; i < result.consumed; ++i) {
                    buffer.pop_front();
                }
                state = StreamState<T>(*result.result);
                return StreamParseResult<T>::complete(result.consumed);
            }
            case ParseResultType::Incomplete: {
                return StreamParseResult<T>::need_more_data(buffer.size());
            }
            case ParseResultType::Error: {
                // Remove consumed bytes even on error
                for (size_t i = 0; i < result.consumed; ++i) {
                    buffer.pop_front();
                }
                state = StreamState<T>(*result.error);
                return StreamParseResult<T>::error(*result.error);
            }
        }

        // Should not reach here
        return StreamParseResult<T>::error("Unexpected parse result");
    }

    std::optional<T> take_result() {
        if (state.type == StreamState<T>::Type::Complete) {
            auto result = std::move(*state.result);
            state = StreamState<T>();
            return result;
        }
        return std::nullopt;
    }

    bool is_complete() const {
        return state.type == StreamState<T>::Type::Complete;
    }

    bool is_error() const {
        return state.type == StreamState<T>::Type::Error;
    }

    size_t buffer_size() const {
        return buffer.size();
    }

    void reset() {
        buffer.clear();
        state = StreamState<T>();
    }

    std::vector<uint8_t> peek_buffer() const {
        return std::vector<uint8_t>(buffer.begin(), buffer.end());
    }
};

// MultiStreamParser implementation
template<typename T>
class MultiStreamParser {
private:
    std::deque<uint8_t> buffer;
    size_t max_buffer_size;
    size_t attempts;
    size_t max_attempts;

public:
    MultiStreamParser(size_t max_size, size_t max_att)
        : max_buffer_size(max_size), attempts(0), max_attempts(max_att) {}

    template<typename ParserType>
    MultiParseResult<T> feed_and_try(const std::vector<uint8_t>& data,
                                    const std::vector<std::reference_wrapper<const ParserType>>& parsers) {
        // Add data to buffer
        if (buffer.size() + data.size() > max_buffer_size) {
            return MultiParseResult<T>::buffer_full();
        }

        std::copy(data.begin(), data.end(), std::back_inserter(buffer));
        attempts++;

        if (attempts > max_attempts) {
            return MultiParseResult<T>::too_many_attempts();
        }

        // Convert buffer to vector
        std::vector<uint8_t> buffer_data(buffer.begin(), buffer.end());

        // Try each parser
        for (size_t index = 0; index < parsers.size(); ++index) {
            const auto& parser = parsers[index].get();
            auto result = parser.parse(buffer_data);

            switch (result.type) {
                case ParseResultType::Complete: {
                    // Remove consumed bytes
                    for (size_t i = 0; i < result.consumed; ++i) {
                        buffer.pop_front();
                    }
                    return MultiParseResult<T>::success(*result.result, index,
                                                       result.consumed, buffer.size());
                }
                case ParseResultType::Incomplete:
                    // This parser needs more data, continue to next
                    continue;
                case ParseResultType::Error:
                    // This parser failed, continue to next
                    continue;
            }
        }

        // No parser succeeded
        return MultiParseResult<T>::need_more_data(buffer.size(), attempts);
    }

    void reset() {
        buffer.clear();
        attempts = 0;
    }
};

// ParseContinuation implementation
template<typename T>
class ParseContinuation {
private:
    std::function<ContinuationResult<T>(const std::vector<uint8_t>&)> parser_state;

public:
    template<typename Func>
    ParseContinuation(Func&& state_fn)
        : parser_state(std::forward<Func>(state_fn)) {}

    ContinuationResult<T> continue_with(const std::vector<uint8_t>& data) {
        return parser_state(data);
    }
};

// Explicit template instantiations for common types
template class StreamParser<std::string>;
template class StreamParser<std::vector<uint8_t>>;
template class MultiStreamParser<std::string>;
template class MultiStreamParser<std::vector<uint8_t>>;
template class ParseContinuation<std::string>;
template class ParseContinuation<std::vector<uint8_t>>;

} // namespace cppfort::stage0