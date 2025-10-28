#pragma once

#include "rbcursive.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace cppfort {
namespace ir {

using ByteSpan = std::span<const std::uint8_t>;

enum class Signal {
    Accept,
    NeedMore,
    Reject,
};

enum class ParseError {
    InvalidInput,
    UnexpectedEnd,
    InvalidProtocol,
    InvalidMethod,
    InvalidHeader,
    InvalidLength,
};

template <typename T, typename E>
class ParseResult {
public:
    enum class Kind { Complete, Incomplete, Error };

    struct CompletePayload {
        T value;
        std::size_t consumed;
    };

    struct IncompletePayload {
        std::size_t consumed;
    };

    struct ErrorPayload {
        E error;
        std::size_t consumed;
    };

    static ParseResult complete(T value, std::size_t consumed) {
        return ParseResult(CompletePayload{std::move(value), consumed});
    }

    static ParseResult incomplete(std::size_t consumed) {
        return ParseResult(IncompletePayload{consumed});
    }

    static ParseResult error(E error, std::size_t consumed) {
        return ParseResult(ErrorPayload{error, consumed});
    }

    Kind kind() const {
        return static_cast<Kind>(payload_.index());
    }

    bool isComplete() const {
        return std::holds_alternative<CompletePayload>(payload_);
    }

    std::size_t consumed() const {
        if (auto* complete = std::get_if<CompletePayload>(&payload_)) {
            return complete->consumed;
        }
        if (auto* incomplete = std::get_if<IncompletePayload>(&payload_)) {
            return incomplete->consumed;
        }
        return std::get<ErrorPayload>(payload_).consumed;
    }

    std::optional<std::pair<T, std::size_t>> intoComplete() const {
        if (auto* complete = std::get_if<CompletePayload>(&payload_)) {
            return std::make_pair(complete->value, complete->consumed);
        }
        return std::nullopt;
    }

    std::optional<std::pair<E, std::size_t>> intoError() const {
        if (auto* error = std::get_if<ErrorPayload>(&payload_)) {
            return std::make_pair(error->error, error->consumed);
        }
        return std::nullopt;
    }

    template <typename F>
    auto map(F&& mapper) const
        -> ParseResult<std::invoke_result_t<F, const T&>, E> {
        using U = std::invoke_result_t<F, const T&>;
        if (auto* complete = std::get_if<CompletePayload>(&payload_)) {
            return ParseResult<U, E>::complete(mapper(complete->value), complete->consumed);
        }
        if (auto* incomplete = std::get_if<IncompletePayload>(&payload_)) {
            return ParseResult<U, E>::incomplete(incomplete->consumed);
        }
        auto& error = std::get<ErrorPayload>(payload_);
        return ParseResult<U, E>::error(error.error, error.consumed);
    }

    template <typename F>
    auto mapError(F&& mapper) const -> ParseResult<T, std::invoke_result_t<F, const E&>> {
        using E2 = std::invoke_result_t<F, const E&>;
        if (auto* complete = std::get_if<CompletePayload>(&payload_)) {
            return ParseResult<T, E2>::complete(complete->value, complete->consumed);
        }
        if (auto* incomplete = std::get_if<IncompletePayload>(&payload_)) {
            return ParseResult<T, E2>::incomplete(incomplete->consumed);
        }
        auto& error = std::get<ErrorPayload>(payload_);
        return ParseResult<T, E2>::error(mapper(error.error), error.consumed);
    }

    Signal signal() const {
        switch (kind()) {
            case Kind::Complete:
                return Signal::Accept;
            case Kind::Incomplete:
                return Signal::NeedMore;
            case Kind::Error:
                return Signal::Reject;
        }
        return Signal::Reject;
    }

private:
    explicit ParseResult(CompletePayload payload) : payload_(std::move(payload)) {}
    explicit ParseResult(IncompletePayload payload) : payload_(payload) {}
    explicit ParseResult(ErrorPayload payload) : payload_(std::move(payload)) {}

    std::variant<CompletePayload, IncompletePayload, ErrorPayload> payload_;
};

class ByteParser {
public:
    using value_type = std::uint8_t;
    using error_type = ParseError;

    explicit ByteParser(std::uint8_t target) : target_(target) {}

    ParseResult<value_type, error_type> parse(ByteSpan input) const {
        if (input.empty()) {
            return ParseResult<value_type, error_type>::incomplete(0);
        }
        if (input.front() == target_) {
            return ParseResult<value_type, error_type>::complete(input.front(), 1);
        }
        return ParseResult<value_type, error_type>::error(ParseError::InvalidInput, 0);
    }

private:
    std::uint8_t target_;
};

class TakeParser {
public:
    using value_type = ByteSpan;
    using error_type = ParseError;

    explicit TakeParser(std::size_t count) : count_(count) {}

    ParseResult<value_type, error_type> parse(ByteSpan input) const {
        if (input.size() < count_) {
            return ParseResult<value_type, error_type>::incomplete(input.size());
        }
        return ParseResult<value_type, error_type>::complete(input.first(count_), count_);
    }

private:
    std::size_t count_;
};

class TakeUntilParser {
public:
    using value_type = ByteSpan;
    using error_type = ParseError;

    TakeUntilParser(std::uint8_t delimiter, const SimdScanner& scanner)
        : delimiter_(delimiter), scanner_(scanner) {}

    ParseResult<value_type, error_type> parse(ByteSpan input) const {
        if (input.empty()) {
            return ParseResult<value_type, error_type>::incomplete(0);
        }
        const std::array<std::uint8_t, 1> targets{delimiter_};
        auto positions = scanner_.scanBytes(input, std::span<const std::uint8_t>(targets));
        if (!positions.empty()) {
            std::size_t pos = positions.front();
            return ParseResult<value_type, error_type>::complete(input.first(pos), pos);
        }
        return ParseResult<value_type, error_type>::incomplete(input.size());
    }

private:
    std::uint8_t delimiter_;
    const SimdScanner& scanner_;
};

template <typename Predicate>
class TakeWhileParser {
public:
    using value_type = ByteSpan;
    using error_type = ParseError;

    explicit TakeWhileParser(Predicate predicate) : predicate_(std::move(predicate)) {}

    ParseResult<value_type, error_type> parse(ByteSpan input) const {
        std::size_t count = 0;
        for (auto byte : input) {
            if (predicate_(byte)) {
                ++count;
            } else {
                break;
            }
        }
        return ParseResult<value_type, error_type>::complete(input.first(count), count);
    }

private:
    Predicate predicate_;
};

class TagParser {
public:
    using value_type = ByteSpan;
    using error_type = ParseError;

    explicit TagParser(std::vector<std::uint8_t> tag) : tag_(std::move(tag)) {}

    ParseResult<value_type, error_type> parse(ByteSpan input) const {
        if (input.size() < tag_.size()) {
            return ParseResult<value_type, error_type>::incomplete(input.size());
        }
        if (std::equal(tag_.begin(), tag_.end(), input.begin())) {
            return ParseResult<value_type, error_type>::complete(input.first(tag_.size()), tag_.size());
        }
        return ParseResult<value_type, error_type>::error(ParseError::InvalidInput, 0);
    }

private:
    std::vector<std::uint8_t> tag_;
};

template <typename First, typename Second>
class SequenceParser {
public:
    using value_type = std::pair<typename First::value_type, typename Second::value_type>;
    using error_type = ParseError;

    SequenceParser(First first, Second second)
        : first_(std::move(first)), second_(std::move(second)) {}

    ParseResult<value_type, error_type> parse(ByteSpan input) const {
        auto firstResult = first_.parse(input);
        if (auto firstComplete = firstResult.intoComplete()) {
            const auto& [firstValue, firstConsumed] = *firstComplete;
            auto secondResult = second_.parse(input.subspan(firstConsumed));
            if (auto secondComplete = secondResult.intoComplete()) {
                const auto& [secondValue, secondConsumed] = *secondComplete;
                return ParseResult<value_type, error_type>::complete(
                    std::make_pair(firstValue, secondValue),
                    firstConsumed + secondConsumed);
            }
            if (secondResult.kind() == ParseResult<typename Second::value_type, error_type>::Kind::Incomplete) {
                return ParseResult<value_type, error_type>::incomplete(
                    firstConsumed + secondResult.consumed());
            }
            if (auto secondError = secondResult.intoError()) {
                return ParseResult<value_type, error_type>::error(
                    secondError->first,
                    firstConsumed + secondError->second);
            }
        }
        if (firstResult.kind() == ParseResult<typename First::value_type, error_type>::Kind::Incomplete) {
            return ParseResult<value_type, error_type>::incomplete(firstResult.consumed());
        }
        if (auto firstError = firstResult.intoError()) {
            return ParseResult<value_type, error_type>::error(firstError->first, firstError->second);
        }
        return ParseResult<value_type, error_type>::error(ParseError::InvalidInput, 0);
    }

private:
    First first_;
    Second second_;
};

template <typename First, typename Second>
class AlternativeParser {
public:
    using value_type = typename First::value_type;
    using error_type = ParseError;

    AlternativeParser(First first, Second second)
        : first_(std::move(first)), second_(std::move(second)) {}

    ParseResult<value_type, error_type> parse(ByteSpan input) const {
        auto firstResult = first_.parse(input);
        switch (firstResult.kind()) {
            case ParseResult<value_type, error_type>::Kind::Complete:
            case ParseResult<value_type, error_type>::Kind::Incomplete:
                return firstResult;
            case ParseResult<value_type, error_type>::Kind::Error:
                return second_.parse(input);
        }
        return second_.parse(input);
    }

private:
    First first_;
    Second second_;
};

template <typename ParserT, typename Mapper>
class MapParser {
public:
    using intermediate_type = typename ParserT::value_type;
    using value_type = std::invoke_result_t<Mapper, const intermediate_type&>;
    using error_type = ParseError;

    MapParser(ParserT parser, Mapper mapper)
        : parser_(std::move(parser)), mapper_(std::move(mapper)) {}

    ParseResult<value_type, error_type> parse(ByteSpan input) const {
        auto result = parser_.parse(input);
        if (auto complete = result.intoComplete()) {
            const auto& [value, consumed] = *complete;
            return ParseResult<value_type, error_type>::complete(mapper_(value), consumed);
        }
        if (result.kind() == ParseResult<intermediate_type, error_type>::Kind::Incomplete) {
            return ParseResult<value_type, error_type>::incomplete(result.consumed());
        }
        if (auto err = result.intoError()) {
            return ParseResult<value_type, error_type>::error(err->first, err->second);
        }
        return ParseResult<value_type, error_type>::error(ParseError::InvalidInput, result.consumed());
    }

private:
    ParserT parser_;
    Mapper mapper_;
};

class ByteRangeWhileParser {
public:
    using value_type = ByteSpan;
    using error_type = ParseError;

    ByteRangeWhileParser(std::uint8_t start, std::uint8_t end, std::size_t min, std::optional<std::size_t> max)
        : start_(start), end_(end), min_(min), max_(max) {}

    ParseResult<value_type, error_type> parse(ByteSpan input) const {
        if (input.empty()) {
            return ParseResult<value_type, error_type>::incomplete(0);
        }
        std::size_t len = 0;
        std::size_t bound = max_.has_value() ? std::min<std::size_t>(*max_, input.size()) : input.size();
        while (len < bound) {
            auto byte = input[len];
            if (byte < start_ || byte > end_) {
                break;
            }
            ++len;
        }
        if (len < min_) {
            if (input.size() < min_) {
                return ParseResult<value_type, error_type>::incomplete(input.size());
            }
            return ParseResult<value_type, error_type>::error(ParseError::InvalidInput, len);
        }
        return ParseResult<value_type, error_type>::complete(input.first(len), len);
    }

private:
    std::uint8_t start_;
    std::uint8_t end_;
    std::size_t min_;
    std::optional<std::size_t> max_;
};

class ConfixParser {
public:
    using value_type = ByteSpan;
    using error_type = ParseError;

    ConfixParser(std::uint8_t open, std::uint8_t close, bool allowNested)
        : open_(open), close_(close), allowNested_(allowNested) {}

    ParseResult<value_type, error_type> parse(ByteSpan input) const {
        if (input.empty()) {
            return ParseResult<value_type, error_type>::incomplete(0);
        }
        if (input.front() != open_) {
            return ParseResult<value_type, error_type>::error(ParseError::InvalidInput, 0);
        }
        if (!allowNested_) {
            for (std::size_t i = 1; i < input.size(); ++i) {
                if (input[i] == close_) {
                    return ParseResult<value_type, error_type>::complete(input.first(i + 1), i + 1);
                }
            }
            return ParseResult<value_type, error_type>::incomplete(input.size());
        }
        std::size_t depth = 1;
        for (std::size_t i = 1; i < input.size(); ++i) {
            auto current = input[i];
            if (current == open_) {
                ++depth;
            } else if (current == close_) {
                if (--depth == 0) {
                    return ParseResult<value_type, error_type>::complete(input.first(i + 1), i + 1);
                }
            }
        }
        return ParseResult<value_type, error_type>::incomplete(input.size());
    }

private:
    std::uint8_t open_;
    std::uint8_t close_;
    bool allowNested_;
};

inline ByteParser byte(std::uint8_t target) { return ByteParser(target); }
inline ByteParser chlit(std::uint8_t target) { return byte(target); }
inline TakeParser take(std::size_t count) { return TakeParser(count); }
inline TakeUntilParser takeUntil(std::uint8_t delimiter, const SimdScanner& scanner) {
    return TakeUntilParser(delimiter, scanner);
}

template <typename Predicate>
inline TakeWhileParser<Predicate> takeWhile(Predicate predicate) {
    return TakeWhileParser<Predicate>(std::move(predicate));
}

inline TagParser tag(std::span<const std::uint8_t> pattern) {
    return TagParser(std::vector<std::uint8_t>(pattern.begin(), pattern.end()));
}

template <typename First, typename Second>
inline SequenceParser<First, Second> sequence(First first, Second second) {
    return SequenceParser<First, Second>(std::move(first), std::move(second));
}

template <typename First, typename Second>
inline AlternativeParser<First, Second> alternative(First first, Second second) {
    return AlternativeParser<First, Second>(std::move(first), std::move(second));
}

inline ByteRangeWhileParser rangeWhile(std::uint8_t start, std::uint8_t end, std::size_t min,
                                       std::optional<std::size_t> max) {
    return ByteRangeWhileParser(start, end, min, max);
}

inline ConfixParser confix(std::uint8_t open, std::uint8_t close, bool allowNested) {
    return ConfixParser(open, close, allowNested);
}

template <typename ParserT, typename Mapper>
inline MapParser<ParserT, Mapper> map(ParserT parser, Mapper mapper) {
    return MapParser<ParserT, Mapper>(std::move(parser), std::move(mapper));
}

inline bool isSpace(std::uint8_t byte) {
    return byte == static_cast<std::uint8_t>(' ') || byte == static_cast<std::uint8_t>('\t');
}

inline bool isCrlf(std::uint8_t byte) {
    return byte == static_cast<std::uint8_t>('\r') || byte == static_cast<std::uint8_t>('\n');
}

inline bool isAlpha(std::uint8_t byte) {
    return (byte >= static_cast<std::uint8_t>('A') && byte <= static_cast<std::uint8_t>('Z')) ||
           (byte >= static_cast<std::uint8_t>('a') && byte <= static_cast<std::uint8_t>('z'));
}

inline bool isDigit(std::uint8_t byte) {
    return byte >= static_cast<std::uint8_t>('0') && byte <= static_cast<std::uint8_t>('9');
}

inline bool isTokenChar(std::uint8_t byte) {
    if (isDigit(byte) || isAlpha(byte)) {
        return true;
    }
    switch (byte) {
        case '!':
        case '#':
        case '$':
        case '%':
        case '&':
        case '\'':
        case '*':
        case '+':
        case '-':
        case '.':
        case '^':
        case '_':
        case '`':
        case '|':
        case '~':
            return true;
        default:
            return false;
    }
}

} // namespace ir
} // namespace cppfort
