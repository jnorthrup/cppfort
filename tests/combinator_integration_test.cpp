// Combinator Integration Tests
// Real-world parsing scenarios from docs/COMBINATORS.md recipes

#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <memory>
#include <optional>
#include <variant>
#include <functional>
#include <utility>
#include <type_traits>
#include <stdexcept>
#include <cstdint>
#include <iterator>
#include <map>
#include <array>
#include <cassert>
#include <cctype>

namespace cpp2 {
    template<typename T> auto to_string(T const& x) -> std::string {
        if constexpr (std::is_same_v<T, std::string>) { return x; }
        else if constexpr (std::is_same_v<T, const char*>) { return std::string(x); }
        else if constexpr (std::is_same_v<T, char>) { return std::string(1, x); }
        else if constexpr (std::is_same_v<T, bool>) { return x ? "true" : "false"; }
        else if constexpr (std::is_arithmetic_v<T>) { return std::to_string(x); }
        else { std::ostringstream oss; oss << x; return oss.str(); }
    }
    template<typename T, typename U> constexpr auto is(U const& x) -> bool {
        if constexpr (std::is_same_v<T, U> || std::is_base_of_v<T, U>) { return true; }
        else if constexpr (std::is_polymorphic_v<U>) { return dynamic_cast<T const*>(&x) != nullptr; }
        else { return false; }
    }
    template<typename T, typename U> constexpr auto as(U const& x) -> T {
        if constexpr (std::is_same_v<T, U>) { return x; }
        else if constexpr (std::is_base_of_v<T, U>) { return static_cast<T const&>(x); }
        else if constexpr (std::is_polymorphic_v<U>) { return dynamic_cast<T const&>(x); }
        else { return static_cast<T>(x); }
    }
} // namespace cpp2

#include "../include/bytebuffer.hpp"
#include "../include/strview.hpp"
#include "../include/combinators/structural.hpp"
#include "../include/combinators/transformation.hpp"
#include "../include/combinators/reduction.hpp"
#include "../include/combinators/parsing.hpp"

// ============================================================================
// Recipe 1: HTTP Header Parsing
// ============================================================================

auto parse_header_line(cpp2::ByteBuffer line) -> std::optional<std::pair<std::string, std::string>> {
    if (line.empty()) { return std::nullopt; }
    
    const char* data = line.data();
    size_t size = line.size();
    size_t colon_pos = 0;
    bool found = false;
    
    for (size_t i = 0; i < size; i++) {
        if (data[i] == ':') {
            colon_pos = i;
            found = true;
            break;
        }
    }
    
    if (!found) { return std::nullopt; }
    
    auto key_buf = line.slice(0, colon_pos);
    
    size_t value_start = colon_pos + 1;
    while (value_start < size && data[value_start] == ' ') {
        value_start++;
    }
    auto value_buf = line.slice(value_start, size);
    
    size_t value_end = value_buf.size();
    while (value_end > 0 && (value_buf.data()[value_end - 1] == '\r' || 
                             value_buf.data()[value_end - 1] == '\n' || 
                             value_buf.data()[value_end - 1] == ' ')) {
        value_end--;
    }
    
    std::string key(key_buf.data(), key_buf.size());
    std::string value(value_buf.data(), value_end);
    
    return std::make_pair(key, value);
}

auto parse_http_headers(cpp2::ByteBuffer buf) -> std::map<std::string, std::string> {
    std::map<std::string, std::string> result;
    
    auto lines = cpp2::combinators::split(buf, '\n');
    
    for (auto line : lines) {
        auto header = parse_header_line(line);
        if (header.has_value()) {
            result[header.value().first] = header.value().second;
        }
    }
    
    return result;
}

auto test_http_header_parsing() -> void {
    std::cout << "test_http_header_parsing\n";
    
    std::string http_data = "Content-Type: application/json\nContent-Length: 42\nX-Custom-Header: some value\n";
    cpp2::ByteBuffer buf(http_data.data(), http_data.size());
    
    auto headers = parse_http_headers(buf);
    
    assert(headers.size() == 3);
    assert(headers["Content-Type"] == "application/json");
    assert(headers["Content-Length"] == "42");
    assert(headers["X-Custom-Header"] == "some value");
    
    std::cout << "  PASS\n";
}

auto test_http_header_with_crlf() -> void {
    std::cout << "test_http_header_with_crlf\n";
    
    std::string http_data = "Host: example.com\r\nUser-Agent: test\r\n\r\n";
    cpp2::ByteBuffer buf(http_data.data(), http_data.size());
    
    auto headers = parse_http_headers(buf);
    
    assert(headers.size() >= 2);
    assert(headers["Host"] == "example.com");
    assert(headers["User-Agent"] == "test");
    
    std::cout << "  PASS\n";
}

// ============================================================================
// Recipe 2: Null-Terminated String
// ============================================================================

auto read_c_string(cpp2::ByteBuffer buf) -> cpp2::ByteBuffer {
    const char* data = buf.data();
    size_t size = buf.size();
    
    size_t null_pos = 0;
    while (null_pos < size && data[null_pos] != '\0') {
        null_pos++;
    }
    
    return buf.slice(0, null_pos);
}

auto test_c_string_parsing() -> void {
    std::cout << "test_c_string_parsing\n";
    
    std::array<char, 15> data = {'H', 'e', 'l', 'l', 'o', '\0', 'g', 'a', 'r', 'b', 'a', 'g', 'e', '!', '!'};
    cpp2::ByteBuffer buf(data.data(), data.size());
    
    auto str = read_c_string(buf);
    
    assert(str.size() == 5);
    std::string result(str.data(), str.size());
    assert(result == "Hello");
    
    std::cout << "  PASS\n";
}

auto test_c_string_no_null() -> void {
    std::cout << "test_c_string_no_null\n";
    
    std::string data = "NoNull";
    cpp2::ByteBuffer buf(data.data(), data.size());
    
    auto str = read_c_string(buf);
    
    assert(str.size() == 6);
    
    std::cout << "  PASS\n";
}

auto test_c_string_empty() -> void {
    std::cout << "test_c_string_empty\n";
    
    std::array<char, 3> data = {'\0', 'x', 'y'};
    cpp2::ByteBuffer buf(data.data(), data.size());
    
    auto str = read_c_string(buf);
    
    assert(str.empty());
    
    std::cout << "  PASS\n";
}

// ============================================================================
// Recipe 3: Binary Protocol Parsing (Length-Prefixed Messages)
// ============================================================================

struct Message {
    uint8_t msg_type;
    cpp2::ByteBuffer payload;
};

auto parse_message(cpp2::ByteBuffer buf) -> std::optional<std::pair<Message, cpp2::ByteBuffer>> {
    if (buf.size() < 3) { return std::nullopt; }
    
    const char* data = buf.data();
    
    uint8_t msg_type = static_cast<uint8_t>(data[0]);
    
    uint16_t length = (static_cast<uint16_t>(static_cast<uint8_t>(data[2])) << 8) | 
                       static_cast<uint16_t>(static_cast<uint8_t>(data[1]));
    
    if (buf.size() < 3 + length) { return std::nullopt; }
    
    auto payload = buf.slice(3, 3 + length);
    auto remaining = buf.slice(3 + length, buf.size());
    
    Message msg{msg_type, payload};
    return std::make_pair(msg, remaining);
}

auto test_binary_message_parsing() -> void {
    std::cout << "test_binary_message_parsing\n";
    
    std::array<char, 13> data = {
        0x01,                           // type
        0x05, 0x00,                      // length (5, little-endian)
        'H', 'e', 'l', 'l', 'o',        // payload
        0x02,                           // next message type
        0x02, 0x00,                      // length (2)
        'O', 'K'                        // payload
    };
    cpp2::ByteBuffer buf(data.data(), data.size());
    
    auto result1 = parse_message(buf);
    assert(result1.has_value());
    assert(result1.value().first.msg_type == 1);
    assert(result1.value().first.payload.size() == 5);
    std::string payload1(result1.value().first.payload.data(), result1.value().first.payload.size());
    assert(payload1 == "Hello");
    
    auto result2 = parse_message(result1.value().second);
    assert(result2.has_value());
    assert(result2.value().first.msg_type == 2);
    assert(result2.value().first.payload.size() == 2);
    std::string payload2(result2.value().first.payload.data(), result2.value().first.payload.size());
    assert(payload2 == "OK");
    
    assert(result2.value().second.empty());
    
    std::cout << "  PASS\n";
}

auto test_binary_message_truncated() -> void {
    std::cout << "test_binary_message_truncated\n";
    
    std::array<char, 6> data = {0x01, 0x0A, 0x00, 'a', 'b', 'c'};
    cpp2::ByteBuffer buf(data.data(), data.size());
    
    auto result = parse_message(buf);
    assert(!result.has_value());
    
    std::cout << "  PASS\n";
}

// ============================================================================
// Recipe 4: CSV Line Parsing
// ============================================================================

auto parse_csv_line(cpp2::ByteBuffer line) -> std::vector<std::string> {
    std::vector<std::string> result;
    
    auto fields = cpp2::combinators::split(line, ',');
    
    for (auto field : fields) {
        size_t start = 0;
        size_t end = field.size();
        
        while (start < end && (field.data()[start] == ' ' || field.data()[start] == '\t')) {
            start++;
        }
        while (end > start && (field.data()[end - 1] == ' ' || field.data()[end - 1] == '\t')) {
            end--;
        }
        
        if (end > start) {
            result.push_back(std::string(field.data() + start, end - start));
        } else {
            result.push_back("");
        }
    }
    
    return result;
}

auto test_csv_parsing() -> void {
    std::cout << "test_csv_parsing\n";
    
    std::string line = "name, age, city";
    cpp2::ByteBuffer buf(line.data(), line.size());
    
    auto fields = parse_csv_line(buf);
    
    assert(fields.size() == 3);
    assert(fields[0] == "name");
    assert(fields[1] == "age");
    assert(fields[2] == "city");
    
    std::cout << "  PASS\n";
}

auto test_csv_with_empty_fields() -> void {
    std::cout << "test_csv_with_empty_fields\n";
    
    std::string line = "a,,c";
    cpp2::ByteBuffer buf(line.data(), line.size());
    
    auto fields = parse_csv_line(buf);
    
    assert(fields.size() == 3);
    assert(fields[0] == "a");
    assert(fields[1] == "");
    assert(fields[2] == "c");
    
    std::cout << "  PASS\n";
}

// ============================================================================
// Recipe 5: Log Entry Parsing
// ============================================================================

struct LogEntry {
    std::string timestamp;
    std::string level;
    std::string message;
};

auto parse_log_entry(cpp2::ByteBuffer line) -> std::optional<LogEntry> {
    if (line.size() < 5) { return std::nullopt; }
    
    const char* data = line.data();
    size_t size = line.size();
    
    if (data[0] != '[') { return std::nullopt; }
    
    size_t bracket_end = 1;
    while (bracket_end < size && data[bracket_end] != ']') {
        bracket_end++;
    }
    if (bracket_end >= size) { return std::nullopt; }
    
    std::string timestamp(data + 1, bracket_end - 1);
    
    size_t pos = bracket_end + 1;
    while (pos < size && data[pos] == ' ') {
        pos++;
    }
    
    size_t level_start = pos;
    while (pos < size && data[pos] != ':') {
        pos++;
    }
    if (pos >= size) { return std::nullopt; }
    
    std::string level(data + level_start, pos - level_start);
    
    pos++;
    while (pos < size && data[pos] == ' ') {
        pos++;
    }
    
    std::string message(data + pos, size - pos);
    
    while (!message.empty() && (message.back() == '\n' || message.back() == '\r')) {
        message.pop_back();
    }
    
    return LogEntry{timestamp, level, message};
}

auto test_log_parsing() -> void {
    std::cout << "test_log_parsing\n";
    
    std::string line = "[2024-01-15 10:30:45] INFO: Server started\n";
    cpp2::ByteBuffer buf(line.data(), line.size());
    
    auto entry = parse_log_entry(buf);
    
    assert(entry.has_value());
    assert(entry.value().timestamp == "2024-01-15 10:30:45");
    assert(entry.value().level == "INFO");
    assert(entry.value().message == "Server started");
    
    std::cout << "  PASS\n";
}

auto test_log_parsing_multiline() -> void {
    std::cout << "test_log_parsing_multiline\n";
    
    std::string logs = "[2024-01-15 10:30:45] INFO: Starting\n[2024-01-15 10:30:46] ERROR: Failed\n[2024-01-15 10:30:47] DEBUG: Retrying\n";
    cpp2::ByteBuffer buf(logs.data(), logs.size());
    
    std::vector<LogEntry> entries;
    
    auto lines = cpp2::combinators::split(buf, '\n');
    for (auto line : lines) {
        if (!line.empty()) {
            auto entry = parse_log_entry(line);
            if (entry.has_value()) {
                entries.push_back(entry.value());
            }
        }
    }
    
    assert(entries.size() == 3);
    assert(entries[0].level == "INFO");
    assert(entries[1].level == "ERROR");
    assert(entries[2].level == "DEBUG");
    
    std::cout << "  PASS\n";
}

// ============================================================================
// Recipe 6: Pipeline Composition for Data Transformation
// ============================================================================

auto test_word_count_pipeline() -> void {
    std::cout << "test_word_count_pipeline\n";
    
    std::string text = "hello world hello cpp2 world world";
    cpp2::ByteBuffer buf(text.data(), text.size());
    
    auto words = cpp2::combinators::split(buf, ' ');
    
    std::map<std::string, int> counts;
    for (auto word : words) {
        if (!word.empty()) {
            std::string w(word.data(), word.size());
            counts[w]++;
        }
    }
    
    assert(counts["hello"] == 2);
    assert(counts["world"] == 3);
    assert(counts["cpp2"] == 1);
    
    std::cout << "  PASS\n";
}

auto test_filter_transform_pipeline() -> void {
    std::cout << "test_filter_transform_pipeline\n";
    
    std::string data = "a1B2c3D4e5F6";
    cpp2::ByteBuffer buf(data.data(), data.size());
    
    auto letters = cpp2::combinators::from(buf)
        | cpp2::combinators::curried::filter([](char c) -> bool { return std::isalpha(c) != 0; })
        | cpp2::combinators::curried::map([](char c) -> char { return static_cast<char>(std::toupper(c)); });
    
    std::string result;
    for (char c : letters) {
        result += c;
    }
    
    assert(result == "ABCDEF");
    
    std::cout << "  PASS\n";
}

auto test_skip_take_sum_pipeline() -> void {
    std::cout << "test_skip_take_sum_pipeline\n";
    
    std::string data = "01234567";
    cpp2::ByteBuffer buf(data.data(), data.size());
    
    auto selected = cpp2::combinators::from(buf)
        | cpp2::combinators::curried::skip(2)
        | cpp2::combinators::curried::take(4);
    
    int sum = 0;
    for (char c : selected) {
        sum += c - '0';
    }
    
    assert(sum == 14);  // 2+3+4+5
    
    std::cout << "  PASS\n";
}

// ============================================================================
// Recipe 7: Configuration File Parsing
// ============================================================================

auto parse_config(cpp2::ByteBuffer buf) -> std::map<std::string, std::string> {
    std::map<std::string, std::string> result;
    
    auto lines = cpp2::combinators::split(buf, '\n');
    
    for (auto line : lines) {
        if (line.empty()) { continue; }
        
        if (line.data()[0] == '#') { continue; }
        
        const char* data = line.data();
        size_t size = line.size();
        size_t eq_pos = 0;
        bool found = false;
        
        for (size_t i = 0; i < size; i++) {
            if (data[i] == '=') {
                eq_pos = i;
                found = true;
                break;
            }
        }
        
        if (found) {
            std::string key(data, eq_pos);
            std::string value(data + eq_pos + 1, size - eq_pos - 1);
            
            while (!value.empty() && (value.back() == '\r' || value.back() == '\n')) {
                value.pop_back();
            }
            
            result[key] = value;
        }
    }
    
    return result;
}

auto test_config_file_parsing() -> void {
    std::cout << "test_config_file_parsing\n";
    
    std::string config = "# Configuration file\nhost=localhost\nport=8080\ndebug=true\n";
    cpp2::ByteBuffer buf(config.data(), config.size());
    
    auto settings = parse_config(buf);
    
    assert(settings.size() == 3);
    assert(settings["host"] == "localhost");
    assert(settings["port"] == "8080");
    assert(settings["debug"] == "true");
    
    std::cout << "  PASS\n";
}

// ============================================================================
// Recipe 8: Find Patterns in Data
// ============================================================================

auto test_find_pattern() -> void {
    std::cout << "test_find_pattern\n";
    
    std::string data = "The quick brown fox jumps over the lazy dog";
    cpp2::ByteBuffer buf(data.data(), data.size());
    
    auto found_vowel = cpp2::combinators::reduce_from(buf)
        .find([](char c) -> bool { return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u'; });
    
    assert(found_vowel.has_value());
    assert(found_vowel.value() == 'e');
    
    auto q_index = cpp2::combinators::reduce_from(buf)
        .find_index([](char c) -> bool { return c == 'q'; });
    
    assert(q_index.has_value());
    assert(q_index.value() == 4);
    
    std::cout << "  PASS\n";
}

auto test_any_all_predicates() -> void {
    std::cout << "test_any_all_predicates\n";
    
    std::string digits = "1234567890";
    cpp2::ByteBuffer buf1(digits.data(), digits.size());
    
    bool all_digits = cpp2::combinators::reduce_from(buf1)
        .all([](char c) -> bool { return std::isdigit(c) != 0; });
    assert(all_digits);
    
    cpp2::ByteBuffer buf2(digits.data(), digits.size());
    bool any_even = cpp2::combinators::reduce_from(buf2)
        .any([](char c) -> bool { return (c - '0') % 2 == 0; });
    assert(any_even);
    
    std::cout << "  PASS\n";
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    std::cout << "=== Combinator Integration Tests ===\n";
    std::cout << "Real-world parsing scenarios\n\n";
    
    std::cout << "--- HTTP Header Parsing ---\n";
    test_http_header_parsing();
    test_http_header_with_crlf();
    
    std::cout << "\n--- Null-Terminated Strings ---\n";
    test_c_string_parsing();
    test_c_string_no_null();
    test_c_string_empty();
    
    std::cout << "\n--- Binary Protocol ---\n";
    test_binary_message_parsing();
    test_binary_message_truncated();
    
    std::cout << "\n--- CSV Parsing ---\n";
    test_csv_parsing();
    test_csv_with_empty_fields();
    
    std::cout << "\n--- Log Parsing ---\n";
    test_log_parsing();
    test_log_parsing_multiline();
    
    std::cout << "\n--- Pipeline Composition ---\n";
    test_word_count_pipeline();
    test_filter_transform_pipeline();
    test_skip_take_sum_pipeline();
    
    std::cout << "\n--- Configuration Files ---\n";
    test_config_file_parsing();
    
    std::cout << "\n--- Pattern Finding ---\n";
    test_find_pattern();
    test_any_all_predicates();
    
    std::cout << "\n=== All Integration Tests PASSED ===\n";
    return 0;
}
