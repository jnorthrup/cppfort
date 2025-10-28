#include <cassert>
#include <iostream>
#include <sstream>
#include <string>

#include "cpp2_emitter.h"
#include "orbit_pipeline.h"

using cppfort::stage0::testing::transform_parameter_for_testing;

int main() {
    cppfort::stage0::OrbitPipeline pipeline;
    const std::string pattern_path = "../../../patterns/bnfc_cpp2_complete.yaml";
    assert(pipeline.load_patterns(pattern_path));
    const auto& patterns = pipeline.patterns();

    const cppfort::stage0::PatternData* parameter_pattern = nullptr;
    for (const auto& pattern : patterns) {
        if (pattern.name == "cpp2_parameter") {
            parameter_pattern = &pattern;
            break;
        }
    }
    assert(parameter_pattern && "cpp2_parameter pattern not found");

    auto trim_copy = [](const std::string& text) {
        size_t begin = 0;
        size_t end = text.size();
        while (begin < end && std::isspace(static_cast<unsigned char>(text[begin]))) ++begin;
        while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1]))) --end;
        return text.substr(begin, end - begin);
    };

    cppfort::stage0::CPP2Emitter evidence_emitter;

    auto assert_cpp2_parameter = [&](const std::string& text, const std::string& expected_name, const std::string& expected_type) {
        auto segments = evidence_emitter.extract_alternating_segments(text, *parameter_pattern);
        assert(segments.size() >= 2);
        std::string raw_name = trim_copy(segments.front());
        size_t token_pos = raw_name.find_last_of(' ');
        if (token_pos != std::string::npos) {
            raw_name = raw_name.substr(token_pos + 1);
        }
        std::string raw_type = trim_copy(segments.back());
        assert(raw_name == expected_name);
        assert(raw_type == expected_type);
    };

    {
        const std::string transformed = transform_parameter_for_testing("inout s: std::string");
        assert(transformed == "std::string& s");
        assert_cpp2_parameter("inout s: std::string", "s", "std::string");
    }

    {
        const std::string transformed = transform_parameter_for_testing("in_ref name: std::string");
        assert(transformed == "const std::string& name");
        assert_cpp2_parameter("in_ref name: std::string", "name", "std::string");
    }

    {
        const std::string transformed = transform_parameter_for_testing("copy value: int");
        assert(transformed == "int value");
        assert_cpp2_parameter("copy value: int", "value", "int");
    }

    {
        const std::string transformed = transform_parameter_for_testing("move payload: std::string");
        assert(transformed == "std::string&& payload");
        assert_cpp2_parameter("move payload: std::string", "payload", "std::string");
    }

    {
        const std::string transformed = transform_parameter_for_testing("forward value: std::string");
        assert(transformed == "std::string&& value");
        assert_cpp2_parameter("forward value: std::string", "value", "std::string");
    }

    {
        const std::string transformed = transform_parameter_for_testing("forward args...: Args");
        assert(transformed == "Args&&... args");
        assert_cpp2_parameter("forward args...: Args", "args...", "Args");
    }

    {
        const std::string transformed = transform_parameter_for_testing("out result: Widget");
        assert(transformed == "cpp2::impl::out<Widget> result");
        assert_cpp2_parameter("out result: Widget", "result", "Widget");
    }

    {
        const std::string transformed = transform_parameter_for_testing("out outputs...: Value");
        assert(transformed == "cpp2::impl::out<Value>... outputs");
        assert_cpp2_parameter("out outputs...: Value", "outputs...", "Value");
    }

    {
        // Unsupported constructs should fall back to original text.
        const std::string transformed = transform_parameter_for_testing("this");
        assert(transformed == "this");
    }

    {
        const std::string input =
            "foo: (inout s: std::string, forward args...: Args) -> std::string = {\n"
            "    return s;\n"
            "}";

        cppfort::stage0::CPP2Emitter emitter;
        std::vector<cppfort::stage0::PatternData> patterns;
        std::ostringstream out;
        emitter.emit_depth_based(input, out, patterns);

        const std::string expected =
            "#include <string>\n"
            "std::string foo(std::string& s, Args&&... args) {\n"
            "    return s;\n"
            "}";
        assert(out.str() == expected);
    }

    {
        const std::string input = "id: (x: int) = x;";

        cppfort::stage0::CPP2Emitter emitter;
        std::vector<cppfort::stage0::PatternData> patterns;
        std::ostringstream out;
        emitter.emit_depth_based(input, out, patterns);

        const std::string expected = "auto id(const int& x) { return x; }";
        assert(out.str() == expected);
    }

    std::cout << "Parameter transformation unit tests passed.\n";
    return 0;
}
