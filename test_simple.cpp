#include <iostream>
#include <string>
#include <regex>

std::string cpp2_to_cpp(const std::string& source) {
    std::string output = source;
    
    // Function transformation
    std::regex func_regex(R"((\w+)\s*:\s*\(([^)]*)\)\s*(?:->\s*([^=\n{]+))?\s*=\s*\{)");
    std::smatch match;
    std::string::const_iterator search_start(output.cbegin());
    
    while (std::regex_search(search_start, output.cend(), match, func_regex)) {
        std::string func_name = match[1].str();
        std::string params = match[2].str();
        std::string ret_type = match[3].str();
        
        std::string cpp_signature = "auto " + func_name + "(" + params + ")";
        if (!ret_type.empty() && ret_type != "auto") {
            cpp_signature += " -> " + ret_type;
        }
        cpp_signature += " {";
        
        size_t start_pos = match.position(0);
        size_t length = match.length(0);
        output.replace(start_pos, length, cpp_signature);
        
        search_start = output.cbegin() + start_pos + cpp_signature.length();
        std::cout << "Transformed: " << func_name << std::endl;
    }
    
    return output;
}

int main() {
    std::string test = "main: () -> int = {\n    x: int = 42;\n    return x;\n}\n";
    std::cout << "Input:\n" << test << "\n";
    std::cout << "Output:\n" << cpp2_to_cpp(test) << std::endl;
    return 0;
}
