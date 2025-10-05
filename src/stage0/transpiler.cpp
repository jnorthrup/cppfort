#include "transpiler.h"
#include <iostream>
#include <sstream>
#include <regex>

namespace cppfort::stage0 {

std::string Transpiler::transformCpp2ToCpp(const std::string& cpp2_body) {
    std::string result = cpp2_body;
    std::cout << "DEBUG: Input body: " << result << std::endl;
    
    // First, transform 'as' casts to static_cast
    // Pattern: expression as Type
    // Becomes: static_cast<Type>(expression)
    std::regex as_cast(R"(\s+(\w+(?:\([^)]*\))?|"[^"]*"|\d+(?:\.\d+)?)\s+as\s+([^;,\)\}\s]+))");
    result = std::regex_replace(result, as_cast, " static_cast<$2>($1)");
    std::cout << "DEBUG: After as casts: " << result << std::endl;
    
    // Transform colon declarations with initializers
    // Pattern: variable_name: Type = initializer;
    // Becomes: Type variable_name = initializer;
    std::regex colon_decl_with_init(R"(\b(\w+)\s*:\s*([^=;]+?)\s*=\s*([^;]+);)");
    result = std::regex_replace(result, colon_decl_with_init, "$2 $1 = $3;");
    std::cout << "DEBUG: After colon decls with init: " << result << std::endl;
    
    // Transform colon declarations without initializers
    // Pattern: variable_name: Type; (must be at start of line/statement)
    // Becomes: Type variable_name;
    std::regex colon_decl_no_init(R"(^\s*(\w+)\s*:\s*([^;=]+);\s*$)");
    result = std::regex_replace(result, colon_decl_no_init, "$2 $1;");
    std::cout << "DEBUG: After colon decls no init: " << result << std::endl;
    // () should become {} for default initialization
    std::regex empty_init(R"(\s+=\s*\(\)\s*;)");
    result = std::regex_replace(result, empty_init, " = {}; ");
    
    // Clean up extra spaces and semicolons
    std::regex extra_spaces(R"(\s+)");
    result = std::regex_replace(result, extra_spaces, " ");
    std::regex double_semicolons(R"(;;)");
    result = std::regex_replace(result, double_semicolons, ";");
    
    return result;
}

TranslationUnit Transpiler::parse(const std::string& source, const std::string& filename) {
    TranslationUnit unit;
    unit.source_grammar = "CPP2";
    
    std::istringstream iss(source);
    std::string line;
    bool in_function = false;
    OrbitFunctionDecl current_function;
    std::string function_body;
    int brace_count = 0;
    
    while (std::getline(iss, line)) {
        // Remove trailing whitespace
        line.erase(line.find_last_not_of(" \t\r\n") + 1);
        
        // Skip empty lines and comments
        if (line.empty() || line.substr(0, 2) == "//") {
            continue;
        }
        
        // Handle includes
        if (line.substr(0, 8) == "#include") {
            OrbitIncludeDecl include;
            include.path = line;
            unit.includes.push_back(include);
            continue;
        }
        
        // Look for function declaration: name: (params) -> return_type = {
        std::regex func_pattern(R"((\w+):\s*\(([^)]*)\)\s*->\s*([^=]+)\s*=\s*\{)");
        std::smatch match;
        if (std::regex_search(line, match, func_pattern)) {
            // Start of function
            current_function.name = match[1].str();
            current_function.return_type = match[3].str();
            
            // Parse parameters
            std::string params_str = match[2].str();
            if (!params_str.empty()) {
                std::regex param_pattern(R"(\s*(\w+)\s*:\s*([^,]+)\s*,?)");
                std::sregex_iterator iter(params_str.begin(), params_str.end(), param_pattern);
                std::sregex_iterator end;
                
                for (; iter != end; ++iter) {
                    OrbitParameter param;
                    param.name = (*iter)[1].str();
                    param.type = (*iter)[2].str();
                    current_function.parameters.push_back(param);
                }
            }
            
            in_function = true;
            brace_count = 1; // We already saw the opening brace
            function_body = "{\n";
            continue;
        }
        
        // Look for function declaration split across lines: name: (params) -> return_type
        std::regex func_start_pattern(R"((\w+):\s*\(([^)]*)\)\s*->\s*(.+))");
        if (std::regex_match(line, match, func_start_pattern) && !in_function) {
            current_function.name = match[1].str();
            current_function.return_type = match[3].str();
            
            // Parse parameters
            std::string params_str = match[2].str();
            if (!params_str.empty()) {
                std::regex param_pattern(R"(\s*(\w+)\s*:\s*([^,]+)\s*,?)");
                std::sregex_iterator iter(params_str.begin(), params_str.end(), param_pattern);
                std::sregex_iterator end;
                
                for (; iter != end; ++iter) {
                    OrbitParameter param;
                    param.name = (*iter)[1].str();
                    param.type = (*iter)[2].str();
                    current_function.parameters.push_back(param);
                }
            }
            continue;
        }
        
        // Look for = { on a separate line
        if (!in_function && !current_function.name.empty() && 
            std::regex_match(line, std::regex(R"(\s*=\s*\{\s*)"))) {
            in_function = true;
            brace_count = 1;
            function_body = "{\n";
            continue;
        }
        
        // Accumulate function body with brace counting
        if (in_function) {
            for (char c : line) {
                if (c == '{') brace_count++;
                else if (c == '}') brace_count--;
            }
            
            function_body += line + "\n";
            
            // Check if function is complete
            if (brace_count == 0) {
                OrbitBlock block;
                block.statements = {}; // TODO: parse individual statements
                current_function.body = block;
                // Transform cpp2 syntax to C++ syntax in the body
                std::string transformed_body = transformCpp2ToCpp(function_body);
                current_function.raw_body = transformed_body;
                unit.functions.push_back(current_function);
                in_function = false;
                current_function = OrbitFunctionDecl{};
                function_body.clear();
            }
            continue;
        }
    }
    
    return unit;
}

} // namespace cppfort::stage0