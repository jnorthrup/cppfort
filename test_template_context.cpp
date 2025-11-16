#include <iostream>
#include <string>

// Simple template context tracker for >> disambiguation
struct TemplateContext {
    int angle_depth = 0;
    bool in_template = false;
    
    void update(char c) {
        if (c == '<') {
            angle_depth++;
            in_template = true;
        } else if (c == '>') {
            angle_depth--;
            if (angle_depth <= 0) {
                angle_depth = 0;
                in_template = false;
            }
        }
    }
    
    void reset() {
        angle_depth = 0;
        in_template = false;
    }
    
    bool should_treat_as_template_close() const {
        return in_template && angle_depth >= 2;
    }
};

void test_template_context() {
    std::cout << "Testing template context tracking:\n\n";
    
    struct TestCase {
        std::string code;
        std::string name;
        std::vector<bool> expected_template_states; // For each character
    };
    
    TestCase cases[] = {
        {"vector<int> x;", "simple_template", {false, false, false, true, false, false, false, false}},
        {"a >> b", "right_shift", {false, false, false, false, false}},
        {"vector<map<string,int>> data;", "nested_template", 
         {false, false, false, true, false, true, false, false, false, false, true, false, false, false, false, false, false, false, false}},
        {"if (x >> 2) {", "bitshift_condition", {false, false, false, false, false, false, false, false, false, false, false}},
        {"cout << data >> 8;", "stream_operators", {false, false, false, false, false, false, false, false, false, false, false, false, false, false, false}}
    };
    
    for (const auto& test : cases) {
        std::cout << "Test: " << test.name << "\n";
        std::cout << "Code: " << test.code << "\n";
        
        TemplateContext ctx;
        std::cout << "Template states: ";
        
        for (size_t i = 0; i < test.code.length(); ++i) {
            ctx.update(test.code[i]);
            std::cout << (ctx.in_template ? "T" : "F");
            if (i + 1 < test.code.length()) std::cout << " ";
        }
        std::cout << "\n";
        
        // Test >> detection
        for (size_t i = 0; i < test.code.length() - 1; ++i) {
            if (test.code[i] == '>' && test.code[i + 1] == '>') {
                bool should_split = ctx.should_treat_as_template_close();
                std::cout << "  >> at position " << i << ": " 
                         << (should_split ? "SPLIT (template)" : "KEEP (shift)") << "\n";
            }
        }
        
        std::cout << "Final state: depth=" << ctx.angle_depth 
                 << ", in_template=" << ctx.in_template << "\n\n";
    }
}

void test_specific_cases() {
    std::cout << "Testing specific problematic cases:\n\n";
    
    // Test the >> logic step by step
    std::string test = "vector<map<string,int>> data;";
    std::cout << "Testing: " << test << "\n";
    
    TemplateContext ctx;
    for (size_t i = 0; i < test.length(); ++i) {
        char c = test[i];
        ctx.update(c);
        
        if (c == '>' && i + 1 < test.length() && test[i + 1] == '>') {
            std::cout << "Position " << i << ": Found >>\n";
            std::cout << "  Before update: depth=" << ctx.angle_depth 
                     << ", in_template=" << ctx.in_template << "\n";
            
            bool should_split = ctx.should_treat_as_template_close();
            std::cout << "  Should split: " << should_split << "\n";
            
            if (should_split) {
                std::cout << "  -> Treat as two separate angle closes\n";
            } else {
                std::cout << "  -> Treat as right-shift operator\n";
            }
        }
    }
}

int main() {
    std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  Template Context Tracking Test                            ║\n";
    std::cout << "║  For >> Disambiguation                                     ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n\n";
    
    test_template_context();
    test_specific_cases();
    
    return 0;
}
