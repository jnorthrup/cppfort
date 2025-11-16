#include <iostream>
#include <string>

// Fixed template confix context for >> disambiguation
struct TemplateConfixContext {
    int angle_depth = 0;
    bool in_template_confix = false;
    
    void update(char c) {
        if (c == '<') {
            angle_depth++;
            in_template_confix = true;
        } else if (c == '>') {
            angle_depth--;
            if (angle_depth <= 0) {
                angle_depth = 0;
                in_template_confix = false;
            }
        }
    }
    
    bool should_split_double_angle() const {
        return in_template_confix && angle_depth >= 2;
    }
};

// Fixed processing logic
void process_with_template_context(const std::string& text) {
    std::cout << "Processing: " << text << "\n";
    
    TemplateConfixContext template_ctx;
    
    for (size_t i = 0; i < text.length(); ++i) {
        char c = text[i];
        
        // Update context FIRST
        template_ctx.update(c);
        
        if (c == '>') {
            std::cout << "Position " << i << ": Found '>'\n";
            std::cout << "  After update: depth=" << template_ctx.angle_depth 
                     << ", in_template=" << template_ctx.in_template_confix << "\n";
            
            // Check for >>
            if (i + 1 < text.length() && text[i + 1] == '>') {
                if (template_ctx.should_split_double_angle()) {
                    std::cout << "  -> SPLIT: Treat as two template closes\n";
                    // Process first >
                    std::cout << "  -> Added angle close at " << i << "\n";
                    // Process second >
                    std::cout << "  -> Added angle close at " << (i + 1) << "\n";
                    i++; // Skip the second >
                } else {
                    std::cout << "  -> KEEP: Treat as right-shift operator\n";
                }
            } else {
                std::cout << "  -> Single angle close\n";
            }
        }
    }
    
    std::cout << "Final: depth=" << template_ctx.angle_depth 
             << ", in_template=" << template_ctx.in_template_confix << "\n\n";
}

int main() {
    std::cout << "Testing fixed template confix context:\n\n";
    
    std::vector<std::string> test_cases = {
        "a >> b",                    // Should be shift
        "vector<int> x;",            // Template
        "vector<map<string,int>> data;", // Template with >>
        "if (x >> 2) {",             // Shift in condition
        "cout << data >> 8;"         // Stream operators
    };
    
    for (const auto& test : test_cases) {
        process_with_template_context(test);
    }
    
    return 0;
}
