#include <iostream>
#include <string>

// Template confix context for >> disambiguation
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

void debug_process(const std::string& text) {
    std::cout << "=== Processing: " << text << " ===\n";
    
    TemplateConfixContext template_ctx;
    
    for (size_t i = 0; i < text.length(); ++i) {
        char c = text[i];
        
        std::cout << "Pos " << i << ": '" << c << "'";
        
        // Check for >> BEFORE updating context
        if (c == '>' && i + 1 < text.length() && text[i + 1] == '>') {
            std::cout << " [>> detected]";
            bool should_split = template_ctx.should_split_double_angle();
            std::cout << " - should_split=" << should_split 
                     << " (depth=" << template_ctx.angle_depth 
                     << ", in_template=" << template_ctx.in_template_confix << ")";
            
            if (should_split) {
                std::cout << " -> SPLIT into two template closes\n";
                // Process first >
                std::cout << "   -> Added angle close at position " << i << "\n";
                // Process second >
                std::cout << "   -> Added angle close at position " << (i + 1) << "\n";
                template_ctx.update(c); // Update for first >
                template_ctx.update(text[i + 1]); // Update for second >
                i++; // Skip the second >
                continue;
            } else {
                std::cout << " -> KEEP as right-shift operator\n";
            }
        }
        
        // Update context for normal characters
        template_ctx.update(c);
        
        if (c == '<' || c == '>') {
            std::cout << " - updated: depth=" << template_ctx.angle_depth 
                     << ", in_template=" << template_ctx.in_template_confix;
        }
        
        std::cout << "\n";
    }
    
    std::cout << "Final: depth=" << template_ctx.angle_depth 
             << ", in_template=" << template_ctx.in_template_confix << "\n\n";
}

int main() {
    std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  Debug Template Confix Logic                               ║\n";
    std::cout << "║  For >> Disambiguation                                     ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n\n";
    
    std::vector<std::string> test_cases = {
        "a >> b",                    // Should be shift
        "vector<int> x;",            // Template
        "vector<map<string,int>> data;", // Template with >>
        "if (x >> 2) {",             // Shift in condition
        "cout << data >> 8;"         // Stream operators
    };
    
    for (const auto& test : test_cases) {
        debug_process(test);
    }
    
    return 0;
}
