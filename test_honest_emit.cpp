#include "src/stage0/orbit_emitter.h"
#include "src/stage0/orbit_ring.h"
#include "src/stage0/confix_orbit.h"
#include <iostream>

int main() {
    std::string source = "int x = 42;";
    
    cppfort::stage0::ConfixOrbit orbit('{', '}');
    orbit.start_pos = 0;
    orbit.end_pos = source.size();
    
    // Add MODIFIED evidence (not source)
    cppfort::stage0::EvidenceSpan evidence;
    evidence.start_pos = 0;
    evidence.end_pos = source.size();
    evidence.content = "float y = 99.9;";  // DIFFERENT from source
    evidence.confidence = 1.0;
    orbit.add_evidence(evidence);
    
    cppfort::stage0::OrbitEmitter emitter;
    auto tokens = emitter.emit_tokens(&orbit, source);
    
    std::cout << "Source: " << source << "\n";
    std::cout << "Evidence: " << evidence.content << "\n";
    std::cout << "Emitted: " << tokens[0].text << "\n";
    std::cout << "Test: " << (tokens[0].text == evidence.content ? "PASS - uses evidence" : "FAIL - uses source") << "\n";
    
    return 0;
}
