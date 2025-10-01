#include "rabin_karp.h"
#include "orbit_mask.h"
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace cppfort {
namespace ir {

class RabinKarpTest : public ::testing::Test {
protected:
  void SetUp() override {
    rabinKarp = std::make_unique<RabinKarp>();
  }

  std::unique_ptr<RabinKarp> rabinKarp;
};

TEST_F(RabinKarpTest, Initialization) {
  // Test that RabinKarp initializes correctly
  EXPECT_EQ(rabinKarp->orbitCount(0), 0); // No braces initially
  EXPECT_EQ(rabinKarp->orbitCount(1), 0); // No brackets initially
}

TEST_F(RabinKarpTest, ProcessOrbitContext) {
  OrbitContext context(100);
  
  // Add some orbit elements
  context.update('(');
  context.update('{');
  context.update('[');
  
  auto hashes = rabinKarp->processOrbitContext(context);
  
  // Should have computed hashes for all orbit types
  EXPECT_TRUE(hashes[0] != 0 || hashes[1] != 0 || hashes[2] != 0 || 
              hashes[3] != 0 || hashes[4] != 0 || hashes[5] != 0);
}

TEST_F(RabinKarpTest, UpdateOrbitContext) {
  OrbitContext context(100);
  
  // Initial state
  rabinKarp->updateOrbitContext(context);
  EXPECT_EQ(rabinKarp->orbitCount(3), 0); // No parens initially
  
  // Add parentheses
  context.update('(');
  context.update(')');
  rabinKarp->updateOrbitContext(context);
  
  EXPECT_EQ(rabinKarp->orbitCount(3), 2); // Should have 2 paren elements
}

} // namespace ir
} // namespace cppfort