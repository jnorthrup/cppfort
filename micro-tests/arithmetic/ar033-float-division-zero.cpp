// ar033-float-division-zero.cpp
// Float division by zero (infinity)
// Test #033


float test_float_div_zero(float x) {
    return x / 0.0f;
}

int main() {
    // Returns infinity
    float result = test_float_div_zero(1.0f);
    return result > 1000.0f ? 1 : 0;
}
