// ar034-float-nan.cpp
// Float NaN generation
// Test #034


float test_nan() {
    return 0.0f / 0.0f;
}

int main() {
    float result = test_nan();
    return (result != result) ? 1 : 0;  // NaN != NaN
}
