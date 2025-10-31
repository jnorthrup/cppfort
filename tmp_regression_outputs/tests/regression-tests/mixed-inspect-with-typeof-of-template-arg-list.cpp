
template <int i, int j>
auto calc() {
    return i + j;
}

int fun(auto v) { return inspect v -> int {
        is int  = calc<1,2>();
        is _ = 0;
    }; }

int main() { return fun(42); }