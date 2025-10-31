#include <set>

int fine() { auto up = unique.new<int>(1);
    auto sp = shared.new<int>(2);
    std::optional<int> op = (3);

    return up* + sp* + op*; }

int bad_shared_ptr_access() { auto sp = std::make_shared<int>(1);
    sp.reset();
    return sp*; }

int main() { std::set_terminate(std::abort);
    return fine() + bad_shared_ptr_access(); }