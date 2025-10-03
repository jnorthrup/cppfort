// mem069-enable-shared-from-this.cpp
// Enable shared from this
// Test #149


#include <memory>

class Node : public std::enable_shared_from_this<Node> {
public:
    int value;
    Node(int v) : value(v) {}
    std::shared_ptr<Node> getPtr() {
        return shared_from_this();
    }
};

int main() {
    std::shared_ptr<Node> node = std::make_shared<Node>(42);
    std::shared_ptr<Node> ptr = node->getPtr();
    return ptr->value;
}
