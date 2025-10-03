// cf054-switch-enum.cpp
// Switch on enum type
// Test #054


enum Color { RED = 0, GREEN = 1, BLUE = 2 };

int test_switch_enum(Color c) {
    switch (c) {
        case RED:
            return 10;
        case GREEN:
            return 20;
        case BLUE:
            return 30;
    }
    return 0;
}

int main() {
    return test_switch_enum(GREEN);
}
