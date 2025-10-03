// cf051-switch-no-default.cpp
// Switch without default case
// Test #051


int test_switch_no_default(int x) {
    int result = -1;
    switch (x) {
        case 1:
            result = 10;
            break;
        case 2:
            result = 20;
            break;
    }
    return result;
}

int main() {
    return test_switch_no_default(1);
}
