// cf099-state-machine.cpp
// State machine pattern
// Test #099


enum State { INIT, PROCESSING, DONE, ERROR };

int test_state_machine(int input) {
    State state = INIT;
    int result = 0;

    while (state != DONE && state != ERROR) {
        switch (state) {
            case INIT:
                if (input > 0) {
                    state = PROCESSING;
                } else {
                    state = ERROR;
                }
                break;
            case PROCESSING:
                result = input * 2;
                state = DONE;
                break;
            default:
                state = ERROR;
        }
    }

    return (state == DONE) ? result : -1;
}

int main() {
    return test_state_machine(21);
}
