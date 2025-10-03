// cf069-goto-state-machine.cpp
// Goto-based state machine
// Test #069


int test_goto_state_machine(int input) {
    int state = 0;
    int result = 0;

state0:
    if (input == 0) goto state1;
    goto end;

state1:
    result = 10;
    if (input == 0) goto state2;
    goto end;

state2:
    result = 20;

end:
    return result;
}

int main() {
    return test_goto_state_machine(0);
}
