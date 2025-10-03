// mem015-extern-variable.cpp
// Extern variable
// Test #095


extern int extern_var;
int extern_var = 100;

int test_extern() {
    return extern_var;
}

int main() {
    return test_extern();
}
