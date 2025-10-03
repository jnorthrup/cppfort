// cf057-switch-duffs-device.cpp
// Duff's device (switch/loop interleaving)
// Test #057


void test_duffs_device(int* dest, const int* src, int count) {
    int n = (count + 7) / 8;
    switch (count % 8) {
        case 0: do { *dest++ = *src++;
        case 7:      *dest++ = *src++;
        case 6:      *dest++ = *src++;
        case 5:      *dest++ = *src++;
        case 4:      *dest++ = *src++;
        case 3:      *dest++ = *src++;
        case 2:      *dest++ = *src++;
        case 1:      *dest++ = *src++;
        } while (--n > 0);
    }
}

int main() {
    int src[10] = {0,1,2,3,4,5,6,7,8,9};
    int dest[10] = {0};
    test_duffs_device(dest, src, 10);
    return dest[5];
}
