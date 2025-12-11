#include "sha256.hpp"
#include <sstream>
#include <iomanip>

namespace cppfort::tests {

std::vector<uint8_t> SHA256::pad(const std::vector<uint8_t>& input) {
    uint64_t bit_len = input.size() * 8;
    std::vector<uint8_t> msg = input;
    msg.push_back(0x80);

    while ((msg.size() % 64) != 56) {
        msg.push_back(0);
    }

    for (int i = 7; i >= 0; --i) {
        msg.push_back((bit_len >> (i * 8)) & 0xFF);
    }

    return msg;
}

std::string SHA256::hash(const std::string& input) {
    std::vector<uint8_t> msg(input.begin(), input.end());
    std::vector<uint8_t> padded = pad(msg);

    uint32_t H[8] = {H[0], H[1], H[2], H[3], H[4], H[5], H[6], H[7]};

    for (size_t chunk = 0; chunk < padded.size(); chunk += 64) {
        uint32_t W[64];

        for (int i = 0; i < 16; ++i) {
            W[i] = (static_cast<uint32_t>(padded[chunk + i*4]) << 24) |
                  (static_cast<uint32_t>(padded[chunk + i*4 + 1]) << 16) |
                  (static_cast<uint32_t>(padded[chunk + i*4 + 2]) << 8) |
                  (static_cast<uint32_t>(padded[chunk + i*4 + 3]));
        }

        for (int i = 16; i < 64; ++i) {
            W[i] = sigma1(W[i-2]) + W[i-7] + sigma0(W[i-15]) + W[i-16];
        }

        uint32_t a = H[0];
        uint32_t b = H[1];
        uint32_t c = H[2];
        uint32_t d = H[3];
        uint32_t e = H[4];
        uint32_t f = H[5];
        uint32_t g = H[6];
        uint32_t h = H[7];

        for (int i = 0; i < 64; ++i) {
            uint32_t S1 = Sigma1(e);
            uint32_t ch = Ch(e, f, g);
            uint32_t temp1 = h + S1 + ch + K[i] + W[i];
            uint32_t S0 = Sigma0(a);
            uint32_t maj = Maj(a, b, c);
            uint32_t temp2 = S0 + maj;

            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }

        H[0] += a;
        H[1] += b;
        H[2] += c;
        H[3] += d;
        H[4] += e;
        H[5] += f;
        H[6] += g;
        H[7] += h;
    }

    std::stringstream ss;
    for (int i = 0; i < 8; ++i) {
        ss << std::hex << std::setw(8) << std::setfill('0') << H[i];
    }

    return ss.str();
}

} // namespace cppfort::tests