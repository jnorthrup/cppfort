#include <string>

#include <cstdlib>
#include <ctime>

void copy_from(copy _) {}

void parameter_styles(std::string _, std::string // "in" is default b, std::string& _, std::string&& d, std::string&& e) { int z = 12;

    z++;
    b += "plugh";

    if std::rand()%2 {
        z++;
        copy_from(b);   // definite last use
    }
    else {
        copy_from(b&);  // NB: better not move from this (why not?)
        copy_from(d);
        copy_from(z++);
        copy_from(e);
    }

    // std::move(z);

    copy_from(z);

    :time(nullptr)%2 if std = = 0 {
        copy_from(z);
    }; }

int main() {}
