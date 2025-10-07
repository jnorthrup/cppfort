#!/bin/bash
# Move the methods inside namespace
tail -n 10 src/stage0/confix_orbit.cpp > /tmp/methods.txt
head -n -10 src/stage0/confix_orbit.cpp > /tmp/main.txt
cat /tmp/main.txt > src/stage0/confix_orbit.cpp
cat /tmp/methods.txt >> src/stage0/confix_orbit.cpp
echo "} // namespace cppfort::stage0" >> src/stage0/confix_orbit.cpp
