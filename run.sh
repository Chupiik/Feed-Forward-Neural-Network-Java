#!/bin/bash

. /etc/profile.d/modules.sh

module add jdk

mkdir -p bin
javac -d bin src/*.java

echo "Running Neural Network with low priority..."
nice -n 19 java -Xmx4g -cp bin Main