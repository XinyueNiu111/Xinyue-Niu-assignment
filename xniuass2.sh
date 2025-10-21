#!/bin/bash

for i in {1..5}
do
    echo "This is the first line of content of file $i." >File-$i.txt
    echo "This is the second line of content of file $i." >>File-$i.txt

    mkdir -p Directory-$i
    cp File-$i.txt Directory-$i/
    mv File-$i.txt Old-File-$i.txt
done
echo "Script execution completed!"
