#!/bin/bash
set -e

docker build \
    --build-arg UID="$(id -u)" \
    --build-arg GID="$(id -g)" \
    --build-arg UNAME="$(whoami)" \
    -t motion-magnification-dtcwt .

echo "Built motion-magnification-dtcwt image as user: $(whoami) (uid=$(id -u), gid=$(id -g))"
