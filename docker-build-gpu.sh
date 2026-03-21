#!/bin/bash
set -e

VERSION=$(cat VERSION)

docker build \
    --build-arg UID="$(id -u)" \
    --build-arg GID="$(id -g)" \
    --build-arg UNAME="$(whoami)" \
    --build-arg VERSION="${VERSION}" \
    -f Dockerfile.gpu \
    -t motion-mag-dtcwt-gpu:${VERSION} \
    -t motion-mag-dtcwt-gpu:latest .

echo "Built motion-mag-dtcwt-gpu:${VERSION} (also tagged :latest)"
