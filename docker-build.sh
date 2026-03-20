#!/bin/bash
set -e

VERSION=$(cat VERSION)

docker build \
    --build-arg UID="$(id -u)" \
    --build-arg GID="$(id -g)" \
    --build-arg UNAME="$(whoami)" \
    --build-arg VERSION="${VERSION}" \
    -t motion-mag-dtcwt:${VERSION} \
    -t motion-mag-dtcwt:latest .

echo "Built motion-mag-dtcwt:${VERSION} (also tagged :latest)"
