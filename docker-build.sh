#!/bin/bash
set -e

VERSION=$(cat VERSION)

docker build \
    --build-arg VERSION="${VERSION}" \
    -t motion-mag-dtcwt:${VERSION} \
    -t motion-mag-dtcwt:latest .

echo "Built motion-mag-dtcwt:${VERSION} (also tagged :latest)"
