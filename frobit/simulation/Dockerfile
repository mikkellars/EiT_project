FROM ros:melodic

RUN /bin/ash -c 'set -ex && \
    ARCH=`uname -m` && \
    if [ "$ARCH" == "x86_64" ]; then \
       echo "x86_64" && \
       apk add some-package; \
    else \
       echo "unknown arch" && \
       apk add some-other-package; \
    fi'