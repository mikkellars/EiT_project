FROM ros:melodic
RUN apt update && apt install -y ros-melodic-joy git libusb-1.0-0-dev curl cabextract
RUN git clone https://github.com/medusalix/xow /xov
WORKDIR /xov
RUN make BUILD=RELEASE
COPY joy.launch /joy.launch
