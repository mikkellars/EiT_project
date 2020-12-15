# Running the docker after build
docker run --cpuset-cpus 0 --network host --name ros-fence-inspection --privileged -v /dev/bus/usb:/dev/bus/usb --mount type=bind,source="$(pwd)"/assets,target=/assets --rm -it ros-fence-inspection

# First run if on new terminal

xhost +

# Then to run the docker

docker run --cpuset-cpus 0 --network host --name ros-fence-inspection --privileged -v /tmp/.X11-unix:/tmp/.X11-unix -v /dev/bus/usb:/dev/bus/usb -e DISPLAY=$DISPLAY --mount type=bind,source="$(pwd)"/assets,target=/assets --rm -it ros-fence-inspection
s