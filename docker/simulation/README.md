# Running the docker
 docker run --network host --name frobit_sim -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY --rm -it frobit_sim