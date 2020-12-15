docker run --network host --name ros-fence-inspection --mount type=bind,source="$(pwd)"/nodes/detection/images,target=/images --rm -it ros-fence-inspection

