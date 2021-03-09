SOURCE_DIR = src
BUILD_DIR = build
EXEC_FILE = NN-classifier

CPU_SOURCE_FILES := $(shell find $(SOURCEDIR) -name '*.cpp')
GPU_SOURCE_FILES := $(shell find $(SOURCEDIR) -name '*.cu')

build: FORCE
	mkdir -p ${BUILD_DIR}
	touch ${SOURCE_DIR}/dataset/coordinates_output_zero.txt
	touch ${SOURCE_DIR}/dataset/coordinates_output_one.txt
	touch ${SOURCE_DIR}/dataset/coordinates_targets_zero.txt
	touch ${SOURCE_DIR}/dataset/coordinates_targets_one.txt

	nvcc ${CPU_SOURCE_FILES} ${GPU_SOURCE_FILES} -lineinfo -o ${BUILD_DIR}/${EXEC_FILE}

run:
	./${BUILD_DIR}/${EXEC_FILE}
	python3 ${SOURCE_DIR}/dataset/plot_coordinates.py

clean:
	rm -rf ${BUILD_DIR}

FORCE: