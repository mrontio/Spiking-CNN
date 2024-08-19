OUTPUT_DIR = ./bin
LIBS = -lcnpy
DEBUG_FLAGS = -g -O0 -D DEBUG
FLAGS = ${DEBUG_FLAGS}

${OUTPUT_DIR}/main: main.cpp Tensor.tpp Tensor.h Convolutional.tpp Convolutional.h AvgPool.h AvgPool.tpp
	g++ ${LIBS} ${FLAGS} -o ${OUTPUT_DIR}/main main.cpp

run: ${OUTPUT_DIR}/main
	${OUTPUT_DIR}/main

test:
	g++ ${LIBS} ${FLAGS} -o ${OUTPUT_DIR}/test test.cpp Tensor.h
	${OUTPUT_DIR}/test
