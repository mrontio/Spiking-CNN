OUTPUT_DIR = ./bin
LIBS = -lcnpy
FLAGS = -g -O1

${OUTPUT_DIR}/main: main.cpp Tensor4D.tpp Tensor4D.h Convolutional.tpp Convolutional.h
	g++ ${LIBS} ${FLAGS} -o ${OUTPUT_DIR}/main main.cpp

run: ${OUTPUT_DIR}/main
	${OUTPUT_DIR}/main
