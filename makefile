OUTPUT_DIR = ./bin
LIBS = -lcnpy
DEBUG_FLAGS = -g -O0 -D DEBUG
RUN_FLAGS = -O2

ifdef $DEBUG
	FLAGS = ${DEBUG_FLAGS}
else
	FLAGS = ${RUN_FLAGS}
endif

${OUTPUT_DIR}/main: main.cpp Tensor.tpp Tensor.h Convolutional.tpp Convolutional.h AvgPool.h AvgPool.tpp Linear.h Linear.tpp
	g++ ${LIBS} ${FLAGS} -o ${OUTPUT_DIR}/main main.cpp

run: ${OUTPUT_DIR}/main
	${OUTPUT_DIR}/main

valgrind: ${OUTPUT_DIR}/main
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes ${OUTPUT_DIR}/main

test: Tensor.h  Linear.h
	g++ ${LIBS} ${FLAGS} -o ${OUTPUT_DIR}/test test.cpp Tensor.h Linear.h
	${OUTPUT_DIR}/test
