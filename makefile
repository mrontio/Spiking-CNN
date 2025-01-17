OUTPUT_DIR = ./bin
LIBS = -lcnpy
DEBUG_FLAGS = -g -O0 -D DEBUG
RUN_FLAGS = -O2
FLAGS = ${DEBUG_FLAGS}
RUN_FILE = ${OUTPUT_DIR}/run-output.txt
# ifdef $DEBUG
# 	FLAGS = ${DEBUG_FLAGS}
# else
# 	FLAGS = ${RUN_FLAGS}
# endif

${OUTPUT_DIR}/main: main.cpp Tensor.tpp Tensor.h Convolutional.tpp Convolutional.h AvgPool.h AvgPool.tpp Linear.h Linear.tpp IntegrateFire.h IntegrateFire.tpp
	g++ ${LIBS} ${FLAGS} -o ${OUTPUT_DIR}/main main.cpp

run: ${OUTPUT_DIR}/main
	${OUTPUT_DIR}/main

run-bg: ${OUTPUT_DIR}/main
	${OUTPUT_DIR}/main >${RUN_FILE} &
	echo "Output being written to ${RUN_FILE}"

valgrind-main: ${OUTPUT_DIR}/main
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes ${OUTPUT_DIR}/main

valgrind-test: test
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes ${OUTPUT_DIR}/test

test:
	g++ ${LIBS} ${FLAGS} -o ${OUTPUT_DIR}/test test.cpp
	${OUTPUT_DIR}/test
