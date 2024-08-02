OUTPUT_DIR = ./bin
LIBS = -lcnpy
${OUTPUT_DIR}/main: main.cpp
	g++ ${LIBS} -g -o ${OUTPUT_DIR}/main main.cpp

run: ${OUTPUT_DIR}/main
	${OUTPUT_DIR}/main
