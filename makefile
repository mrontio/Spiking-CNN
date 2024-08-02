OUTPUT_DIR = ./bin
LIBS = -lncpy
main: main.c
	gcc ${LIBS} -g -o ${OUTPUT_DIR}/main main.c
run: main
	./main
