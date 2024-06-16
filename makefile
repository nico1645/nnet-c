CC = clang
CFLAGS = -Wall -Wextra -std=c11 -O3

BIN_DIR = bin
HEADER_DIR = include

BIN_TARGET = $(BIN_DIR)/lights_out
RM = rm

.PHONY: all build test clean

all: build test

build: 
	mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -I$(HEADER_DIR) -c lib/ndimarr.c -o lib/ndimarr.o 
	ar rcs lib/libndimarr.a lib/ndimarr.o
	$(CC) $(CFLAGS) -I$(HEADER_DIR) -c lib/nnetfunc.c -o lib/nnetfunc.o
	ar rcs lib/libnnetfunc.a lib/nnetfunc.o
	$(CC) $(CFLAGS) -I$(HEADER_DIR) -c lib/nnetmodels.c -o lib/nnetmodels.o
	ar rcs lib/libnnetmodels.a lib/nnetmodels.o
	$(CC) $(CFLAGS) -g src/main.c -Llib/ -lndimarr -lnnetfunc -lnnetmodels -o $(BIN_DIR)/main	
	$(BIN_DIR)/main

test:
	mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -I$(HEADER_DIR) -c lib/ndimarr.c -o lib/ndimarr.o 
	ar rcs lib/libndimarr.a lib/ndimarr.o
	$(CC) $(CFLAGS) -I$(HEADER_DIR) -c lib/nnetfunc.c -o lib/nnetfunc.o
	ar rcs lib/libnnetfunc.a lib/nnetfunc.o
	$(CC) $(CFLAGS) -I$(HEADER_DIR) -c lib/nnetmodels.c -o lib/nnetmodels.o
	ar rcs lib/libnnetmodels.a lib/nnetmodels.o
	$(CC) $(CFLAGS) -g test/ndimarr.c -Llib/ -lndimarr -lnnetfunc -lnnetmodels -o $(BIN_DIR)/run_tests
	$(BIN_DIR)/run_tests

clean:
	$(RM) -r $(BIN_DIR)	
	$(RM) -r lib/*.a
	$(RM) -r lib/*.o



