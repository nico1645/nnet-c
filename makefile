CC = clang
CFLAGS = -Wall -Wextra -std=c11

BIN_DIR = bin
HEADER_DIR = include

BIN_TARGET = $(BIN_DIR)/lights_out
RM = rm

.PHONY: all build test clean

all: build test

build: 
	mkdir -p $(BIN_DIR)

test:
	mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -I$(HEADER_DIR)/ndimarr.h -c lib/ndimarr.c -o lib/ndimarr.o
	ar rcs lib/libndimarr.a lib/ndimarr.o
	$(CC) $(CFLAGS) -I$(HEADER_DIR)/nnetfunc.h -c lib/nnetfunc.c -o lib/nnetfunc.o
	ar rcs lib/libnnetfunc.a lib/nnetfunc.o
	$(CC) $(CFLAGS) test/ndimarr.c -Llib/ -lndimarr -lnnetfunc -o $(BIN_DIR)/run_tests	
	$(BIN_DIR)/run_tests

clean:
	$(RM) -r $(BIN_DIR)	
	$(RM) -r lib/*.a
	$(RM) -r lib/*.o



