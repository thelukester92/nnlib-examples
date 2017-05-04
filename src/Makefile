BIN:=../bin
CFLAGS:=-Wall -std=c++11
LFLAGS:=
CXX:=g++
UNAME:=$(shell uname -s)

ifeq ($(UNAME),Darwin)
	LFLAGS+=-framework Accelerate
else
	LFLAGS+=-L/usr/local/lib -lopenblas -lpthread
endif

CFLAGS_OPT:=$(CFLAGS) -O3 -DOPTIMIZE
CFLAGS_DBG:=$(CFLAGS) -g -DDEBUG

LFLAGS_OPT:=$(LFLAGS)
LFLAGS_DBG:=$(LFLAGS)

DBG_OUT:=$(OUT)_dbg

mnist: $(BIN) $(BIN)/mnist
$(BIN)/mnist: mnist.cpp
	$(CXX) mnist.cpp $(CFLAGS_OPT) $(LFLAGS_OPT) -o $@

mnist_dbg: $(BIN) $(BIN)/mnist_dbg
$(BIN)/mnist_dbg: mnist.cpp
	$(CXX) mnist.cpp $(CFLAGS_DBG) $(LFLAGS_DBG) -o $@

clean:
	rm -rf $(BIN)/*

$(BIN):
	mkdir -p $@

.PHONY: clean mnist mnist_dbg
