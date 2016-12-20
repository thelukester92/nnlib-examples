BIN:=../bin
CFLAGS:=-Wall -std=c++11
LFLAGS:=
CXX:=g++
UNAME:=$(shell uname -s)

ifeq ($(UNAME),Darwin)
	LFLAGS+=-framework Accelerate
	CFLAGS+=-DAPPLE
else
	LFLAGS+=-lblas -llapack
endif

CFLAGS_OPT:=$(CFLAGS) -O3 -DOPTIMIZE
CFLAGS_DBG:=$(CFLAGS) -g -DDEBUG

LFLAGS_OPT:=$(LFLAGS)
LFLAGS_DBG:=$(LFLAGS)

DBG_OUT:=$(OUT)_dbg

mnist: $(BIN) $(BIN)/mnist
$(BIN)/mnist: mnist.cpp
	$(CXX) mnist.cpp $(CFLAGS_OPT) $(LFLAGS_OPT) -o $@

clean:
	rm -rf $(BIN)/*

$(BIN):
	mkdir -p $@

.PHONY: clean mnist
