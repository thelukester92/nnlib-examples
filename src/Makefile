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

opt: airline mnist
dbg: airline_dbg mnist_dbg

clean:
	rm -rf $(BIN)/*

airline: $(BIN) $(BIN)/airline
$(BIN)/airline: airline.cpp
	$(CXX) $< $(CFLAGS_OPT) $(LFLAGS_OPT) -o $@

airline_dbg: $(BIN) $(BIN)/airline_dbg
$(BIN)/airline_dbg: airline.cpp
	$(CXX) $< $(CFLAGS_DBG) $(LFLAGS_DBG) -o $@

mnist: $(BIN) $(BIN)/mnist
$(BIN)/mnist: mnist.cpp
	$(CXX) $< $(CFLAGS_OPT) $(LFLAGS_OPT) -o $@

mnist_dbg: $(BIN) $(BIN)/mnist_dbg
$(BIN)/mnist_dbg: mnist.cpp
	$(CXX) $< $(CFLAGS_DBG) $(LFLAGS_DBG) -o $@

$(BIN):
	mkdir -p $@

.PHONY: opt dbg clean airline airline_dbg mnist mnist_dbg
