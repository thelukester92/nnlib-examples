# all   - make all examples
# clean - remove build artifacts

NNLIB    := nnlib
CXXFLAGS := -Wall

override CXXFLAGS += -std=c++11
override OPTFLAGS := $(CXXFLAGS) -O3
override DBGFLAGS := $(CXXFLAGS) -O0 -g
override DEPFILES := $(shell find src -name "*.cpp")
override DEPFILES := $(DEPFILES:src/%.cpp=obj/%.d) $(DEPFILES:src/%.cpp=obj/dbg/%.d)
override APPS     := classify timeseries
override APPS     := $(APPS:%=bin/%) $(APPS:%=bin/%_dbg)

all: $(APPS)
clean:
	@$(MAKE) -C data clean
	rm -rf bin obj

bin/%: obj/%.o
	@$(MAKE) -C data $(basename $(notdir $@))
	@mkdir -p $(dir $@)
	$(CXX) $< $(OPTFLAGS) $(LDFLAGS) -l$(NNLIB) -MMD -o $@

bin/%_dbg: obj/dbg/%.o
	@mkdir -p $(dir $@)
	$(CXX) $< $(DBGFLAGS) $(LDFLAGS) -l$(NNLIB)_dbg -MMD -o $@

obj/%.o: src/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $< $(OPTFLAGS) -c -MMD -o $@

obj/dbg/%.o: src/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $< $(DBGFLAGS) -c -MMD -o $@

.PHONY: all opt dbg clean

-include $(DEPFILES)
