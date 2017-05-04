GZIP_FILES=$(wildcard *.gz)
ARFF_FILES=$(GZIP_FILES:%.arff.gz=%.arff)

decompress: $(ARFF_FILES)

%.arff : %.arff.gz
	gzip -dc $< > $@
