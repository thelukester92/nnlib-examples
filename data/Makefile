GZIP_FILES=$(wildcard *.gz)
CSV_FILES=$(GZIP_FILES:%.csv.gz=%.csv)

decompress: $(CSV_FILES)

%.csv : %.csv.gz
	gzip -dc $< > $@
