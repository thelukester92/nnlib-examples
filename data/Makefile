all: classify
classify: mnist.train.csv mnist.test.csv
clean:
	rm -rf *.bin

%.bin : %.bin.gz
	gzip -dc $< > $@

.PHONY: all classify
