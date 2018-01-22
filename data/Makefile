all: classify
classify: mnist.train.bin mnist.test.bin
clean:
	rm -rf *.bin

%.bin : %.bin.gz
	gzip -dc $< > $@

.PHONY: all classify
