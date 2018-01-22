all:        classify timeseries
classify:   mnist.train.bin mnist.test.bin
timeseries: airline.bin
clean:
	rm -rf *.bin

%.bin : %.bin.gz
	gzip -dc $< > $@

.PHONY: all classify timeseries
