#!/bin/bash
~/Documents/Projects/waffles/bin/waffles_plot scatter \
	blue $1 row 0 -radius 0.5 \
	red $1 row 1 -radius 0.5 -thickness 0 \
	green $1 row 2 -radius 0
