#!/bin/bash

benchData=/home/wald/deepeeData
models="bmw ls pp rungholt truck"

configs="cuBQL-double cuBQL-float owl-rtx owl-double-distance owl-double-triTest"
#mkdir experiments/build_cuBQL_double
#cmake -S . -B experiments/build_cuBQL_double
#cmake --build experiments/build_cuBQL_double


#mkdir experiments/build_cuBQL_float
#mkdir experiments/build_owl_

#cmake --con
mkdir experiments
for config in $configs; do
    flags=`cat config-${config}`
    for f in $models; do
	mkdir experiments/$config
	cmake -S . -B experiments/$config ${flags}
	cmake --build experiments/$config
    done
done

