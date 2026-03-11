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
    echo "======================================================="
    echo config $config
    echo "======================================================="
    flags=`cat config-${config}`
    echo flags $flags
    echo "======================================================="
    mkdir experiments/$config
    cmake -S . -B experiments/$config ${flags}
    cmake --build experiments/$config --parallel 32
    for model in $models; do
	experiments/$config/dpMakePrimaryRays\
	    $benchData/$model.dpmini\
	    `cat $benchData/$model.dpmini.view` \
	    -orf experiments/$config/$model.dprays \
	    -oif experiments/$config/$model.ppm 
    done
done

