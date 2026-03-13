#!/bin/bash

benchData=/home/wald/deepeeData
models="bmw ls pp rungholt truck donotshare-e89-open donotshare-e89-closed donotshare-headlight donotshare-tokamak"
#models="truck"

configs="cuBQL-double cuBQL-float owl-rtx owl-double-distance owl-double-triTest"
#shifts="0 100 10000 "
shifts="0 100 10000 1000000 100000000"
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
    CMAKE_PREFIX_PATH=/home/wald/opt/ cmake -S . -B experiments/$config ${flags}
    cmake --build experiments/$config --parallel 32
    for model in $models; do
	for ms in $shifts; do
	    experiments/$config/dpMakePrimaryRays\
		$benchData/$model.dpmini\
		`cat $benchData/$model.dpmini.view` \
		--shift $ms \
		-orf experiments/$config/$model-persp$ms.dprays \
		-ohf experiments/$config/$model-persp$ms.dphits \
		-oif experiments/$config/$model-persp$ms.ppm 
	done
	for os in $shifts; do
	    experiments/$config/dpMakePrimaryRays\
		$benchData/$model.dpmini\
		`cat $benchData/$model.dpmini.view` \
		--ortho `cat $benchData/$model.dpmini.ortho` \
		--shift $os \
		-orf experiments/$config/$model-ortho$os.dprays \
		-ohf experiments/$config/$model-ortho$os.dphits \
		-oif experiments/$config/$model-ortho$os.ppm 
	done
    done
done

