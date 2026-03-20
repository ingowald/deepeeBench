#!/bin/bash

benchData=/home/wald/deepeeData
models="bmw pp rungholt truck donotshare-e89-open donotshare-e89-closed donotshare-headlight donotshare-tokamak ls"
#models="bmw ls pp rungholt truck donotshare-e89-open donotshare-e89-closed donotshare-headlight donotshare-tokamak"
models="truck"

configs="cuBQL-float cuBQL-double owl-rtx owl-double-distance owl-double-triTest"
#shifts="0 100 10000 "
shifts="00 01 02 03 04 05 06 08"
#shifts="00 01 02 03 04 05 06 08 10"
shifts="00"
#mkdir experiments/build_cuBQL_double
#cmake -S . -B experiments/build_cuBQL_double
#cmake --build experiments/build_cuBQL_double


#mkdir experiments/build_cuBQL_float
#mkdir experiments/build_owl_

#cmake --con
mkdir experiments
for model in $models; do
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
	for ms in $shifts; do
	    cmd="experiments/$config/dpMakePrimaryRays\
		$benchData/$model.dpmini\
                --watchDog 120\
		`cat $benchData/$model.dpmini.view` \
		--native-scale `cat $benchData/$model.dpmini.ortho` \
		--shift $ms \
		-omf experiments/$config/$model-persp$ms.dpmini \
		-orf experiments/$config/$model-persp$ms.dprays \
		-ohf experiments/$config/$model-persp$ms.dphits \
		-oif experiments/$config/$model-persp$ms.ppm"
	    echo cmd is: $cmd
	    $cmd
	    
	    cmd="experiments/$config/dpBench\
		-imf experiments/$config/$model-persp$ms.dpmini \
		-irf experiments/$config/$model-persp$ms.dprays "
	    echo cmd is: $cmd
	    $cmd | tee experiments/$config/$model-ortho$ms.log
	done
	for os in $shifts; do
	    cmd="experiments/$config/dpMakePrimaryRays\
		$benchData/$model.dpmini\
		`cat $benchData/$model.dpmini.view` \
                --watchDog 120\
		--native-scale `cat $benchData/$model.dpmini.ortho` \
		--shift $os \
		--ortho \
		-omf experiments/$config/$model-ortho$os.dpmini \
		-orf experiments/$config/$model-ortho$os.dprays \
		-ohf experiments/$config/$model-ortho$os.dphits \
		-oif experiments/$config/$model-ortho$os.ppm"
	    echo cmd is: $cmd
	    $cmd
	    cmd="experiments/$config/dpBench\
		-imf experiments/$config/$model-ortho$os.dpmini \
		-irf experiments/$config/$model-ortho$os.dprays "
	    echo cmd is: $cmd
	    $cmd | tee experiments/$config/$model-ortho$os.log
	done
    done
done

