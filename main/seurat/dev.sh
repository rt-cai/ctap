HERE=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DOCKER_IMAGE=seurat_test
VER=test

case $1 in
    -e) # test env
        env_name=$DOCKER_IMAGE
        echo y | mamba env remove -n $env_name
        mamba env create -n $env_name -f $HERE/env.yml
    ;;
    -b) # docker
        # pre-download requirements
        mkdir -p $HERE/cache
        cd $HERE/cache
        TINI_VERSION=v0.19.0
        ! [ -f tini ] && wget https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini
        cd $HERE

        # build the docker container locally
        export DOCKER_BUILDKIT=1
        docker build \
            --build-arg="CONDA_ENV=${DOCKER_IMAGE}" \
            -t $DOCKER_IMAGE:$VER .
    ;;
    -bs) # apptainer image *from docker*
        apptainer build ./cache/$DOCKER_IMAGE.sif docker-daemon://$DOCKER_IMAGE:$VER
    ;;
    -r) # docker
        shift
        docker run -it --rm \
            -u $(id -u):$(id -g) \
            --mount type=bind,source="$HERE",target="/ws"\
            --workdir="/ws" \
            $DOCKER_IMAGE:$VER /bin/bash
    ;;
    *)
        echo "arg missing or unrecognized"
        echo "example: ./dev.sh -d"
    ;;
esac