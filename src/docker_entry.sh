export CTAP_DATA=/data
export CTAP_RESULTS=$(pwd -P)/ctap_results
export NXF_HOME=$CTAP_RESULTS/nextflow_work
echo "CTAP version $(cat /app/VERSION)"
mkdir -p $NXF_HOME \
    && cd $NXF_HOME \
    && nextflow -c /app/main.cfg run /app/main.nf $@
