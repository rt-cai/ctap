ARG CONDA_ENV=for_container

# https://mamba.readthedocs.io/en/latest/user_guide/mamba.html
FROM condaforge/miniforge3
# scope var from global
ARG CONDA_ENV

# I believe this is to avoid permission issues with 
# manipulating added files to places like /opt
RUN old_umask=`umask` \
    && umask 0000 \
    && umask $old_umask

# software dependencies with mamba
COPY ./envs/base.yml /opt/env.yml
RUN --mount=type=cache,target=/opt/conda/pkgs \
    mamba env create -n ${CONDA_ENV} --no-default-packages -f /opt/env.yml
ENV PATH /opt/conda/envs/${CONDA_ENV}/bin:/app/:$PATH

# data dependencies
ENV CTAP_DATA /data
COPY ./data/GSE267869_Processed_data_files.tar.gz /data/
COPY ./data/PanglaoDB.csv /data/
COPY ./scratch/results /example_results
COPY /src /app

# entrypoints
ARG VERSION
RUN echo $VERSION >/app/VERSION
RUN echo "cp -r /example_results ./" >/usr/bin/example && chmod +x /usr/bin/example
RUN echo "/app/docker_entry.sh \$@" >/usr/bin/ctap && chmod +x /usr/bin/ctap

# Singularity uses tini, but raises warnings
# we set it up here correctly for singularity
ADD ./lib/tini /tini
RUN chmod +x /tini
    
# singularity doesn't use the -s flag, and that causes warnings.
# -g kills process group on ctrl+C
ENTRYPOINT ["/tini", "-s", "-g", "--"]
