ARG CONDA_ENV=for_container

# https://mamba.readthedocs.io/en/latest/user_guide/mamba.html
FROM condaforge/miniforge3

# I believe this is to avoid permission issues with 
# manipulating added files to places like /opt
RUN old_umask=`umask` \
    && umask 0000 \
    && umask $old_umask

# scope var from global
ARG CONDA_ENV
COPY ./env.yml /opt/env.yml
# mamba install
RUN --mount=type=cache,target=/opt/conda/pkgs \
    mamba env create -n ${CONDA_ENV} --no-default-packages -f /opt/env.yml
ENV PATH /opt/conda/envs/${CONDA_ENV}/bin:/root/.local/bin:$PATH
# scgpt install
RUN pipx install poetry
COPY ./cache/scGPT /app/scGPT
RUN cd /app/scGPT \
    && poetry lock --no-update \
    && poetry install

# https://github.com/krallin/tini
COPY ./cache/tini /tini
RUN chmod +x /tini

# singularity uses tini, but doesn't use the -s flag, and that causes warnings.
# -g kills process group on ctrl+C
ENTRYPOINT ["/tini", "-s", "-g", "--"]
