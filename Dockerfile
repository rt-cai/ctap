ARG CONDA_ENV=for_container

# https://mamba.readthedocs.io/en/latest/user_guide/mamba.html
FROM condaforge/miniforge3
# scope var from global
ARG CONDA_ENV
COPY ./envs/base.yml /opt/env.yml
RUN --mount=type=cache,target=/opt/conda/pkgs \
    mamba env create -n ${CONDA_ENV} --no-default-packages -f /opt/env.yml

# I believe this is to avoid permission issues with 
# manipulating added files to places like /opt
RUN old_umask=`umask` \
    && umask 0000 \
    && umask $old_umask

# # use a smaller runtime image
# # jammy is ver. 22.04 LTS
# # https://wiki.ubuntu.com/Releases
# FROM ubuntu:jammy
# # scope var from global
# ARG CONDA_ENV
# COPY --from=build-env /opt/conda/envs/${CONDA_ENV} /opt/conda/envs/${CONDA_ENV}
ENV PATH /opt/conda/envs/${CONDA_ENV}/bin:$PATH

ADD ./src/${PACKAGE} /app/${PACKAGE}
RUN echo "python -m ${PACKAGE} \$@" >/usr/bin/${PACKAGE} && chmod +x /usr/bin/${PACKAGE}
ENV PYTHONPATH /app:$PYTHONPATH

# Singularity uses tini, but raises warnings
# we set it up here correctly for singularity
ADD ./lib/tini /tini
RUN chmod +x /tini
    
# singularity doesn't use the -s flag, and that causes warnings.
# -g kills process group on ctrl+C
ENTRYPOINT ["/tini", "-s", "-g", "--"]
