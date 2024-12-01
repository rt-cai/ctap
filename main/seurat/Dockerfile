ARG CONDA_ENV=for_container

# https://mamba.readthedocs.io/en/latest/user_guide/mamba.html
FROM satijalab/seurat:latest

# I believe this is to avoid permission issues with 
# manipulating added files to places like /opt
RUN old_umask=`umask` \
    && umask 0000 \
    && umask $old_umask

COPY ./load/* /opt/
RUN R < /opt/install.R

# # mamba install
# ARG CONDA_ENV
# COPY ./env.yml /opt/env.yml
# RUN --mount=type=cache,target=/opt/conda/pkgs \
#     mamba env create -n ${CONDA_ENV} --no-default-packages -f /opt/env.yml
# ENV PATH /opt/conda/envs/${CONDA_ENV}/bin:$PATH
# # tosica install
# COPY ./cache/TOSICA/TOSICA /app/TOSICA
# COPY ./src_edits/* /app/TOSICA/

# https://github.com/krallin/tini
# singularity uses tini, but doesn't use the -s flag, and that causes warnings.
# -g kills process group on ctrl+C
COPY ./cache/tini /tini
RUN chmod +x /tini
ENTRYPOINT ["/tini", "-s", "-g", "--"]
