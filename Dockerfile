# Minimal, standalone environment for the TBR trajectory benchmarks
# Includes dependencies for every method in methods.py
FROM python:3.12.13-bookworm

# ── System libraries ──────────────────────────────────────────────────────
# build-essential/gfortran: compiling scientific Python deps (scipy, etc.)
# libcurl/libssl/libxml2: required by R + Bioconductor package installation
RUN apt-get update && \
    apt-get install -y \
        build-essential gfortran curl gnupg \
        libcurl4-openssl-dev libssl-dev libxml2-dev && \
    rm -rf /var/lib/apt/lists/*

# Debian bookworm's stock r-base (4.2.2) is too old for rpy2 — its libR.so is
# missing symbols (e.g. R_getVar) that rpy2's cffi bindings require. Pull R
# 4.4+ from CRAN's own apt repo instead.
RUN mkdir -p /etc/apt/keyrings && \
    curl -fsSL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x95C0FAF38DB3CCAD0C080A7BDC78B2DDEABC47B7" \
        | gpg --dearmor -o /etc/apt/keyrings/r-project.gpg && \
    echo "deb [signed-by=/etc/apt/keyrings/r-project.gpg] https://cloud.r-project.org/bin/linux/debian bookworm-cran40/" \
        > /etc/apt/sources.list.d/r-project.list && \
    apt-get update && \
    apt-get install -y r-base r-base-dev && \
    rm -rf /var/lib/apt/lists/*

# ── R + Bioconductor edgeR (required by pertpy's Milo solver="edger") ────
RUN R -q -e 'install.packages("BiocManager", repos="https://cloud.r-project.org")' && \
    R -q -e 'BiocManager::install("edgeR")'

WORKDIR /benchmarks

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir rpy2

COPY . .

CMD ["bash", "run.sh"]
