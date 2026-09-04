FROM mcr.microsoft.com/devcontainers/base:dev-ubuntu24.04

RUN sed -i 's|http://archive.ubuntu.com|http://ftp.riken.jp/Linux|g; s|http://security.ubuntu.com|http://ftp.riken.jp/Linux|g' /etc/apt/sources.list.d/ubuntu.sources 2>/dev/null || \
    sed -i 's|http://archive.ubuntu.com|http://ftp.riken.jp/Linux|g; s|http://security.ubuntu.com|http://ftp.riken.jp/Linux|g' /etc/apt/sources.list

RUN \
  --mount=type=cache,target=/var/lib/apt,sharing=locked \
  --mount=type=cache,target=/var/cache/apt,sharing=locked \
  apt-get update && apt-get install -y git-filter-repo libatomic1
