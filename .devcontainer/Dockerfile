FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

# Remove the default `ubuntu` user (UID 1000) so that common-utils can
# create the `vscode` user at UID 1000 to match the host user.
RUN userdel -r ubuntu 2>/dev/null || true

RUN sed -i 's|http://archive.ubuntu.com|http://ftp.riken.jp/Linux|g; s|http://security.ubuntu.com|http://ftp.riken.jp/Linux|g' /etc/apt/sources.list.d/ubuntu.sources 2>/dev/null || \
    sed -i 's|http://archive.ubuntu.com|http://ftp.riken.jp/Linux|g; s|http://security.ubuntu.com|http://ftp.riken.jp/Linux|g' /etc/apt/sources.list

RUN \
  --mount=type=cache,target=/var/lib/apt,sharing=locked \
  --mount=type=cache,target=/var/cache/apt,sharing=locked \
  apt-get update && apt-get install -y git-filter-repo
