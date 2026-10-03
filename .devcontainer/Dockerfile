FROM mcr.microsoft.com/devcontainers/python:1-3.11-bookworm

# The base image ships a Yarn apt source whose signing key no longer verifies, which makes
# `apt-get update` (and therefore the docker-in-docker feature) fail. Node comes from the
# node feature, so the Yarn source is not needed.
RUN rm -f /etc/apt/sources.list.d/yarn.list
