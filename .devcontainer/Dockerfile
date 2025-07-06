# Use an official Python image as the base
FROM nvcr.io/nvidia/pytorch:23.08-py3

RUN useradd -m vscode
WORKDIR /workspaces/mercInterview

RUN chown -R vscode:vscode /workspaces/mercInterview

RUN pip install virtualenv

USER vscode