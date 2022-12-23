#!/usr/bin/env bash

# adapted from @Soth and @peter-mortensen
# on https://stackoverflow.com/questions/2853803/how-to-echo-shell-commands-as-they-are-executed
# Function to display commands
exe() { printf '\n$ %s\n\n' "$*" ; eval "$*" ; }

. /opt/conda/etc/profile.d/conda.sh

for command in \
        'conda env create -n test-env-create -f `dirname -- '"${BASH_SOURCE[0]}"'`/sample_pytorch_environment.yml' \
        'conda activate test-env-create && conda info | grep active' \
        'TEST_ENV_CREATE=`conda list --export` && printf %s "$TEST_ENV_CREATE"' \
        'conda deactivate && conda env remove --name test-env-create'; do
    exe "$command";
done

for command in \
        'conda create -y -n test-create-install python=3.10' \
        'conda activate test-create-install && conda info | grep active' \
        'conda install -y pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia' \
        'TEST_CREATE_INSTALL=`conda list --export` && printf %s "$TEST_CREATE_INSTALL"' \
        'conda deactivate && conda env remove --name test-create-install'; do
    exe "$command";
done

if [[ "$TEST_ENV_CREATE" = "$TEST_CREATE_INSTALL" ]]; then
    printf '\nTest passed.'
else
    printf '\nTest failed.'
fi
