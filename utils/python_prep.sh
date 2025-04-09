#! /usr/bin/bash

python3 -m venv env
source env/bin/activate
pip install --upgrade pip
ls
pip install -r utils/requirements.txt
