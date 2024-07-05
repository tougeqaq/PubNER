#!/bin/bash

python main.py --config ./config/BC2GM.json --stage 1
python main.py --config ./config/BC4CHEMD.json --stage 1
python main.py --config ./config/BC5CDR-chem.json --stage 1
python main.py --config ./config/BC5CDR-disease.json --stage 1
python main.py --config ./config/JNLPBA.json --stage 1
python main.py --config ./config/NCBI.json --stage 1