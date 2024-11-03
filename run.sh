#!/bin/bash

python3 main.py 
uvicorn src.nlp_engineer_assignment.api:app --reload