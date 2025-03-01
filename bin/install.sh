#!/usr/bin/bash

# install packages from requirements.txt
docker-compose exec --user root tensorflow pip install -r requirements.txt