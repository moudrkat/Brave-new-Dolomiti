# How to run

Start a container:

```
docker-compose up -d
```

(optional) Install packages from requirements.txt (only first time setup):

```
./bin/install.sh
```

The container is up and running. Now run your script:

```
./bin/run.sh
```


portforwarding: 
`ssh -L 8501:localhost:8501 aorus`
