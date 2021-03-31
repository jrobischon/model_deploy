## Model Deploy


**Docker Commands**<br>
Building image
```shell script
docker-compose up --build
```

Launch interactive Python session in container
```shell script
docker-compose run model_deploy
```

Running unit tests
```shell script
docker-compose run model_deploy pytest --cov=model_deploy model_deploy/tests/
```