## Model Deploy


**Docker Commands**<br>
Building image
```shell script
docker-compose up --build
```

Launch interactive Python session in container
```shell script
docker-compose run ml_api
```

Running unit tests
```shell script
docker-compose run ml_api pytest --cov=ml_api ml_api/tests/
```