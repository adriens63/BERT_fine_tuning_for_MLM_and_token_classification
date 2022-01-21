## Install docker

Follow [the official guide](https://docs.docker.com/engine/install/).




## Pull the doccano image

```
docker pull doccano/doccano
```




## Create a container

```
docker container create --name doccano \
  -e "ADMIN_USERNAME=admin" \
  -e "ADMIN_EMAIL=admin@example.com" \
  -e "ADMIN_PASSWORD=password" \
  -p 8000:8000 doccano/doccano
```

like this   
```
docker container create --name doccano \
-e "ADMIN_USERNAME=adri" \
-e "ADMIN_EMAIL=serviere.adrien@gmail.com" \
-e "ADMIN_PASSWORD=password" \
-p 8096:8000 doccano/doccano 
```




## Start the container before labeling and stop it when you are done

To start :   
```
docker container start doccano
```

Then go to  
```
http://127.0.0.1:8000/
```
in your browser.   

To stop :   
```
docker container stop doccano -t 5
```
