docker run -d -p 27017:27017 --name mongodb mongo
docker run -d -p 9200:9200 -p 9300:9300 --name elasticsearch -e "discovery.type=single-node" elasticsearch:8.7.0
docker run -d -p 6333:6333 --name qdrant qdrant/qdrant