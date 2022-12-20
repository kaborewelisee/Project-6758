#!/bin/bash

echo "TODO: fill in the docker run command"

docker run --env COMET_API_KEY=`echo $COMET_API_KEY` -p 9998:5000 ift6758/prediction_service