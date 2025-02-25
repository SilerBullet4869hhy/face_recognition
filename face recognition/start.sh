#!/bin/bash
eval $(minikube docker-env)
docker build -t face-ws-app .
#minikube image load face-ws-app:latest
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
#一切正常后，执行此指令发布服务，要等1分钟左右
#kubectl port-forward service/my-face-ws-app 8800:8800
#执行此命令查看看板
#minikube dashboard