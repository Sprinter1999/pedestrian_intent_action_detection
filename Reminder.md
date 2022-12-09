# Some often used commands



```python
python tools/train.py \
    --config_file configs/JAAD_intent.yaml \
    --gpu 0 \
    STYLE PIE \
    MODEL.TASK action_intent_single \
    MODEL.WITH_TRAFFIC True \
    SOLVER.INTENT_WEIGHT_MAX 1 \
    SOLVER.CENTER_STEP 800.0 \
    SOLVER.STEPS_LO_TO_HI 200.0 \
    SOLVER.MAX_ITERS 15000 \
    SOLVER.SCHEDULER none \
    DATASET.BALANCE False
```



```
docker run  -t -i --gpus all --shm-size 8G -v /home/xuefengj/jaad:/workspace/pedestrian_intent_action_detection -v /home/xuefengj/Downloads:/workspace/pedestrian_intent_action_detection/data ed6e0b2c6129
```



when constructing docker image:

```
sh-4.2# docker version
Client: Docker Engine - Community
 Version:           20.10.21
 API version:       1.41
 Go version:        go1.18.7
 Git commit:        baeda1f
 Built:             Tue Oct 25 18:04:24 2022
 OS/Arch:           linux/amd64
 Context:           default
 Experimental:      true
Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?
sh-4.2# sudo systemctl start docker
sh-4.2# docker version
Client: Docker Engine - Community
 Version:           20.10.21
 API version:       1.41
 Go version:        go1.18.7
 Git commit:        baeda1f
 Built:             Tue Oct 25 18:04:24 2022
 OS/Arch:           linux/amd64
 Context:           default
 Experimental:      true

Server: Docker Engine - Community
 Engine:
  Version:          20.10.21
  API version:      1.41 (minimum version 1.12)
  Go version:       go1.18.7
  Git commit:       3056208
  Built:            Tue Oct 25 18:02:38 2022
  OS/Arch:          linux/amd64
  Experimental:     false
 containerd:
  Version:          1.6.12
  GitCommit:        a05d175400b1145e5e6a735a6710579d181e7fb0
 runc:
  Version:          1.1.4
  GitCommit:        v1.1.4-0-g5fd4c4d
 docker-init:
  Version:          0.19.0
  GitCommit:        de40ad0
```

