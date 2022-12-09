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

