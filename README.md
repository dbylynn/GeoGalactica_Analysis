# GeoGlactica Analysis

This repo is prepare for someone who want to draw training curve, weight value hist and heat map.

## Training Curve

From tensorborad event log draw traing cure (including loss, learning rate, grad norm curve)

```shell
export tensorboard_path=""
python draw_training_curve --tensorboard_path ${tensorboard_path} 
```

## Weight Hist 

1. save model’s weight

   prepare `qkv/ out/ fc` weight for 

```shell
export layer=0
export lla_model_path=""
export gla_model_path=""
python save_weight_value.py --lla_model_path ${lla_model_path} --gla_model_path ${gla_model_path} --layer ${layer}
```



2. draw weigth hist

```shell
export weight_type="" # [qkv, out, fc]
python draw_hist.py --weight_type ${weight_type} --layer 0
```



## Heat Map
1. calculate the each categlory prompt similarty.
```shell
export prompt_path = ""
python cal_prompt_simliarty.py --prompt ${prompt_path}
```

2. draw heat map
```shell
export similarity_matrix="" # [qkv, out, fc]
python draw_heat_map.py --similarity_matrix ${similarity_matrix}
```


