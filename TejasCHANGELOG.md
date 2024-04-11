save_model_weights.py : line 20 : model_file = model_cfg.get_model_default_weights_file(model_name) TO model_file = str(model_cfg.get_model_default_weights_file(model_name))

model_cfg.py : added line 101 - 105

evaluation.py : line 173 : /project/jpwalter_148/hnwang/datasets/ImageNet/ TO /home/tejas/PipeEdge