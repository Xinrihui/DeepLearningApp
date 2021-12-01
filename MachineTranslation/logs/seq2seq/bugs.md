
## 使用 tf.function 包装器出现的问题

### 1. Error in PredictCost() for the op: op: "Softmax" attr

    原文件: nmt_seq2seq_v2_timestep_xrh.py
    
    日志信息:
    
    2021-11-29 10:44:59.596496: W tensorflow/core/grappler/costs/op_level_cost_estimator.cc:689] 

    Error in PredictCost() for the op: op: "Softmax" attr { key: "T" value { type: DT_FLOAT } } 
    inputs { dtype: DT_FLOAT shape { unknown_rank: true } } 
    device { type: "GPU" vendor: "NVIDIA" model: "GeForce GTX 1050" frequency: 1493 num_cores: 5 environment 
    { key: "architecture" value: "6.1" }
     environment { key: "cuda" value: "11020" } 
     environment { key: "cudnn" value: "8100" } 
     num_registers: 65536 l1_cache_size: 24576 l2_cache_size: 524288 
     shared_memory_size_per_multiprocessor: 98304 memory_size: 1385211086 bandwidth: 112128000 } 
     outputs { dtype: DT_FLOAT shape { unknown_rank: true } }
    
    15it [00:52,  3.31s/it]2021-11-29 10:45:02.878455: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:812] 
    implementation_selector failed: INVALID_ARGUMENT: Invalid format of input node name:  Expected: {forward_node_name}:{index}
    
    原因分析:
    
    https://github.com/tensorflow/tensorflow/issues/50575