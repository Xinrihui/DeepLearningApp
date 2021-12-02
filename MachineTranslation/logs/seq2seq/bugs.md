
## 使用 tf.function 包装器出现的问题

### 1. Error in PredictCost() for the op: op: "Softmax" attr

    程序文件: nmt_seq2seq_v2_timestep_xrh.py
    
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
    
    
## 使用 tf.range 出现的问题

### 1. function_optimizer failed: INVALID_ARGUMENT

    程序文件: seq2seq_xrh.py
    
    日志信息:
    
    2021-12-02 14:16:37.946191: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:812] function_optimizer failed: INVALID_ARGUMENT: Input 7 of node gradient_tape/model/trian_decoder/while/model/trian_decoder/while_grad/body/_301/gradient_tape/model/trian_decoder/while/gradients/model/trian_decoder/while/lstm_1/PartitionedCall_grad/PartitionedCall was passed variant from gradient_tape/model/trian_decoder/while/model/trian_decoder/while_grad/body/_301/gradient_tape/model/trian_decoder/while/gradients/model/trian_decoder/while/lstm_1/PartitionedCall_grad/TensorListPopBack_2:1 incompatible with expected float.
    2021-12-02 14:16:38.020325: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:812] shape_optimizer failed: OUT_OF_RANGE: src_output = 30, but num_outputs is only 30
    2021-12-02 14:16:38.057480: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:812] layout failed: OUT_OF_RANGE: src_output = 30, but num_outputs is only 30
    2021-12-02 14:16:38.148917: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:812] tfg_optimizer{} failed: INVALID_ARGUMENT: Input 7 of node gradient_tape/model/trian_decoder/while/model/trian_decoder/while_grad/body/_301/gradient_tape/model/trian_decoder/while/gradients/model/trian_decoder/while/lstm_1/PartitionedCall_grad/PartitionedCall was passed variant from gradient_tape/model/trian_decoder/while/model/trian_decoder/while_grad/body/_301/gradient_tape/model/trian_decoder/while/gradients/model/trian_decoder/while/lstm_1/PartitionedCall_grad/TensorListPopBack_2:1 incompatible with expected float.
        when importing GraphDef to MLIR module in GrapplerHook