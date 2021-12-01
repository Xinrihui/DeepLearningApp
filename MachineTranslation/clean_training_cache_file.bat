@echo off
::后续命令使用的是：UTF-8编码
chcp 65001

:: 杀死 tensorboard 的进程
taskkill /F /im tensorboard.exe

:: 删除缓存的训练过程中持久化的模型
rmdir /s/q .\models\cache\
mkdir .\models\cache\


:: 删除日志
rmdir /s/q .\logs\train\
mkdir .\logs\train\

rmdir /s/q .\logs\validation\
mkdir .\logs\validation\

:: 开启 tensorboard
:: tensorboard --logdir ./logs

:: 查找进程  tasklist|findstr "tensorboard"






