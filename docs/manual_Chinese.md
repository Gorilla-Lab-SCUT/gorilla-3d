# gorilla3d 常用函数及代码介绍

`gorilla3d` 是一个基于 `gorilla` 的3d深度学习库，目前的主要任务在于提供一个平台给大家存放和规范代码。该基础库的文件目录如下：

```sh
gorilla3d
    ├── datasets
    ├── evaluation
    ├── losses
    ├── nn
    ├── ops
    ├── post
    ├── utils
    ├── __init__.py
    └── version.py
```

下面介绍一下代码库的结构以及相应的规范。

## ops
`ops` 目录用于存放相应的 `CUDA/C++` 拓展部分的代码，使其能够通过 `python setup.py` 能够顺利的安装，目前已有的为 `PointNet++` 中的：
```python
furthest_point_sample, ball_query, gather_points, grouping_operation, three_interpolate, three
```
以及较为常用的 `chamfer_distance`。

## nn
`nn` 模块目前分为两级 `models` 和更底层的 `modules`，当同学们写网络结构时希望大家遵循模块化的设计。
例如网络大致可以分为 `Backbone` 和 `Head` 以及其他的一些部分，这些部分能单独写出来就单独写出来，然后在 `models` 里面拼装在一起，尽量不要所有的细节都写入 `models` 中。

以 `PointNet++` 为例，如果要实现 classification，`Backbone` 就是 `PointSAModuleMSG`，然后 `Head` 部分可能就是非常简单的 mlp，那么在 `models` 里面就直接定义 `Backbone` 为 `PointNet2SASSG`，然后定义 mlp 拼接上去，这种非常轻量化或者另外一些别人基本用不到的就写入描述当前网络的 `models` 就可以了。
同理，利用 `PointNet++` 实现 part segmentation，则在 `modules` 中定义好 `PointNet2SASSG`，然后在 `models` 中调用并且拼接上 `mlp` 作为 `Head` 即可。

如果比较复杂的例如检测任务，设计 `RPN` 等结构的，尽量以 `module` 包装实现再调用。

## losses
<!-- TODO: fix here -->
`losses` 模块目前也是非常简单的，只有一个计算 `chamfer_distance` 的包装函数，希望大家将常用的函数也能包装进来。

## datasets
`datasets` 模块旨在存放大家的数据集，存放目录结构如下：
```python
datasets
    ├── name1
    │   ├── name1_task.py
    │   └── ...
    ├── name2
    │   └── ...
    ├── ...
    └── namen
        └── ...
```
大家以每个数据集为独立的个体，如果有不同的任务则在数据集名后以相应的 `task` 作为后缀存放。

## evaluation
`evaluation` 模块是基于 `gorilla.evaluation` 的实现模块，存放结构如下：
```python
evaluation
    ├── metric
    │   ├── name1
    │   │   ├── xxx.py
    │   │   └── ...
    │   ├── ...
    │   └── namen
    │       ├── xxx.py
    │       └── ...
    └── evaluator
        ├── name1.py
        ├── ...
        └── namen.py
```
其中 `metric` 中存放的基本都是计算指标或者读取数据集的 **功能函数**，`evaluator` 中存放的则是继承于 `gorilla.evaluation` 针对特定数据集的 `evaluator`。在写 `metric` 的时候希望同学们能够尽量保持函数的独立性和复用性。

## post
`post` 模块用于存放网络的后处理部分，部分同学可能用不到该模块。例如重建中浮块的去除以及检测任务中的解码和对齐部分，以及NMS。

## utils
`utils` 则用于放置一些专门处理3d数据的杂项函数，根据每个人的需求不同，相应的 `utils` 也不相同，需要同学们多贡献。


