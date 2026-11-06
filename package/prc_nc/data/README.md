# 数据来源与处理

## Dolly

来源：[Databricks Dolly 15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)，Databricks 员工编写的指令、上下文和回答，原数据许可 [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)。本目录的 Dolly 子集与归一化数据沿用该许可并保留来源。

修改：先用固定 seed 20260917 从 15,011 条中无放回抽取 512 条；将 context/response 归一化为 input/reference_response，保留类别与原行号。实跑时按照模型分词后的预先指定长度条件取前 count 条。参考回答不输入模型，也不用于选择胜负。

原文件 SHA256、抽样规则见同目录 `provenance.json`。这些是真实人工编写的任务内容，但不是生产请求到达轨迹；到达间隔由实验指定。

## 构造共享负载

两组固定教学上下文，加上不同问题编号，刻意产生可复用前缀。只用于检查共享页路径及压力边界，不估计自然负载中共享结构的频率。

所有输入的筛选均先于方法结果；三种策略共享同一分词后输入哈希、输出 cap、采样温度、到达间隔及页池容量。逐请求输出另行核对，不能仅由相同 cap 推断输出完全相同。
