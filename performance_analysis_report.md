# CS 533 项目3：霍普金斯统计量时间性能分析报告

## 1. 实验环境

- **系统规格**：
  - 操作系统：[将替换为实际系统]
  - 处理器：[将替换为实际处理器]
  - Python版本：[将替换为实际版本]

## 2. 算法主要函数分析

### 2.1 主要函数及其操作

| 函数名 | 功能描述 | 操作类型 |
|--------|----------|----------|
| hopkins_stat | 计算霍普金斯统计量 | 随机采样、临近点查找、距离计算 |
| nearest_neighbor | 寻找最近邻点 | 网格搜索、距离计算 |
| get_neighbors | 获取邻居点 | 网格访问、数组操作 |
| distance | 计算欧氏距离 | 数学计算 |
| Grid.__init__ | 初始化网格结构 | 数据结构创建、数组分配 |

### 2.2 主要计算成本

根据性能分析器结果，霍普金斯统计量算法的主要计算成本在于：

1. **最近邻点查找**：在nearest_neighbor函数中进行
2. **随机点生成**：在hopkins_stat函数开始部分
3. **距离计算**：在distance函数中多次进行

## 3. 理论时间复杂度分析

### 3.1 单次霍普金斯统计量计算

霍普金斯统计量算法的时间复杂度受以下因素影响：
- n：数据集中的点数
- m：随机选择的点数
- 网格查找效率

理论上，时间复杂度范围为：
- 最佳情况：O(m)
- 最坏情况：O(m × n)

### 3.2 Bootstrap过程

Bootstrap过程的时间复杂度取决于：
- B：bootstrap重复次数
- m：随机选择的点数
- n：数据集大小

因此总体复杂度范围：
- 最佳情况：O(B × m)
- 最坏情况：O(B × m × n)

## 4. 实验设计

### 4.1 变化参数

我们将测试以下参数变化对性能的影响：

1. **B值变化**：保持n和m不变，改变B (1000, 3000, 5000, 7000, 9000, 11000)
2. **m值变化**：保持n和B不变，改变m (100, 300, 500, 700, 900, 1100)
3. **n值变化**：保持m和B不变，改变n (500, 1000, 2000, 3000, 4000, 5000)

### 4.2 系统负载测试

- 轻负载：在系统空闲状态下运行
- 重负载：同时运行其他计算密集型任务

### 4.3 输出与记录

对每组实验，我们记录：
- CPU时间
- 墙钟时间(wall time)
- 各函数调用次数和耗时

## 5. 实验结果与分析

### 5.1 性能分析器输出

#### 5.1.1 函数调用频率
[将替换为实际结果]

#### 5.1.2 时间消耗分布
[将替换为实际结果]

### 5.2 参数影响分析

#### 5.2.1 B值对性能的影响
[将替换为实际结果]

#### 5.2.2 m值对性能的影响
[将替换为实际结果]

#### 5.2.3 n值对性能的影响
[将替换为实际结果]

### 5.3 实测时间复杂度

基于实验数据，我们通过线性回归分析得出：
[将替换为实际结果]

### 5.4 系统负载影响

在不同负载下的性能比较：
[将替换为实际结果]

## 6. 结论

[将替换为基于实验结果的结论]

## 7. 参考文献

1. 项目2文档与代码
2. McGeoch, Catherine C. A guide to experimental algorithmics. Cambridge University Press, 2012.

---

# CS 533 Project 3: Hopkins Statistic Time Performance Analysis Report

## 1. Experimental Environment

- **System Specifications**:
  - Operating System: [To be replaced with actual system]
  - Processor: [To be replaced with actual processor]
  - Python Version: [To be replaced with actual version]

## 2. Algorithm Function Analysis

### 2.1 Main Functions and Operations

| Function Name | Purpose | Operation Type |
|--------------|---------|----------------|
| hopkins_stat | Calculate Hopkins statistic | Random sampling, nearest neighbor search, distance calculation |
| nearest_neighbor | Find nearest neighbor point | Grid search, distance calculation |
| get_neighbors | Get neighboring points | Grid access, array operations |
| distance | Calculate Euclidean distance | Mathematical calculation |
| Grid.__init__ | Initialize grid structure | Data structure creation, array allocation |

### 2.2 Primary Computational Costs

Based on profiler results, the main computational costs in the Hopkins statistic algorithm are:

1. **Nearest neighbor search**: Performed in the nearest_neighbor function
2. **Random point generation**: At the beginning of the hopkins_stat function
3. **Distance calculations**: Repeatedly performed in the distance function

## 3. Theoretical Time Complexity Analysis

### 3.1 Single Hopkins Statistic Calculation

The time complexity of the Hopkins statistic algorithm is influenced by:
- n: Number of points in the dataset
- m: Number of randomly selected points
- Grid lookup efficiency

Theoretically, the time complexity range is:
- Best case: O(m)
- Worst case: O(m × n)

### 3.2 Bootstrap Process

The time complexity of the bootstrap process depends on:
- B: Number of bootstrap repetitions
- m: Number of randomly selected points
- n: Dataset size

Therefore, the overall complexity range is:
- Best case: O(B × m)
- Worst case: O(B × m × n)

## 4. Experimental Design

### 4.1 Varying Parameters

We will test the performance impact of varying the following parameters:

1. **B-value variation**: Keep n and m constant, vary B (1000, 3000, 5000, 7000, 9000, 11000)
2. **m-value variation**: Keep n and B constant, vary m (100, 300, 500, 700, 900, 1100)
3. **n-value variation**: Keep m and B constant, vary n (500, 1000, 2000, 3000, 4000, 5000)

### 4.2 System Load Testing

- Light load: Running in an idle system state
- Heavy load: Running simultaneously with other computationally intensive tasks

### 4.3 Output and Recording

For each experiment set, we record:
- CPU time
- Wall clock time
- Function call counts and time spent

## 5. Experimental Results and Analysis

### 5.1 Profiler Output

#### 5.1.1 Function Call Frequency
[To be replaced with actual results]

#### 5.1.2 Time Consumption Distribution
[To be replaced with actual results]

### 5.2 Parameter Impact Analysis

#### 5.2.1 Impact of B-value on Performance
[To be replaced with actual results]

#### 5.2.2 Impact of m-value on Performance
[To be replaced with actual results]

#### 5.2.3 Impact of n-value on Performance
[To be replaced with actual results]

### 5.3 Measured Time Complexity

Based on experimental data, we derive through linear regression analysis:
[To be replaced with actual results]

### 5.4 System Load Impact

Performance comparison under different loads:
[To be replaced with actual results]

## 6. Conclusion

[To be replaced with conclusions based on experimental results]

## 7. References

1. Project 2 Documentation and Code
2. McGeoch, Catherine C. A guide to experimental algorithmics. Cambridge University Press, 2012. 