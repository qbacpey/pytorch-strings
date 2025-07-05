## TPC-H 导出数据

现在数据的来源有两个CSV文件，一个是`string_processing`函数打印出来的数据，另一个是Pytest-Benchmark的输出数据。需要将这两个文件按照`fullname`字段进行合并，并将结果保存到一个新的CSV文件中。
+ `string_processing`：
  + 输出`tuple_count,query_result_size,tuple_element_size_bytes,total_size_bytes`等字段。
  + 文件名为`benchmark_info_from_string_processing.csv`。
+ Pytest-Benchmark：
  + 输出`fullname,mean,min,max,stddev,ops,rounds`等字段。
  + 首先需要将数据保存为JSON格式，然后再转换为CSV格式。
    + JSON：`srun --ntasks=1 --cpus-per-task=16 --gres=gpu:2 --pty pytest -k tpc -s --benchmark-storage='file:///home/qchen/10_DMLab/pytorch-strings/ploting/raw_data/json/Linux-CPython-3.11-64bit/' --benchmark-autosave`
    + CSV：`srun --ntasks=1 --cpus-per-task=16 --gres=gpu:2 --pty py.test-benchmark compare '/home/qchen/10_DMLab/pytorch-strings/ploting/raw_data/json/Linux-CPython-3.11-64bit/*.json'  --csv='/home/qchen/10_DMLab/pytorch-strings/ploting/raw_data/csv/benchmark_info_from_pytest_benchmark'`二次处理之后才能获得CSV文件
  + 文件名为`benchmark_info_from_pytest_benchmark.csv`。

### 合并数据

先后调用`ploting/scripts/10_Post_Benchmark_Process/00_process_benchmark.py`与`ploting/scripts/10_Post_Benchmark_Process/01_Join_By_Name.py`脚本来处理数据。

```
python 00_process_benchmark.py && python 01_Join_By_Name.py

```bash