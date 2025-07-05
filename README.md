# Benchmarking String Encodings in PyTorch

Plan to support Equal, Range, and Prefix queries.

## Dictionary Encoding

### Column-wise dictionary encoding

Match in a character column-wise manner

O (mn) complexity, but m would gradually decrease as the matching progresses

Query support:
- [x] Equal query: O (mn) complexity
- [x] Range query: O (mn) complexity
- [x] Prefix query: O (mn) complexity

### Row-wise dictionary encoding

Match in a Binary searching manner

O (log n) complexity, where n is the size of the dictionary

Query support:
- [ ] Equal query: O (log n) complexity
- [ ] Range query: O (log n) complexity
- [ ] Prefix query

## Commands

```bash
# Only run the mssb data set
srun --ntasks=1 --cpus-per-task=16 --gres=gpu:2 --pty pytest -k mssb -s
```


+ GPU：
  + 适合：多次查询，用内存带宽计算
  + NVIDIA A100-SXM4-40GB 
  + PCIe 4.0 x16：31.508 GB/s
  + 内存带宽：1.6TB/s
  + 多次查询，用内存带宽计算；单次查询，用PCIe带宽计算
+ CPU：1     1
  + 适合：单次查询
  + 内存带宽：58 GB/s to 65 GB/s

从算法上来讲，只有变动数据量是有意义的。变化选择度不会使得计算量发生变化

Chunk 只有在数据非常大的时候才有用，比如说现在是CPU数据量恰好为GPU最大内存的两倍

D select count(1) from lineitem;
┌────────────────┐
│    count(1)    │
│     int64      │
├────────────────┤
│    6001215     │
│ (6.00 million, 6 * 1e6) -> 
│ (sf=100, 600.00 million, 6 * 1e8 * 4 Byte -> 2.4e9 Byte -> 2.4 GB (1 GB -> 1.0e9 Byte) │
│ (sf=1000, 6000.00 million, 6 * 1e9 * 4 Byte -> 2.4e10 Byte -> 24 GB (1 GB -> 1.0e9 Byte) │

1 GB -> 1.0e9 Byte) │
└────────────────┘
D CALL dbgen(sf = 10);
100% ▕████████████████████████████████████████████████████████████▏ 
┌─────────┐
│ Success │
│ boolean │
├─────────┤
│ 0 rows  │
└─────────┘
D select count(1) from lineitem;
┌─────────────────┐
│    count(1)     │
│      int64      │
├─────────────────┤
│    65987267     │
│ (65.99 million) │
└─────────────────┘
