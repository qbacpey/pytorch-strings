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

