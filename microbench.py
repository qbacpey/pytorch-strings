import sys
import torch
import time

def my_function(a,mini,maxi):
    bitmap = (a >= mini) & (a <= maxi)
    # return bitmap
    return bitmap

def test_mask_nonzero(target_device,target_mb,num_streams,mini,maxi):
    num_tuples = (target_mb*(1<<20)) // torch.int64.itemsize
    data = [torch.randint(0,400,(num_tuples,),dtype=torch.int64,device=target_device) for _ in range(num_streams)]
    streams = [torch.cuda.Stream(device=target_device) for _ in range(num_streams)]
    fun_handle = torch.compile(my_function)
    torch.cuda.synchronize()
    fun_handle(data[0],mini,maxi) # warm up
    torch.cuda.synchronize()

    with torch.cuda.nvtx.range(f"Range [{mini},{maxi}]"):
        start = time.perf_counter_ns()
        for streamidx,stream in enumerate(streams):
            with torch.cuda.stream(stream):
                # bitmap = (data[streamidx] >= mini) & (data[streamidx] <= maxi)
                # bitmap = (data[streamidx] >= mini)
                # result = bitmap.nonzero().view(-1)
                result = fun_handle(data[streamidx],mini,maxi)
        for streamidx,stream in enumerate(streams):
            stream.synchronize()
        end = time.perf_counter_ns()

    bw = (target_mb*num_streams/1024)/((end-start)/1e9)
    print(f"{target_mb=}, {num_tuples=}, {data[0].shape[0]*torch.int64.itemsize=}")
    print(f"{mini=}, {maxi=}, {result.sum().item()=}, {(end-start)=}, {bw=:.6f}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <device_id>")
        exit(1)

    device_id = int(sys.argv[1])
    test_mask_nonzero(device_id,1024,1,0,400)
    test_mask_nonzero(device_id,1024,1,0,400)
    test_mask_nonzero(device_id,1024,1,0,400)
    test_mask_nonzero(device_id,1024,1,100,300)
    test_mask_nonzero(device_id,1024,1,200,223)
    test_mask_nonzero(device_id,1024,1,300,301)
    test_mask_nonzero(device_id,1024,1,600,1000)
