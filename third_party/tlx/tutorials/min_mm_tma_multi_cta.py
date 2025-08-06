import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.tools.tensor_descriptor import TensorDescriptor

# BLOCK_SIZE = 64


def torch_matmul(a, b):
    c = torch.matmul(a, b)
    return c


def matmul_tma_set_block_size_hook(nargs):
    BLOCK_M = nargs["BLOCK_SIZE_M"]
    BLOCK_N = nargs["BLOCK_SIZE_N"]
    BLOCK_K = nargs["BLOCK_SIZE_K"]
    nargs["a_desc"].block_shape = [BLOCK_M, BLOCK_K]
    nargs["b_desc"].block_shape = [BLOCK_K, BLOCK_N]
    # EPILOGUE_SUBTILE = nargs.get("EPILOGUE_SUBTILE", False)
    # if EPILOGUE_SUBTILE:
    #     nargs["c_desc"].block_shape = [BLOCK_M, BLOCK_N // 2]
    # else:
    nargs["c_desc"].block_shape = [BLOCK_M, BLOCK_N]


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                # "GROUP_SIZE_M": gs,
                # "NUM_SMEM_BUFFERS": s,
                # "NUM_TMEM_BUFFERS": t,
            },
            num_warps=4,
            num_stages=1,
            num_ctas=2,
            pre_hook=matmul_tma_set_block_size_hook,
        )
    ],
    key=["M", "N", "K"],
)
@triton.jit
def min_matmul_kernel_multi(
        # Pointers to matrices
        a_desc, b_desc, c_desc, a_ptr, b_ptr, c_ptr,  #
        # Matrix dimensions
    M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
):
    pid = tl.program_id(axis=0)
    offs_am = 0
    offs_bn = pid * BLOCK_SIZE_N

    # allocate NUM_SMEM_BUFFERS buffers
    buffers_A = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_K), tl.float16, 1)
    buffers_B = tlx.local_alloc((BLOCK_SIZE_K, BLOCK_SIZE_N), tl.float16, 1)
    buf_A = tlx.local_view(buffers_A, 0)
    buf_B = tlx.local_view(buffers_B, 0)

    smem_full_bars = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    smem_full_bar = tlx.local_view(smem_full_bars, 0)

    # buffers = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_N), tl.float32, tl.constexpr(1), tlx.storage_kind.tmem)
    # acc_tmem = tlx.local_view(buffers, 0)
    # acc_init = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # tlx.local_store(acc_tmem, acc_init, tlx.storage_kind.tmem)

    # load B into SMEM without TMA
    # tlx.barrier_expect_bytes(smem_full_bar, 2 * (BLOCK_SIZE_M) * BLOCK_SIZE_K)  # float16
    # b_ptrs = b_ptr + (tl.arange(0, BLOCK_SIZE_K)[:, None] * stride_bk + (offs_bn + tl.arange(0, BLOCK_SIZE_N))[None, :] * stride_bn)
    # tlx.async_load(b_ptrs, buf_B)
    # tlx.async_load_commit_group()
    # tlx.async_load_wait_group(tl.constexpr(0))
    tlx.barrier_expect_bytes(smem_full_bar, 2 * (BLOCK_SIZE_M + BLOCK_SIZE_N) * BLOCK_SIZE_K // 2)  # float16
    tlx.async_descriptor_load(b_desc, buf_B, [0, offs_bn], smem_full_bar)
    # if pid == 1: # remove this if when not using TMA multicast
    tlx.async_descriptor_load(a_desc, buf_A, [offs_am, 0], smem_full_bar)

    tlx.barrier_wait(smem_full_bar, 0)

    # tlx.async_dot(buf_A, buf_B, acc_tmem, mBarriers=[], out_dtype=tl.float32)

    # result = tlx.local_load(acc_tmem, tlx.storage_kind.tmem)

    a_tensor = tlx.local_load(buf_A)
    b_tensor = tlx.local_load(buf_B)
    result = tl.dot(a_tensor, b_tensor)

    c = result.to(tl.float16)
    c_desc.store([offs_am, offs_bn], c)


def matmul(a, b):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (1, )

    # A dummy block value that will be overwritten when we have the real block size
    dummy_block = [1, 1]
    a_desc = TensorDescriptor(a, a.shape, a.stride(), dummy_block)
    b_desc = TensorDescriptor(b, b.shape, b.stride(), dummy_block)
    c_desc = TensorDescriptor(c, c.shape, c.stride(), dummy_block)

    min_matmul_kernel_multi[grid](
        a_desc, b_desc, c_desc,  #
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        # BLOCK_SIZE,  #
    )
    return c


def run_test(expect, fn, a, b, label, enabled=True):
    print(f"  {label}: ...", end="")
    if enabled:
        actual = fn(a, b)
        passed = torch.allclose(expect, actual.to(expect.dtype))
        icon = "✅" if passed else "❌"
        print(actual)
        print(expect)
        # import pdb; pdb.set_trace()
        torch.testing.assert_close(expect[64:, :], actual.to(expect.dtype)[64:, :], atol=1e-2, rtol=0)
    else:
        icon = "⭕"
    print(f"\r  {label}: {icon}  ")


def validate(M, N, K, dtype):
    print(f"{M=}, {N=}, {K=}, verification Torch Matmul vs: ")
    a = torch.randn((M, K), device="cuda", dtype=torch.float16).to(dtype)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16).to(dtype)

    ref_output = torch_matmul(a, b).to(torch.float16)

    run_test(ref_output, lambda a, b: matmul(a, b), a, b, 'min matmul')
    print()


def main():
    dtype = torch.float16

    torch.manual_seed(0)

    # validate(32, 32, 32, dtype)
    # validate(8192, 8192, 512, dtype)
    validate(128, 128, 128, dtype)


if __name__ == "__main__":
    main()
