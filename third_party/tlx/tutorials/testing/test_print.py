"""Minimal tests for TLX utility APIs."""
import torch
import triton
import triton.language as tl
from triton.language.extra import tlx

# ---------------------------------------------------------------------------
# Register tensor kernel
# ---------------------------------------------------------------------------


@triton.jit
def _kernel_reg_print_element(N: tl.constexpr, IDX: tl.constexpr):
    """Create vals[i] = i*10+1 in registers, print the element at IDX."""
    # Multiply by 10 and add 1 so values are distinct from indices.
    vals = (tl.arange(0, N) * 10 + 1).to(tl.float32)
    # The owning thread is computed automatically from the tensor's layout.
    tlx.print_element(vals, [IDX], prefix="reg")


def test_reg_print_element(capfd):
    """
    tlx.print_reg_element pseudoinverts the tensor's LinearLayout to find
    the (register, lane, warp) that owns the requested logical index, then
    emits a single predicated vprintf from that thread.

    vals[i] = i*10+1, so vals[123456] = 1234561.0.
    Exactly one output line should appear regardless of num_warps.
    """
    N = 2**20  # ~1M elements; must be a power of 2 for tl.arange
    idx = 123456
    expected_val = idx * 10 + 1  # 1234561
    _kernel_reg_print_element[(1, )](N=N, IDX=idx, num_warps=32)
    torch.cuda.synchronize()

    out = capfd.readouterr().out
    print(out)
    assert f"reg[{idx}]" in out, f"Expected 'reg[{idx}]' in output, got:\n{out!r}"
    assert str(expected_val) in out, f"Expected value '{expected_val}' in output, got:\n{out!r}"


# ---------------------------------------------------------------------------
# SMEM kernel
# ---------------------------------------------------------------------------


@triton.jit
def _kernel_smem_print_element(N: tl.constexpr, IDX: tl.constexpr):
    """Store vals[i] = i+100 to SMEM, then print the element at IDX."""
    smem = tlx.local_alloc((N, ), tl.float32, tl.constexpr(1))
    # Add 100 so values are distinct from indices.
    vals = (tl.arange(0, N) + 100).to(tl.float32)
    tlx.local_store(tlx.local_view(smem, 0), vals)
    tlx.print_element(smem, [IDX], prefix="buf")


def test_smem_print_element(capfd):
    """
    tlx.print_element should print the correct value for the requested
    SMEM coordinate.

    vals[i] = i+100, so vals[5] = 105.0.
    Expected output line:
        pid (0, 0, 0) idx (0) buf[5]: 105.000000
    """
    N = 32
    idx = 5
    expected_val = idx + 100  # 105

    _kernel_smem_print_element[(1, )](N=N, IDX=idx, num_warps=1)
    torch.cuda.synchronize()

    out = capfd.readouterr().out
    assert f"buf[{idx}]" in out, (f"Expected 'buf[{idx}]' in kernel stdout, got:\n{out!r}")
    assert str(expected_val) in out, (f"Expected value '{expected_val}' in kernel stdout, got:\n{out!r}")


# ---------------------------------------------------------------------------
# TMEM kernel  (Blackwell only)
# ---------------------------------------------------------------------------


@triton.jit
def _kernel_tmem_print_element(
    ROW: tl.constexpr,
    COL: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
):
    """Allocate TMEM [M, N], fill with 99.0, print element [ROW, COL]."""
    tmem = tlx.local_alloc((M, N), tl.float32, tl.constexpr(1), tlx.storage_kind.tmem)
    vals = tl.full((M, N), 99.0, dtype=tl.float32)
    tlx.local_store(tlx.local_view(tmem, 0), vals)
    tlx.print_element(tmem, [ROW, COL], prefix="tmem")


def test_tmem_print_element(capfd):
    """
    tlx.print_element on a TMEM tensor loads one column (warp-collective),
    then prints from the thread that owns the requested row.

    Expected output line (thread ROW owns row ROW of loaded [M,1] tensor):
        pid (0, 0, 0) idx (...) tmem[5][0]: 99.000000
    """
    M, N = 128, 4  # warpgroup rows × min TMEM columns
    row, col = 5, 0
    _kernel_tmem_print_element[(1, )](ROW=row, COL=col, M=M, N=N, num_warps=4)
    torch.cuda.synchronize()

    out = capfd.readouterr().out
    assert f"tmem[{row}][{col}]" in out, (f"Expected 'tmem[{row}][{col}]' in output, got:\n{out!r}")
    assert "99" in out, f"Expected value '99' in output, got:\n{out!r}"


def test_gemm():
    """
    Test D = A*B
    """

    @triton.jit
    def tcgen5_dot_kernel(
        a_ptr,
        stride_am,
        stride_ak,
        b_ptr,
        stride_bk,
        stride_bn,
        c_ptr1,
        stride_cm,
        stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        OUT_DTYPE: tl.constexpr,
    ):
        offs_m = tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        # async load a and b into SMEM
        buf_alloc_a = tlx.local_alloc((BLOCK_M, BLOCK_K), tl.float16, tl.constexpr(1))
        buf_alloc_b = tlx.local_alloc((BLOCK_K, BLOCK_N), tl.float16, tl.constexpr(1))
        a_smem = tlx.local_view(buf_alloc_a, 0)
        b_smem = tlx.local_view(buf_alloc_b, 0)
        tlx.async_load(a_ptrs, a_smem)
        tlx.async_load(b_ptrs, b_smem)
        tlx.async_load_commit_group()
        tlx.async_load_wait_group(tl.constexpr(0))

        buffers = tlx.local_alloc((BLOCK_M, BLOCK_N), tl.float32, tl.constexpr(1), tlx.storage_kind.tmem)
        acc_tmem = tlx.local_view(buffers, 0)

        # fill tmem d with 1
        acc_init = tl.full((BLOCK_M, BLOCK_N), 1, dtype=tl.float32)
        tlx.local_store(acc_tmem, acc_init)
        # do not use d (so that we get A*B instead of A*B+1)
        tlx.print_element(a_smem, [37, 17], prefix='a')
        tlx.async_dot(a_smem, b_smem, acc_tmem, use_acc=False, mBarriers=[], out_dtype=OUT_DTYPE)

        tlx.print_element(acc_tmem, [38, 18], prefix='acc_tmem')
        # c1 = A*B
        c1 = tlx.local_load(acc_tmem).to(tl.float16)
        tlx.print_element(c1, [39, 19], prefix='acc_rmem')
        c_ptrs = c_ptr1 + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
        tl.store(c_ptrs, c1)

    device = 'cuda'
    torch.manual_seed(0)
    M, N, K = (64, 64, 32)
    x = torch.randn((M, K), device=device, dtype=torch.float16)
    print('SMEM x[37][17]:', x[37][17])
    y = torch.randn((K, N), device=device, dtype=torch.float16)
    z1 = torch.zeros((M, N), device=device, dtype=torch.float16)

    kern_kwargs = {"BLOCK_M": M, "BLOCK_K": K, "BLOCK_N": N, "OUT_DTYPE": tl.float32}
    tcgen5_dot_kernel[(1, 1)](x, x.stride(0), x.stride(1), y, y.stride(0), y.stride(1), z1, z1.stride(0), z1.stride(1),
                              **kern_kwargs)

    xy = torch.matmul(x, y)
    print('RMEM xy[39][19]:', xy[39][19])
    print('TMEM xy[38][18]:', xy[38][18])
    torch.testing.assert_close(z1, xy)


@triton.jit
def generated_kernel_gemm_scale_print(
    ROW: tl.constexpr,
    COL: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
):
    # ── A tile in SMEM: A[i,k] = i+1 ────────────────────────────────────────
    smem_a_buf = tlx.local_alloc((BM, BK), tl.float16, tl.constexpr(1))
    smem_a = tlx.local_view(smem_a_buf, 0)
    a_vals = tl.zeros((BM, BK), dtype=tl.float16) + (tl.arange(0, BM) + 1).to(tl.float16)[:, None]
    tlx.local_store(smem_a, a_vals)

    # ── B tile in SMEM: B[k,j] = 1.0 ────────────────────────────────────────
    smem_b_buf = tlx.local_alloc((BK, BN), tl.float16, tl.constexpr(1))
    smem_b = tlx.local_view(smem_b_buf, 0)
    tlx.local_store(smem_b, tl.full((BK, BN), 1.0, dtype=tl.float16))

    # ── scale in registers: scale[i,j] = j+1 ─────────────────────────────────
    scale = tl.zeros((BM, BN), dtype=tl.float32) + (tl.arange(0, BN) + 1).to(tl.float32)[None, :]

    # ── TMEM accumulator pre-loaded with scale, then MMA adds A@B ────────────
    # tlx.async_dot accumulates into TMEM, so initialising with scale gives
    # D[i,j] = scale[i,j] + (A@B)[i,j] = (j+1) + BK*(i+1) without needing
    # a register round-trip or a second local_store.
    tmem_buf = tlx.local_alloc((BM, BN), tl.float32, tl.constexpr(1), tlx.storage_kind.tmem)
    acc_tmem = tlx.local_view(tmem_buf, 0)
    tlx.local_store(acc_tmem, scale)

    dot_bars = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    dot_bar = tlx.local_view(dot_bars, 0)
    tlx.async_dot(smem_a, smem_b, acc_tmem, mBarriers=[dot_bar], out_dtype=tl.float32)
    tlx.barrier_wait(dot_bar, 0)

    # ── Print one element from each tensor ────────────────────────────────────
    tlx.print_element(smem_a_buf, [ROW, 0], prefix="A")  # → ROW+1
    tlx.print_element(smem_b_buf, [0, COL], prefix="B")  # → 1.0
    tlx.print_element(scale, [ROW, COL], prefix="scale")  # → COL+1
    # tlx.print_element(tmem_buf, [ROW, COL], prefix="D")  # → BK*(ROW+1)+(COL+1)  <--- this line hangs the kernel


def generated_test_gemm(capfd):
    """
    D = A @ B + scale with:
      A[i,k] = i+1  (fp16 SMEM),  B[k,j] = 1  (fp16 SMEM),
      scale[i,j] = j+1  (fp32 register),  D[i,j] = BK*(i+1) + (j+1)  (fp32 TMEM).

    Exercises all three print_element paths in a single kernel:
      - SMEM A and B  (tlx.print_smem_element  → LL-invert + tcgen05-free GEP/load)
      - register scale (tlx.print_reg_element  → LL-pseudoinvert + predicated printf)
      - TMEM D         (tlx.print_tmem_element → LL-pseudoinvert + tcgen05.ld)

    Four lines are expected in stdout; each carries a distinct value that
    cannot be confused with any other.
    """
    BM, BK, BN = 128, 64, 128
    row, col = 5, 7
    expected_a = row + 1  # 6
    expected_b = 1
    expected_scale = col + 1  # 8
    expected_d = BK * (row + 1) + (col + 1)  # 64*6+8 = 392

    generated_kernel_gemm_scale_print[(1, )](ROW=row, COL=col, BM=BM, BK=BK, BN=BN, num_warps=4)
    torch.cuda.synchronize()

    out = capfd.readouterr().out
    lines = out.splitlines()

    def find_line(label):
        return next((l for l in lines if label in l), None)

    a_line = find_line(f"A[{row}][0]")
    b_line = find_line(f"B[0][{col}]")
    scale_line = find_line(f"scale[{row}][{col}]")
    d_line = find_line(f"D[{row}][{col}]")

    assert a_line, f"no A line in output:\n{out!r}"
    assert b_line, f"no B line in output:\n{out!r}"
    assert scale_line, f"no scale line in output:\n{out!r}"
    assert d_line, f"no D line in output:\n{out!r}"

    assert f"{expected_a}." in a_line, f"A value wrong: {a_line!r}"
    assert f"{expected_b}." in b_line, f"B value wrong: {b_line!r}"
    assert f"{expected_scale}." in scale_line, f"scale value wrong: {scale_line!r}"
    assert str(expected_d) in d_line, f"D value wrong: {d_line!r}"
