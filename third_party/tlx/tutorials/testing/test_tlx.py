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
