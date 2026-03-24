import triton.language.core as tl

import re
import triton.runtime.driver as driver


def is_hip():
    target = driver.active.get_current_target()
    return target.backend == "hip"


def cuda_parse_arch(arch):
    pattern = r"^sm(\d+)$"
    match = re.fullmatch(pattern, arch)
    if not match:
        raise ValueError(f"TRITON_OVERRIDE_ARCH must have the form {pattern}")
    return int(match.group(1))


@tl.builtin
def cluster_cta_rank(_semantic=None):
    """
    :return the unique CTA ID within a cluster across all dims
    """
    return tl.tensor(_semantic.builder.create_cluster_cta_rank(), tl.int32)


@tl.builtin
def cluster_size_1d(_semantic=None):
    """
    :return the total number of CTAs in the cluster across all dimensions
    (equal to the product of sizes of every dimension).
    """
    return tl.tensor(_semantic.builder.create_cluster_size_1d(), tl.int32)


@tl.builtin
def thread_id(axis, _semantic=None):
    """
    Returns the id of the current thread instance along the given :code:`axis`.

    :param axis: The axis of the 3D launch grid. Must be 0, 1 or 2.
    :type axis: int
    """
    axis = tl._unwrap_if_constexpr(axis)
    if axis not in (0, 1, 2):
        raise ValueError(f"thread_id axis must be 0, 1, or 2 but got {axis}")
    return tl.tensor(_semantic.builder.create_thread_id(axis), tl.int32)


@tl.builtin
def async_task_replica_id(_semantic=None):
    from triton.language.extra.tlx.compiler.code_generator import _get_region_replica_id_stack

    region_replica_id_stack = _get_region_replica_id_stack()
    assert len(region_replica_id_stack) > 0, (
        "async_task_replica_id must be called inside an async region where the stack must be non-empty")
    return tl.constexpr(region_replica_id_stack[-1])


@tl.builtin
def dtype_of(v, _semantic=None) -> tl.dtype:
    """
    Returns the element type of a given tensor or tensor descriptor.
    """
    if isinstance(v, tl.tensor):
        dtype = v.type.element_ty
        if dtype.is_ptr():
            dtype = dtype.element_ty
        return dtype
    elif isinstance(v, tl.tensor_descriptor_base):
        return v.dtype
    else:
        raise ValueError(f"dtype_of only works on tensors and tensor descriptors, but got {v}")


@tl.builtin
def size_of(dtype: tl.dtype, _semantic=None) -> tl.constexpr:
    """
    Returns the size of a given dtype.
    """
    dtype = tl._unwrap_if_constexpr(dtype)
    assert isinstance(dtype, tl.dtype), f"size_of expects a dtype, but got {type(dtype)}"
    return tl.constexpr(dtype.primitive_bitwidth // 8)


@tl.builtin
def get_fp8_format_name(dtype: tl.dtype, _semantic=None) -> tl.constexpr:
    """
    Returns the FP8 format name string for a given FP8 dtype.

    This extracts the format identifier (e.g., "e5m2", "e4m3") from the dtype
    for use with scaled MMA operations like async_dot_scaled.

    Args:
        dtype: An FP8 dtype (tl.float8e5m2 or tl.float8e4nv)

    Returns:
        A constexpr string with the format name ("e5m2" or "e4m3")

    Raises:
        AssertionError: If the dtype is not a supported FP8 type.

    Example:
        Q_FP8_FORMAT: tl.constexpr = tlx.get_fp8_format_name(tlx.dtype_of(desc_q))
    """
    # Unwrap constexpr if needed (when dtype is passed as a tl.constexpr kernel parameter)
    dtype = tl._unwrap_if_constexpr(dtype)
    assert isinstance(dtype, tl.dtype), f"get_fp8_format_name expects a dtype, but got {type(dtype)}"
    # Only support FP8 types that map to "e5m2" or "e4m3" for scaled MMA operations
    if dtype == tl.float8e5:
        return tl.constexpr("e5m2")
    elif dtype == tl.float8e4nv:
        return tl.constexpr("e4m3")
    else:
        raise AssertionError(f"get_fp8_format_name only supports tl.float8e5 (e5m2) and tl.float8e4nv (e4m3), "
                             f"but got {dtype}")


@tl.builtin
def clock64(_semantic=None):
    """
    Returns the current 64-bit hardware clock value.
    The returned value is the number of clock cycles since the device was powered on or reset.
    This is useful for measuring elapsed time or performance of specific code regions.
    Returns:
        tl.tensor: A tensor containing the current 64-bit clock value as an int64.
    Example:
        start = tlx.clock64()
        # ... kernel code ...
        end = tlx.clock64()
        elapsed = end - start  # Number of clock cycles elapsed
    """
    return tl.tensor(_semantic.builder.create_clock64(), tl.int64)


def _print_element_emit_predicated(value_handle, is_signed, full_prefix, thread, _semantic):
    """Gate tt.print on tid == thread using scf.if."""
    b = _semantic.builder
    tid = b.create_thread_id(0)
    cond = b.create_icmpEQ(tid, b.get_int32(thread))
    if_op = b.create_if_op([], cond, False)
    ip = b.get_insertion_point()
    # create_if_op auto-inserts a scf.yield terminator; insert our ops at
    # the start of the block so they land before the auto-yield.
    b.set_insertion_point_to_start(if_op.get_then_block())
    b.create_print(full_prefix, False, [value_handle], is_signed)
    b.restore_insertion_point(ip)


@tl.builtin
def print_element(tensor_or_buf, indices, prefix="", thread=0, _semantic=None):
    """
    Print a single element of a tensor by logical coordinate, regardless of where
    the tensor lives (register, SMEM, or TMEM).

    Args:
        tensor_or_buf: A register tensor (tl.tensor) or buffered_tensor (SMEM/TMEM).
        indices:  List of compile-time integer indices, one per dimension.
        prefix:   Optional string label prepended to the output.
        thread:   Flat CTA thread ID (tid.x) that issues the print.
                  - Register: ignored; the owning thread is computed automatically
                    from the tensor's LinearLayout via pseudoinverse.
                  - SMEM: must be 0 (the 1×1 local_load only exposes the value to
                    thread 0 via BlockedEncoding).
                  - TMEM: not allowed; raises a compile-time error.

    Output format::

        pid (x, y, z) <prefix>[i][j]: <value>   (register)
        pid (x, y, z) idx (...) <prefix>[i][j]: <value>  (SMEM/TMEM)

    Examples::

        # Register tensor — owning thread computed automatically from layout
        tlx.print_element(vals, [123456], prefix="val")

        # SMEM buffered_tensor
        tlx.print_element(smem_a, [row, col], prefix="A")
    """
    from .types import buffered_tensor as _buffered_tensor, storage_kind as _storage_kind

    indices = [tl._unwrap_if_constexpr(i) for i in indices]
    thread = tl._unwrap_if_constexpr(thread)
    prefix = tl._unwrap_if_constexpr(prefix)

    idx_str = "".join(f"[{i}]" for i in indices)
    full_prefix = f" {prefix}{idx_str}: " if prefix else f" {idx_str}: "

    if isinstance(tensor_or_buf, tl.tensor):
        # ── Register tensor ──────────────────────────────────────────────────
        # Emit tlx.print_reg_element, which is lowered in the NVIDIA TTGIR→LLVM
        # conversion pass.  The lowering pseudoinverts the tensor's LinearLayout
        # to find which (register, lane, warp) owns the requested logical index,
        # then emits a predicated vprintf from only that thread.
        is_signed = tensor_or_buf.dtype.is_int_signed()
        _semantic.builder.create_print_reg_element(tensor_or_buf.handle, indices, full_prefix, is_signed)

    elif isinstance(tensor_or_buf, _buffered_tensor):
        if tensor_or_buf.type.storage == _storage_kind.tmem:
            # ── TMEM buffered_tensor ─────────────────────────────────────────
            # TMEM loads (tcgen05.ld) are warp-collective: all threads in the
            # warp must participate.  We therefore:
            #   1. Load the single column j unconditionally (warp-collective).
            #   2. Gate tt.print on tid == row so only thread `row` prints.
            #      Thread `row` owns element [row, 0] of the loaded [M,1] tensor.
            # The `thread` argument is intentionally ignored for TMEM — the
            # issuing thread is always determined by the row index.
            assert len(indices) == 2, (f"TMEM tensors are rank-2 [M, N]; got {len(indices)} indices.")
            row, col = indices[0], indices[1]
            M = tensor_or_buf.shape[0]

            from .mem_ops import local_load, local_slice, local_view

            buf = tensor_or_buf
            if buf.type.num > 0:
                buf = local_view(buf, 0, _semantic=_semantic)

            # Slice the single column containing the element.
            col_slice = local_slice(buf, [0, col], [M, 1], _semantic=_semantic)
            # Warp-collective load — ALL threads must execute this.
            col_tensor = local_load(col_slice, _semantic=_semantic)
            # Only thread `row` prints; it owns tmem_buf[row, col] = col_tensor[row, 0].
            is_signed = [col_tensor.dtype.is_int_signed()]
            _print_element_emit_predicated(col_tensor.handle, is_signed, full_prefix, row, _semantic)
            return

        # ── SMEM buffered_tensor ─────────────────────────────────────────────
        from .mem_ops import local_view

        buf = tensor_or_buf
        if buf.type.num > 0:
            buf = local_view(buf, 0, _semantic=_semantic)

        is_signed = [buf.dtype.is_int_signed()]
        _semantic.builder.create_print_smem_element(buf.handle, indices, full_prefix, is_signed[0])

    else:
        raise TypeError(f"print_element expects a tl.tensor (register) or buffered_tensor (SMEM/TMEM), "
                        f"got {type(tensor_or_buf).__name__}.")


@tl.builtin
def stoch_round(
    src: tl.tensor,
    dst_ty: tl.dtype,
    rand_bits: tl.tensor,
    _semantic=None,
) -> tl.tensor:
    """
    Hardware-accelerated stochastic rounding for FP32→FP8/BF16/F16 conversions.

    Requires Blackwell GPU (compute capability >= 100).

    Semantics:
        y = tlx.stoch_round(src, dst_ty, rand_bits)

    Maps to PTX (on Blackwell):
        cvt.rs.satfinite.{e4m3x4,e5m2x4}.f32  d, {a,b,c,d}, rbits  (for FP8)
        cvt.rs.satfinite.{bf16x2,f16x2}.f32   d, {a,b}, rbits      (for BF16/F16)

    Args:
        src:
            Source FP32 tensor. Shape defines output shape.
        dst_ty:
            Destination dtype: tl.float8e5, tl.float8e4nv, tl.float16, or tl.bfloat16
        rand_bits:
            Random bits (uint32 tensor) for entropy, must match src shape

    Returns:
        Tensor with dtype dst_ty and shape matching src.
    """
    capability = int(cuda_parse_arch(_semantic.builder.options.arch))
    assert capability >= 100, (f"stoch_round requires compute capability >= 100 (Blackwell GPU), "
                               f"current capability: {capability}")
    src_ty = src.type
    src_sca_ty = src_ty.scalar

    assert src_sca_ty == tl.float32, (f"Stochastic rounding only supports fp32 source, got {src_sca_ty}. "
                                      f"Source must be float32.")
    assert dst_ty in [tl.float8e5, tl.float8e4nv, tl.float16, tl.bfloat16
                      ], (f"Stochastic rounding only supports fp8/fp16/bf16 destination, got {dst_ty}. "
                          f"Supported types: float8e5 (fp8 E5M2), float8e4nv (fp8 E4M3FN), float16, bfloat16")

    # Verify rbits shape matches src shape
    rbits_ty = rand_bits.type
    if src_ty.is_block() and rbits_ty.is_block():
        assert src_ty.shape == rbits_ty.shape, f"rand_bits shape {rbits_ty.shape} must match src shape {src_ty.shape}"
    elif not src_ty.is_block() and not rbits_ty.is_block():
        # Both are scalars - OK
        pass
    else:
        raise ValueError(f"src and rand_bits must both be blocks or both be scalars, "
                         f"got src_ty.is_block()={src_ty.is_block()}, rbits_ty.is_block()={rbits_ty.is_block()}")

    if src_sca_ty == dst_ty:
        return src
    # Construct the proper result type (block type if source is block)
    if src_ty.is_block():
        result_ty = src_ty.with_element_ty(dst_ty)
        dst_ir_ty = result_ty.to_ir(_semantic.builder)
    else:
        result_ty = dst_ty
        dst_ir_ty = dst_ty.to_ir(_semantic.builder)
    dst = _semantic.builder.create_cvt_rs(src.handle, dst_ir_ty, rand_bits.handle)
    return tl.tensor(dst, result_ty)
