#include "PatternTritonGPUOpToLLVM.h"
#include "TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"
#include "Utility.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "tlx/dialect/include/IR/Dialect.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/LinearLayout.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

namespace {

// Returns the printf format specifier for an LLVM type.
static std::string getFormatSubstr(Type ty, bool isSigned) {
  if (ty.isF16() || ty.isBF16() || ty.isF32() || ty.isF64())
    return "%f";
  if (ty.isInteger(64))
    return isSigned ? "%lli" : "%llu";
  return isSigned ? "%i" : "%u";
}

// Lowers tlx.print_reg_element to a predicated vprintf call.
//
// At lowering time, we:
//   1. Pseudoinvert the tensor's LinearLayout to map the compile-time logical
//      index → (register, lane, warp) hardware coordinates.
//   2. At runtime, predicate on (laneId == owningLane && warpId == owningWarp).
//   3. In the true branch, extract elems[owningReg] and call vprintf.
struct PrintRegElementOpConversion
    : public ConvertOpToLLVMPattern<mlir::triton::tlx::PrintRegElementOp> {
  explicit PrintRegElementOpConversion(LLVMTypeConverter &typeConverter,
                                       const TargetInfoBase &targetInfo,
                                       PatternBenefit benefit)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  LogicalResult
  matchAndRewrite(mlir::triton::tlx::PrintRegElementOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();

    // ── 1. Resolve owning hardware coordinates via LinearLayout pseudoinverse
    // ──
    auto tensorTy = cast<RankedTensorType>(op.getValue().getType());
    LinearLayout ll = toLinearLayout(tensorTy);
    LinearLayout invLL = ll.pseudoinvert();

    ArrayRef<int64_t> indices = op.getIndices();
    SmallVector<std::pair<StringAttr, int32_t>> logicalCoords;
    auto outDimNames = standardOutDimNames(ctx, indices.size());
    for (int i = 0; i < (int)indices.size(); ++i)
      logicalCoords.push_back(
          {outDimNames[i], static_cast<int32_t>(indices[i])});

    auto hwCoords = invLL.apply(logicalCoords);

    auto kReg = StringAttr::get(ctx, "register");
    auto kLane = StringAttr::get(ctx, "lane");
    auto kWarp = StringAttr::get(ctx, "warp");

    int32_t owningReg = 0, owningLane = 0, owningWarp = 0;
    for (auto [name, val] : hwCoords) {
      if (name == kReg)
        owningReg = val;
      else if (name == kLane)
        owningLane = val;
      else if (name == kWarp)
        owningWarp = val;
    }

    // ── 2. Extract the owning register's value from the LLVM struct ──
    SmallVector<Value> elems =
        unpackLLElements(loc, adaptor.getValue(), rewriter);
    assert((size_t)owningReg < elems.size() &&
           "register index out of range; logical index may be out of bounds");
    Value elem = elems[owningReg];

    // ── 3. Build the runtime predicate: am I the owning thread? ──
    auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);

    auto mkConst = [&](int32_t v) -> Value {
      return rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(),
                                               rewriter.getI32IntegerAttr(v));
    };
    Value isOwningLane = rewriter.create<LLVM::ICmpOp>(
        loc, LLVM::ICmpPredicate::eq, laneId, mkConst(owningLane));
    Value isOwningWarp = rewriter.create<LLVM::ICmpOp>(
        loc, LLVM::ICmpPredicate::eq, warpId, mkConst(owningWarp));
    Value pred = rewriter.create<LLVM::AndOp>(loc, isOwningLane, isOwningWarp);

    // ── 4. Build the format string ──
    auto module = op->getParentOfType<ModuleOp>();
    std::array<Value, 3> pid;
    for (auto axis : {ProgramIDDim::X, ProgramIDDim::Y, ProgramIDDim::Z})
      pid[(int)axis] = targetInfo.programId(rewriter, loc, module, axis);

    bool isSigned = op.getIsSigned();
    std::string msg = "pid (%u, %u, %u)" + op.getPrefix().str() +
                      getFormatSubstr(elem.getType(), isSigned) + "\n";

    // ── 5. Emit predicated printf via block splitting ──
    // Before: curBlock → [op] → rest-of-curBlock
    // After:  curBlock → printBlock → endBlock (rest-of-curBlock)
    auto *curBlock = rewriter.getInsertionBlock();
    auto *endBlock = curBlock->splitBlock(rewriter.getInsertionPoint());
    auto *printBlock = rewriter.createBlock(
        curBlock->getParent(), std::next(Region::iterator(curBlock)));

    // curBlock: conditional jump to printBlock or directly to endBlock.
    rewriter.setInsertionPointToEnd(curBlock);
    rewriter.create<LLVM::CondBrOp>(loc, pred, printBlock, endBlock);

    // printBlock: emit printf then jump to endBlock.
    rewriter.setInsertionPointToEnd(printBlock);
    SmallVector<Value> args = {pid[0], pid[1], pid[2], elem};
    SmallVector<bool> argsSigned = {false, false, false, isSigned};
    targetInfo.printf(rewriter, msg, args, argsSigned);
    rewriter.create<LLVM::BrOp>(loc, endBlock);

    // endBlock begins with the original op — erase it.
    rewriter.eraseOp(op);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

// Lowers tlx.print_smem_element to a predicated SMEM load + vprintf call.
//
// At lowering time, we:
//   1. Extract the smemObj (base ptr + per-dim subslice offsets) from the
//      converted memdesc struct.
//   2. Compute the static flat element offset by inverting the memdesc's
//      LinearLayout and applying the compile-time logical indices.
//   3. Add the dynamic subslice offset (from any local_view/local_slice ops).
//   4. GEP to the element pointer and predicate on thread 0 (laneId==0 &&
//      warpId==0).
//   5. In the true branch, load the element from SMEM and call vprintf.
struct PrintSmemElementOpConversion
    : public ConvertOpToLLVMPattern<mlir::triton::tlx::PrintSmemElementOp> {
  explicit PrintSmemElementOpConversion(LLVMTypeConverter &typeConverter,
                                        const TargetInfoBase &targetInfo,
                                        PatternBenefit benefit)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  LogicalResult
  matchAndRewrite(mlir::triton::tlx::PrintSmemElementOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();

    auto memDescTy = cast<MemDescType>(op.getValue().getType());
    auto llvmElemTy =
        getTypeConverter()->convertType(memDescTy.getElementType());

    // ── 1. Extract smemObj (base ptr + per-dim subslice offsets) ──
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getValue(), llvmElemTy, rewriter);

    // ── 2. Compute static flat element index within this memdesc ──
    // toLinearLayout maps {offset, block} → {dim0, dim1, ...}.
    // sublayout on {offset} then invert: {dim0, dim1, ...} → {offset}.
    int rank = memDescTy.getRank();
    auto dimNames = triton::standardOutDimNames(ctx, rank);
    auto kOffset = StringAttr::get(ctx, "offset");
    LinearLayout ll;
    if (auto paddedEnc = dyn_cast<triton::gpu::PaddedSharedEncodingAttr>(
            memDescTy.getEncoding()))
      ll = paddedEnc.getLinearComponent();
    else
      ll = triton::gpu::toLinearLayout(memDescTy);
    ll = ll.sublayout({kOffset}, dimNames).invert();

    ArrayRef<int64_t> indices = op.getIndices();
    SmallVector<std::pair<StringAttr, int32_t>> logicalCoords;
    for (int i = 0; i < rank; ++i)
      logicalCoords.push_back({dimNames[i], (int32_t)indices[i]});
    int32_t staticFlatOffset = ll.apply(logicalCoords)[0].second;

    // ── 3. Add the dynamic subslice offset ──
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Value dynamicOffset = smemObj.getShmemOffset(loc, rewriter, memDescTy);
    Value totalOffset = b.add(dynamicOffset, b.i32_val(staticFlatOffset));

    // ── 4. GEP to the element pointer ──
    Value elemPtr = b.gep(smemObj.getBase().getType(), llvmElemTy,
                          smemObj.getBase(), totalOffset);

    // ── 5. Predicate: only thread 0 (laneId==0 && warpId==0) prints ──
    auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
    Value isLane0 = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq,
                                                  laneId, b.i32_val(0));
    Value isWarp0 = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq,
                                                  warpId, b.i32_val(0));
    Value pred = rewriter.create<LLVM::AndOp>(loc, isLane0, isWarp0);

    // ── 6. Build format string ──
    auto module = op->getParentOfType<ModuleOp>();
    std::array<Value, 3> pid;
    for (auto axis : {ProgramIDDim::X, ProgramIDDim::Y, ProgramIDDim::Z})
      pid[(int)axis] = targetInfo.programId(rewriter, loc, module, axis);

    bool isSigned = op.getIsSigned();
    std::string msg = "pid (%u, %u, %u)" + op.getPrefix().str() +
                      getFormatSubstr(llvmElemTy, isSigned) + "\n";

    // ── 7. Emit predicated load + printf via block splitting ──
    auto *curBlock = rewriter.getInsertionBlock();
    auto *endBlock = curBlock->splitBlock(rewriter.getInsertionPoint());
    auto *printBlock = rewriter.createBlock(
        curBlock->getParent(), std::next(Region::iterator(curBlock)));

    rewriter.setInsertionPointToEnd(curBlock);
    rewriter.create<LLVM::CondBrOp>(loc, pred, printBlock, endBlock);

    rewriter.setInsertionPointToEnd(printBlock);
    Value elem = rewriter.create<LLVM::LoadOp>(loc, llvmElemTy, elemPtr);
    SmallVector<Value> args = {pid[0], pid[1], pid[2], elem};
    SmallVector<bool> argsSigned = {false, false, false, isSigned};
    targetInfo.printf(rewriter, msg, args, argsSigned);
    rewriter.create<LLVM::BrOp>(loc, endBlock);

    rewriter.eraseOp(op);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

// Lowers tlx.print_tmem_element to a predicated tcgen05.ld + predicated
// vprintf call.
//
// At lowering time, we:
//   1. Pseudoinvert the TMEM memdesc's LinearLayout to map the compile-time
//      logical index → physical (row, col) hardware coordinates.
//   2. Compute owningWarpInGroup = phys_row / 32 and owningLane = phys_row
//   % 32.
//   3. Emit "@$pred tcgen05.ld.sync.aligned.32x32b.x1.b32" in the main code
//      path so ALL warps execute the asm but non-owning warps get a NOP.
//      This avoids a divergent branch before a warp-collective instruction,
//      which can deadlock when preceded by other divergent print ops.
//   4. Predicate vprintf on (warpId & 3) == owningWarpInGroup && laneId ==
//      owningLane (a single block split for printf only).
struct PrintTmemElementOpConversion
    : public ConvertOpToLLVMPattern<mlir::triton::tlx::PrintTmemElementOp> {
  explicit PrintTmemElementOpConversion(LLVMTypeConverter &typeConverter,
                                        const TargetInfoBase &targetInfo,
                                        PatternBenefit benefit)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  LogicalResult
  matchAndRewrite(mlir::triton::tlx::PrintTmemElementOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();

    auto memDescTy = cast<MemDescType>(op.getValue().getType());
    auto llvmElemTy =
        getTypeConverter()->convertType(memDescTy.getElementType());

    // ── 1. Compute physical (row, col) via LinearLayout pseudoinverse ──
    // tensorMemoryToLinearLayout maps {row, col} (physical) → {dim0, dim1}
    // (logical). Inverting gives logical index → physical TMEM coordinates.
    auto dimNames = triton::standardOutDimNames(ctx, 2);
    LinearLayout ll = triton::gpu::toLinearLayout(memDescTy);
    LinearLayout invLL = ll.pseudoinvert();

    auto kRow = StringAttr::get(ctx, "row");
    auto kCol = StringAttr::get(ctx, "col");

    ArrayRef<int64_t> indices = op.getIndices();
    SmallVector<std::pair<StringAttr, int32_t>> logicalCoords;
    for (int i = 0; i < 2; ++i)
      logicalCoords.push_back({dimNames[i], (int32_t)indices[i]});

    auto hwCoords = invLL.apply(logicalCoords);
    int32_t physRow = 0, physCol = 0;
    for (auto [name, val] : hwCoords) {
      if (name == kRow)
        physRow = val;
      else if (name == kCol)
        physCol = val;
    }

    int32_t owningWarpInGroup = physRow / 32;
    int32_t owningLane = physRow % 32;

    // ── 2. Get the TMEM base address as i32 ──
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Value tmemBasePtr = adaptor.getValue();
    Value tmemBaseInt = b.ptrtoint(rewriter.getI32Type(), tmemBasePtr);
    // Add the owning warp group's row offset: owningWarpInGroup * 32 << 16.
    Value tmemAddr =
        b.add(tmemBaseInt, b.i32_val(owningWarpInGroup << (5 + 16)));

    // ── 3. Build runtime predicates ──
    auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
    Value warpIdInGroup = b.and_(warpId, b.i32_val(3));
    Value isOwningWarpGroup = rewriter.create<LLVM::ICmpOp>(
        loc, LLVM::ICmpPredicate::eq, warpIdInGroup,
        b.i32_val(owningWarpInGroup));
    Value isOwningLane = rewriter.create<LLVM::ICmpOp>(
        loc, LLVM::ICmpPredicate::eq, laneId, b.i32_val(owningLane));

    // ── 4. Build format string + pid ──
    auto module = op->getParentOfType<ModuleOp>();
    std::array<Value, 3> pid;
    for (auto axis : {ProgramIDDim::X, ProgramIDDim::Y, ProgramIDDim::Z})
      pid[(int)axis] = targetInfo.programId(rewriter, loc, module, axis);

    bool isSigned = op.getIsSigned();
    // For 32-bit element types bitcast the raw i32; otherwise print raw bits.
    Type elemTyForFormat =
        (llvmElemTy.isInteger(32) || llvmElemTy.isF32()) ? llvmElemTy : i32_ty;
    std::string msg = "pid (%u, %u, %u)" + op.getPrefix().str() +
                      getFormatSubstr(elemTyForFormat, isSigned) + "\n";

    // ── 5. Emit predicated tcgen05.ld then predicated printf ──
    //
    // The original two-level block split (branch to ldBlock for owning warp,
    // then branch to printBlock for owning lane) causes the GPU to hang when
    // preceded by other print operations with divergent branches. The root
    // cause is that a divergent CondBrOp before tcgen05.ld.sync can prevent
    // all 32 threads of the owning warp from reaching the instruction together.
    //
    // Fix: emit @$pred tcgen05.ld.sync in the main code path so ALL warps
    // see the same PTX asm but warps where isOwningWarpGroup==false get a NOP.
    // Then use a single block split only for vprintf, guarded by the combined
    // isOwningWarpGroup && isOwningLane predicate.
    //
    // Block structure (after fix):
    //   curBlock → @pred tcgen05.ld → (owning thread) → printBlock → endBlock
    //   printBlock → printf → endBlock

    // Issue @pred tcgen05.ld in the main code path (no block split here).
    PTXBuilder ptxBuilder;
    std::string ldOpcode =
        "@$2 tcgen05.ld.sync.aligned.32x32b.x1.b32 {$0}, [$1 + " +
        std::to_string(physCol) + "];";
    auto *resultOp = ptxBuilder.newOperand("=r");
    auto *addrOp = ptxBuilder.newOperand(tmemAddr, "r");
    auto *predOp = ptxBuilder.newOperand(isOwningWarpGroup, "b");
    auto &ldInstr = *ptxBuilder.create<PTXInstr>(ldOpcode);
    ldInstr({resultOp, addrOp, predOp}, /*onlyAttachMLIRArgs=*/true);
    Value rawI32 = ptxBuilder.launch(rewriter, loc, i32_ty);

    // Cast raw i32 to the element type for formatting.
    Value elem;
    if (llvmElemTy.isInteger(32) || llvmElemTy.isF32())
      elem = b.bitcast(rawI32, llvmElemTy);
    else
      elem = rawI32; // sub-32-bit types: print raw i32 bits

    // Single block split: only the owning thread (owning warp + owning lane)
    // enters printBlock.
    Value isOwningThread =
        rewriter.create<LLVM::AndOp>(loc, isOwningWarpGroup, isOwningLane);
    auto *curBlock = rewriter.getInsertionBlock();
    auto *endBlock = curBlock->splitBlock(rewriter.getInsertionPoint());
    auto *printBlock = rewriter.createBlock(
        curBlock->getParent(), std::next(Region::iterator(curBlock)));

    rewriter.setInsertionPointToEnd(curBlock);
    rewriter.create<LLVM::CondBrOp>(loc, isOwningThread, printBlock, endBlock);

    // printBlock: printf and jump to endBlock.
    rewriter.setInsertionPointToEnd(printBlock);
    SmallVector<Value> args = {pid[0], pid[1], pid[2], elem};
    SmallVector<bool> argsSigned = {false, false, false, isSigned};
    targetInfo.printf(rewriter, msg, args, argsSigned);
    rewriter.create<LLVM::BrOp>(loc, endBlock);

    rewriter.eraseOp(op);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

} // namespace

void mlir::triton::NVIDIA::populateTLXOpsToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfoBase &targetInfo, PatternBenefit benefit) {
  patterns.add<PrintRegElementOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<PrintSmemElementOpConversion>(typeConverter, targetInfo,
                                             benefit);
  patterns.add<PrintTmemElementOpConversion>(typeConverter, targetInfo,
                                             benefit);
}
