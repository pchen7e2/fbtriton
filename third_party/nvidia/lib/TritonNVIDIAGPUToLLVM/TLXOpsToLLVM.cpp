#include "PatternTritonGPUOpToLLVM.h"
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
    for (int i = 0; i < (int)indices.size(); ++i)
      logicalCoords.push_back({StringAttr::get(ctx, "dim" + std::to_string(i)),
                               static_cast<int32_t>(indices[i])});

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

} // namespace

void mlir::triton::NVIDIA::populateTLXOpsToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfoBase &targetInfo, PatternBenefit benefit) {
  patterns.add<PrintRegElementOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<PrintSmemElementOpConversion>(typeConverter, targetInfo,
                                             benefit);
}
