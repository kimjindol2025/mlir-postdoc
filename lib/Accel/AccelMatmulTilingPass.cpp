// ===== AccelMatmulTilingPass.cpp: 핵심 Pass 구현 =====
// Step 2→3 변환: Affine Loops → LLVM Intrinsics
// 핵심: 주소 정렬 검증 및 패딩 처리

#include "Accel/AccelMatmulTilingPass.h"
#include "Accel/MemoryUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/LLVM/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "accel-matmul-tiling"

namespace mlir::accel {

// ===== Phase 1: Detection =====
void AccelMatmulTilingPass::findAccelMatmulTiles(
    func::FuncOp func, std::vector<Operation *> &matmulTiles) {
  func.walk([&](Operation *op) {
    // accel.matmul_tile operation을 찾음
    if (op->getName().getStringRef() == "accel.matmul_tile") {
      matmulTiles.push_back(op);
      LLVM_DEBUG(llvm::dbgs() << "Found accel.matmul_tile operation\n");
    }
  });
}

// ===== Phase 2: Memory Analysis =====
MemoryLayoutInfo AccelMatmulTilingPass::analyzeMemoryLayout(Value tensor) {
  // AlignmentAnalyzer를 사용하여 메모리 분석
  return AlignmentAnalyzer::analyzeValue(tensor);
}

AccelMatmulTilingPass::TileDimensions
AccelMatmulTilingPass::extractTileDimensions(Value tensor) {
  TileDimensions dims;

  if (auto memrefType = dyn_cast<MemRefType>(tensor.getType())) {
    auto shape = memrefType.getShape();

    // 2D 타일 (32x32 또는 64x64 등)
    if (shape.size() == 2) {
      dims.height = shape[0];
      dims.width = shape[1];
    }

    // Element size
    auto elementType = memrefType.getElementType();
    if (auto floatType = dyn_cast<FloatType>(elementType)) {
      dims.elementSize = floatType.getWidth() / 8;
    } else {
      dims.elementSize = 4; // default: f32
    }
  }

  return dims;
}

// ===== Phase 3: Alignment Check =====
bool AccelMatmulTilingPass::verifyAlignment(MemoryLayoutInfo &layout) {
  // 현재 정렬 상태 확인
  return (layout.currentAlignment % layout.requiredAlignment) == 0 &&
         layout.isContiguous;
}

Value AccelMatmulTilingPass::insertAlignmentPadding(
    Operation *op, MemoryLayoutInfo &layout, OpBuilder &builder) {
  // 정렬되지 않은 경우 패딩 처리

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(op);

  // 1. 필요한 정렬된 메모리 할당
  Location loc = op->getLoc();

  // 패딩 크기 계산
  int64_t paddingSize = layout.getPaddingSize();
  int64_t alignedSize = layout.totalBytes + paddingSize;

  LLVM_DEBUG(llvm::dbgs() << "Inserting padding: "
                          << "original=" << layout.totalBytes
                          << " padding=" << paddingSize << "\n");

  // 2. 정렬된 메모리 할당 (memref.alloc with alignment)
  auto alignedMemRefType = MemRefType::get(
      {alignedSize}, builder.getF32Type(),
      MemRefLayoutAttrInterface{}, 0);

  // 3. 데이터 복사 (원본 → 정렬된 버퍼)
  // Note: 실제 구현에서는 accel.copy_tile operation 사용
  SmallVector<Value> copyArgs;
  for (auto operand : op->getOperands()) {
    copyArgs.push_back(operand);
  }

  // 4. 정렬된 주소 반환 (placeholder)
  // 실제로는 memref에서 base pointer 추출하여 사용
  return op->getOperand(0);
}

// ===== Phase 4: Lowering =====
void AccelMatmulTilingPass::lowerToLLVMIntrinsic(Operation *op,
                                                   OpBuilder &builder) {
  // accel.matmul_tile → llvm.accel.matmul.32x32로 변환

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(op);
  Location loc = op->getLoc();

  // 1. 피연산자 준비
  Value lhs = op->getOperand(0); // A
  Value rhs = op->getOperand(1); // B

  // 2. LLVM pointer로 변환
  // (실제 구현에서는 memref.extract_aligned_pointer_as_index 사용)
  auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());

  // 3. LLVM Intrinsic 호출 생성
  // call i32 @llvm.accel.matmul.32x32(ptr %addrA, ptr %addrB, ptr %addrC)
  SmallVector<Value> intrinsicArgs = {lhs, rhs};
  auto resultType = builder.getI32Type();

  // 4. Intrinsic call 생성 (placeholder - 실제로는 LLVM::CallOp)
  LLVM_DEBUG(llvm::dbgs() << "Lowering to LLVM intrinsic\n");

  // op를 제거하거나 변환 (현재: 성능 분석용으로 유지)
}

// ===== Main Pass Logic =====
void AccelMatmulTilingPass::runOnOperation() {
  func::FuncOp func = getOperation();

  // Phase 1: 모든 accel.matmul_tile operations 찾기
  std::vector<Operation *> matmulTiles;
  findAccelMatmulTiles(func, matmulTiles);

  LLVM_DEBUG(llvm::dbgs() << "Found " << matmulTiles.size()
                          << " accel.matmul_tile operations\n");

  // Phase 2-4: 각 operation 처리
  OpBuilder builder(func.getContext());

  for (auto *op : matmulTiles) {
    LLVM_DEBUG(llvm::dbgs() << "Processing accel.matmul_tile at "
                            << op->getLoc() << "\n");

    // 메모리 레이아웃 분석
    Value lhs = op->getOperand(0);
    MemoryLayoutInfo lhsLayout = analyzeMemoryLayout(lhs);

    Value rhs = op->getOperand(1);
    MemoryLayoutInfo rhsLayout = analyzeMemoryLayout(rhs);

    // 타일 차원 추출
    auto tileDims = extractTileDimensions(lhs);

    // 정렬 검증
    bool lhsAligned = verifyAlignment(lhsLayout);
    bool rhsAligned = verifyAlignment(rhsLayout);

    // 보고서 출력
    printAnalysisReport(op, lhsLayout, !lhsAligned);

    // 패딩 필요시 처리
    if (!lhsAligned) {
      insertAlignmentPadding(op, lhsLayout, builder);
    }
    if (!rhsAligned) {
      insertAlignmentPadding(op, rhsLayout, builder);
    }

    // LLVM Intrinsic으로 lowering
    lowerToLLVMIntrinsic(op, builder);
  }

  LLVM_DEBUG(llvm::dbgs() << "AccelMatmulTilingPass completed\n");
}

// ===== Utility Methods =====
Value AccelMatmulTilingPass::createAlignedAddress(Value buffer, int64_t offset,
                                                   OpBuilder &builder) {
  // buffer에서 정렬된 주소 추출
  // 실제 구현: memref → ptr → aligned offset 계산
  return buffer;
}

LogicalResult AccelMatmulTilingPass::processFunction(func::FuncOp func) {
  // Phase 1-4를 통합하여 한 함수 처리
  runOnOperation();
  return success();
}

void AccelMatmulTilingPass::printAnalysisReport(Operation *op,
                                                 MemoryLayoutInfo &layout,
                                                 bool needsPadding) {
  llvm::outs() << "\n=== Memory Analysis Report ===\n";
  llvm::outs() << "Operation: " << op->getName() << " at " << op->getLoc()
               << "\n";
  llvm::outs() << "Total Size: " << layout.totalBytes << " bytes\n";
  llvm::outs() << "Contiguous: " << (layout.isContiguous ? "yes" : "no")
               << "\n";
  llvm::outs() << "Required Alignment: " << layout.requiredAlignment
               << " bytes\n";
  llvm::outs() << "Current Alignment: " << layout.currentAlignment << " bytes\n";
  llvm::outs() << "Needs Padding: " << (needsPadding ? "yes" : "no") << "\n";

  if (needsPadding) {
    llvm::outs() << "Padding Size: " << layout.getPaddingSize() << " bytes\n";
  }

  llvm::outs() << "Status: " << (layout.isSafe() ? "✅ SAFE" : "⚠️ NEEDS FIX")
               << "\n";
  llvm::outs() << "=============================\n\n";
}

// ===== MemoryAlignmentValidator 구현 =====
LogicalResult MemoryAlignmentValidator::validateFunctionMemoryAccess(
    func::FuncOp func) {
  bool allValid = true;

  func.walk([&](Operation *op) {
    if (op->getName().getStringRef() == "accel.matmul_tile") {
      if (failed(validateOperationMemoryAccess(op))) {
        allValid = false;
      }
    }
  });

  return success(allValid);
}

LogicalResult MemoryAlignmentValidator::validateOperationMemoryAccess(
    Operation *op) {
  // 각 operand의 메모리 안전성 검증
  for (auto operand : op->getOperands()) {
    auto layout = AlignmentAnalyzer::analyzeValue(operand);
    if (!layout.isSafe()) {
      return failure();
    }
  }
  return success();
}

MemoryAlignmentValidator::ValidationReport
MemoryAlignmentValidator::generateReport(Operation *op) {
  ValidationReport report;
  report.passedChecks = 0;
  report.failedChecks = 0;

  for (auto operand : op->getOperands()) {
    auto layout = AlignmentAnalyzer::analyzeValue(operand);
    if (layout.isSafe()) {
      report.passedChecks++;
    } else {
      report.failedChecks++;
    }
  }

  report.isValid = (report.failedChecks == 0);

  report.report = "Validation: " + std::to_string(report.passedChecks) +
                  "/" + std::to_string(report.passedChecks + report.failedChecks) +
                  " checks passed";

  return report;
}

// ===== AddressAlignmentPass 구현 =====
void AddressAlignmentPass::runOnOperation() {
  func::FuncOp func = getOperation();
  scanAndFixAlignment(func);
}

void AddressAlignmentPass::scanAndFixAlignment(func::FuncOp func) {
  func.walk([&](Operation *op) {
    if (op->getName().getStringRef() == "accel.matmul_tile") {
      auto report = MemoryAlignmentValidator::generateReport(op);
      if (!report.isValid) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Found misaligned memory access at " << op->getLoc()
                   << ": " << report.report << "\n");
      }
    }
  });
}

} // namespace mlir::accel
