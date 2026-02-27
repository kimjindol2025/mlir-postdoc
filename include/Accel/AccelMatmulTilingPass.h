#ifndef ACCEL_MATMUL_TILING_PASS_H
#define ACCEL_MATMUL_TILING_PASS_H

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "Accel/MemoryUtils.h"

#include <vector>

namespace mlir::accel {

/// ============================================================
/// AccelMatmulTilingPass
/// ============================================================
/// Step 2→3 전환: Affine Loops → LLVM Intrinsics
/// 핵심: 주소 정렬(Address Alignment) 검증 및 패딩 처리
/// ============================================================
class AccelMatmulTilingPass
    : public PassWrapper<AccelMatmulTilingPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AccelMatmulTilingPass)

  StringRef getArgument() const final { return "accel-matmul-tiling"; }
  StringRef getDescription() const final {
    return "Lower accel.matmul_tile to LLVM intrinsics with alignment "
           "verification";
  }

  void runOnOperation() override;

private:
  /// ===== Phase 1: Detection =====
  /// "accel.matmul_tile" operation을 찾아서 리스트에 추가
  void findAccelMatmulTiles(func::FuncOp func,
                            std::vector<Operation *> &matmulTiles);

  /// ===== Phase 2: Memory Analysis =====
  /// 각 operand의 메모리 레이아웃 분석
  MemoryLayoutInfo analyzeMemoryLayout(Value tensor);

  /// 타일의 차원 추출
  struct TileDimensions {
    int64_t height;   // 행
    int64_t width;    // 열
    int64_t elementSize;  // bytes (f32 = 4, f64 = 8)
  };

  TileDimensions extractTileDimensions(Value tensor);

  /// ===== Phase 3: Alignment Check =====
  /// 메모리가 정렬되었는지 확인
  /// @return true if aligned, false if padding needed
  bool verifyAlignment(MemoryLayoutInfo &layout);

  /// 정렬이 필요한 경우 패딩 삽입
  /// @param op accel.matmul_tile operation
  /// @param layout 메모리 레이아웃 정보
  /// @return 정렬된 메모리의 주소
  Value insertAlignmentPadding(Operation *op, MemoryLayoutInfo &layout,
                                OpBuilder &builder);

  /// ===== Phase 4: Lowering =====
  /// LLVM Intrinsic으로 변환
  void lowerToLLVMIntrinsic(Operation *op, OpBuilder &builder);

  /// 정렬된 주소를 LLVM ptr로 변환
  Value createAlignedAddress(Value buffer, int64_t offset,
                              OpBuilder &builder);

  /// ===== Utility Methods =====
  /// 함수의 모든 accel ops 처리
  LogicalResult processFunction(func::FuncOp func);

  /// 보고서 출력
  void printAnalysisReport(Operation *op, MemoryLayoutInfo &layout,
                            bool needsPadding);
};

/// ============================================================
/// MemoryAlignmentValidator: 검증 전문 클래스
/// ============================================================
class MemoryAlignmentValidator {
public:
  /// 전체 변환의 안전성을 검증
  /// @param func 검증할 함수
  /// @return 모든 메모리 접근이 안전한가?
  static LogicalResult validateFunctionMemoryAccess(func::FuncOp func);

  /// 단일 operation의 메모리 안전성 검증
  /// @param op 검증할 operation
  /// @return 안전한가?
  static LogicalResult validateOperationMemoryAccess(Operation *op);

  /// 상세 검증 리포트 생성
  struct ValidationReport {
    bool isValid;           // 전체 검증 결과
    unsigned passedChecks;  // 통과한 체크 수
    unsigned failedChecks;  // 실패한 체크 수
    std::string report;     // 상세 메시지
  };

  static ValidationReport generateReport(Operation *op);
};

/// ============================================================
/// AddressAlignmentPass: 실험적 Phase 3 전용 Pass
/// ============================================================
/// 주소 정렬만 전담하는 Pass (필요시 분리)
/// ============================================================
class AddressAlignmentPass
    : public PassWrapper<AddressAlignmentPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AddressAlignmentPass)

  StringRef getArgument() const final { return "accel-address-alignment"; }
  StringRef getDescription() const final {
    return "Verify and fix address alignment for memory operations";
  }

  void runOnOperation() override;

private:
  void scanAndFixAlignment(func::FuncOp func);
};

} // namespace mlir::accel

#endif // ACCEL_MATMUL_TILING_PASS_H
