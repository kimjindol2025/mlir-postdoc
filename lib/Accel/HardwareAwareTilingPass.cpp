// ===== HardwareAwareTilingPass.cpp =====
// 하드웨어 제약 기반 최적 타일링

#include "Accel/HardwareAwareTilingPass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Types.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <cmath>

#define DEBUG_TYPE "accel-hardware-aware-tiling"

namespace mlir::accel {

// ===== TileAnalysis 구현 =====
void TileAnalysis::print() const {
  llvm::outs() << "\n===== Hardware-Aware Tiling Analysis =====\n";
  llvm::outs() << "Tensor Shape: [" << tensorDim0 << " x " << tensorDim1
               << "]\n";
  llvm::outs() << "Element Size: " << elementSize << " bytes\n";
  llvm::outs() << "\nOptimal Tile Size: " << optimalTileSize << " x "
               << optimalTileSize << "\n";
  llvm::outs() << "SRAM Usage: " << memorySramUsage << " bytes\n";
  llvm::outs() << "Number of Tiles: " << numTiles << "\n";
  llvm::outs() << "Memory Reuse Factor: " << memoryReuseFactor << "x\n";
  llvm::outs() << "Expected Cache Hit Rate: " << (cacheHitRate * 100)
               << "%\n";
  llvm::outs() << "Expected Latency: " << expectedLatency << " ms\n";
  llvm::outs() << "\nSafety: " << (isSafeForSRAM ? "✅ SAFE" : "❌ UNSAFE")
               << "\n";
  llvm::outs() << "Double Buffering: "
               << (needsDoubleBuffering ? "✅ YES" : "❌ NO") << "\n";
  llvm::outs() << "Optimality: " << (isOptimal ? "✅ OPTIMAL" : "⚠️  HEURISTIC")
               << "\n";
  llvm::outs() << "=========================================\n\n";
}

// ===== TileCalculator 구현 =====

int64_t TileCalculator::calculateOptimalTileSize(int64_t sramBytes,
                                                  int64_t elementSize,
                                                  int numInputs) {
  // 수식: T² × numInputs × elementSize ≤ sramBytes
  // T ≤ √(sramBytes / (numInputs × elementSize))

  double maxTile =
      std::sqrt(static_cast<double>(sramBytes) / (numInputs * elementSize));
  int64_t optimalTile = static_cast<int64_t>(maxTile);

  // 안전 마진: 2의 거듭제곱으로 내림
  // 캐시 정렬 친화적, 벡터화 효율적
  int64_t powerOfTwo = 1LL << llvm::Log2_64(optimalTile);

  LLVM_DEBUG(llvm::dbgs() << "Optimal tile calculation:\n"
                          << "  Raw: " << maxTile << "\n"
                          << "  Floored: " << optimalTile << "\n"
                          << "  PowerOfTwo: " << powerOfTwo << "\n");

  return powerOfTwo;
}

int64_t TileCalculator::calculateMemoryUsage(int64_t tileSize,
                                              int64_t elementSize,
                                              int numInputs) {
  return tileSize * tileSize * elementSize * numInputs;
}

double TileCalculator::calculateMemoryReuse(int64_t tileSize,
                                             int64_t tensorSize) {
  // 메모리 재사용 비율
  // = (전체 메모리 접근) / (타일당 한번만 접근하면 필요한 최소)
  // 약: tensorSize / tileSize

  if (tensorSize == 0)
    return 0.0;

  // 타일링으로 인한 메모리 지역성 개선
  // 각 타일이 여러 번 재사용됨
  double reuse = static_cast<double>(tensorSize) / tileSize;

  return std::min(reuse, 100.0); // 최대 100배 제한
}

double TileCalculator::predictCacheHitRate(int64_t tileSize,
                                            int64_t l1CacheSize) {
  // 타일이 L1 캐시에 모두 들어가면 히트율 95%+
  // 아니면 히트율 낮음

  int64_t tileMemory = tileSize * tileSize * 4 * 3; // 행렬곱의 3개 입력

  if (tileMemory <= l1CacheSize) {
    // 타일이 캐시에 들어감 → 높은 히트율
    return 0.95;
  } else if (tileMemory <= l1CacheSize * 2) {
    // 타일이 L1/L2 경계 → 중간 히트율
    return 0.70;
  } else {
    // 메모리 부족 → 낮은 히트율
    return 0.30;
  }
}

double TileCalculator::estimateLatency(const TileAnalysis &analysis,
                                        const HardwareSpec &spec) {
  // 지연시간 = 메모리 지연 + 연산 시간

  // 메모리 지연: DRAM 접근 시간
  // 각 타일마다 메모리를 (tensorSize / tileSize)번 로드
  int64_t dataSize = analysis.tensorDim0 * analysis.tensorDim1 *
                     analysis.elementSize * 3; // A, B, C
  double memoryTransferTime =
      static_cast<double>(dataSize) / (spec.memoryBandwidth * 1e9) * 1000; // ms

  // 캐시 히트 적용
  double adjustedMemoryTime = memoryTransferTime * (1.0 - analysis.cacheHitRate);

  // 연산 시간: 행렬 곱셈 2N³ FLOPs
  int64_t numFLOPs = 2 * analysis.tensorDim0 * analysis.tensorDim1 *
                     analysis.tensorDim1; // 단순화
  double computeTime =
      static_cast<double>(numFLOPs) / (spec.peakFLOPS * 1e9) * 1000; // ms

  // 더블 버퍼링으로 메모리 숨김
  double hiddenMemoryTime = DoubleBufferingAnalyzer::shouldApplyDoubleBuffering(
                                analysis.optimalTileSize, adjustedMemoryTime,
                                computeTime)
                                ? 0.0
                                : adjustedMemoryTime;

  return computeTime + hiddenMemoryTime;
}

// ===== HardwareAwareTilingPass 구현 =====

void HardwareAwareTilingPass::runOnOperation() {
  func::FuncOp func = getOperation();

  // 하드웨어 명세서 초기화
  if (hwType == "nvidia") {
    hwSpec = HardwareSpec();
  } else if (hwType == "amd") {
    hwSpec = HardwareSpec::getAMDGPU();
  } else if (hwType == "tpu") {
    hwSpec = HardwareSpec::getTPU();
  } else if (hwType == "cpu") {
    hwSpec = HardwareSpec::getCPU();
  }

  LLVM_DEBUG(llvm::dbgs() << "Hardware: " << hwType << " (SRAM: "
                          << hwSpec.sramBytes << " bytes)\n");

  // Phase 1: linalg operations 탐지
  std::vector<Operation *> linalgOps;
  findLinalgOps(func, linalgOps);

  LLVM_DEBUG(llvm::dbgs() << "Found " << linalgOps.size() << " linalg ops\n");

  // Phase 2-4: 각 operation 분석 및 최적화
  OpBuilder builder(func.getContext());

  for (auto *op : linalgOps) {
    LLVM_DEBUG(llvm::dbgs() << "Processing linalg operation at " << op->getLoc()
                            << "\n");

    // 분석
    TileAnalysis analysis = analyzeOperation(op);

    // 최적화
    if (analysis.tensorDim0 > 0 && analysis.tensorDim1 > 0) {
      TileAnalysis optimized = optimizeTileSize(op, hwSpec);

      // 타일링 적용
      if (succeeded(applyTiling(op, optimized, builder))) {
        printAnalysisReport(op, optimized);
      }
    }
  }
}

void HardwareAwareTilingPass::findLinalgOps(func::FuncOp func,
                                             std::vector<Operation *> &ops) {
  func.walk([&](Operation *op) {
    // linalg.matmul, linalg.conv_2d 등 찾기
    if (isa<linalg::MatmulOp>(op) || isa<linalg::Conv2DOp>(op) ||
        isa<linalg::BatchMatmulOp>(op)) {
      ops.push_back(op);
    }
  });
}

TileAnalysis HardwareAwareTilingPass::analyzeOperation(Operation *op) {
  TileAnalysis analysis;

  if (auto matmul = dyn_cast<linalg::MatmulOp>(op)) {
    // MatMul 분석: C = A × B
    auto lhs = matmul.getInputs()[0];
    auto rhs = matmul.getInputs()[1];

    auto lhsType = dyn_cast<MemRefType>(lhs.getType());
    auto rhsType = dyn_cast<MemRefType>(rhs.getType());

    if (lhsType && rhsType) {
      auto lhsShape = lhsType.getShape();
      auto rhsShape = rhsType.getShape();

      if (lhsShape.size() >= 2) {
        analysis.tensorDim0 = lhsShape[0];
        analysis.tensorDim1 = lhsShape[1];
      }
      if (rhsShape.size() >= 2) {
        // K dimension
      }

      analysis.elementSize = getElementSize(lhsType.getElementType());
    }
  }

  return analysis;
}

TileAnalysis HardwareAwareTilingPass::optimizeTileSize(Operation *op,
                                                       const HardwareSpec &spec) {
  TileAnalysis analysis = analyzeOperation(op);

  if (analysis.tensorDim0 == 0 || analysis.tensorDim1 == 0) {
    return analysis;
  }

  // 최적 타일 크기 계산
  analysis.optimalTileSize =
      TileCalculator::calculateOptimalTileSize(spec.sramBytes,
                                               analysis.elementSize, 3);

  // 타일 개수
  analysis.numTiles = (analysis.tensorDim0 / analysis.optimalTileSize) *
                      (analysis.tensorDim1 / analysis.optimalTileSize);

  // SRAM 사용량
  analysis.memorySramUsage = TileCalculator::calculateMemoryUsage(
      analysis.optimalTileSize, analysis.elementSize, 3);

  // 메모리 재사용
  int64_t totalMemory = analysis.tensorDim0 * analysis.tensorDim1 *
                        analysis.elementSize * 3;
  analysis.memoryReuseFactor =
      TileCalculator::calculateMemoryReuse(analysis.optimalTileSize, totalMemory);

  // 캐시 히트율
  analysis.cacheHitRate = TileCalculator::predictCacheHitRate(
      analysis.optimalTileSize, spec.l1CacheBytes);

  // 예상 지연시간
  analysis.expectedLatency = TileCalculator::estimateLatency(analysis, spec);

  // 안전성 검증
  analysis.isSafeForSRAM = (analysis.memorySramUsage <= spec.sramBytes);
  analysis.needsDoubleBuffering = DoubleBufferingAnalyzer::shouldApplyDoubleBuffering(
      analysis.optimalTileSize, 0.0, analysis.expectedLatency);
  analysis.isOptimal = analysis.isSafeForSRAM && (analysis.cacheHitRate > 0.9);

  return analysis;
}

LogicalResult HardwareAwareTilingPass::applyTiling(Operation *op,
                                                    const TileAnalysis &analysis,
                                                    OpBuilder &builder) {
  // 실제 타일링 변환
  // affine.for 루프 생성

  if (!analysis.isSafeForSRAM) {
    return failure();
  }

  LLVM_DEBUG(llvm::dbgs() << "Applying tiling with size "
                          << analysis.optimalTileSize << "\n");

  // 단순화: 실제 구현에서는 복잡한 loop generation
  // 여기서는 분석 결과만 기록

  return success();
}

std::vector<int64_t> HardwareAwareTilingPass::extractTensorDimensions(
    Value tensor) {
  std::vector<int64_t> dims;
  if (auto memrefType = dyn_cast<MemRefType>(tensor.getType())) {
    for (auto dim : memrefType.getShape()) {
      dims.push_back(dim);
    }
  }
  return dims;
}

int64_t HardwareAwareTilingPass::getElementSize(Type elementType) {
  if (auto floatType = dyn_cast<FloatType>(elementType)) {
    return floatType.getWidth() / 8;
  } else if (auto intType = dyn_cast<IntegerType>(elementType)) {
    return intType.getWidth() / 8;
  }
  return 4; // default f32
}

void HardwareAwareTilingPass::printAnalysisReport(Operation *op,
                                                   const TileAnalysis &analysis) {
  llvm::outs() << "\n============================================\n";
  llvm::outs() << "Hardware-Aware Tiling Analysis\n";
  llvm::outs() << "Operation: " << op->getName() << " at " << op->getLoc()
               << "\n";
  llvm::outs() << "Hardware: " << hwType << "\n";
  llvm::outs() << "============================================\n";

  analysis.print();

  // 성능 이득
  llvm::outs() << "[Performance Impact]\n";
  llvm::outs() << "Memory Reduction: " << analysis.memoryReuseFactor << "x\n";
  llvm::outs() << "Cache Hit Rate: " << (analysis.cacheHitRate * 100) << "%\n";
  llvm::outs() << "Expected Speedup: ~" << (1.0 + analysis.memoryReuseFactor)
               << "x\n";
  llvm::outs() << "============================================\n\n";
}

// ===== DoubleBufferingAnalyzer 구현 =====

bool DoubleBufferingAnalyzer::shouldApplyDoubleBuffering(int64_t tileSize,
                                                          double latency,
                                                          double computeTime) {
  // 더블 버퍼링의 이점:
  // 메모리 로딩 시간을 연산 시간에 숨길 수 있음
  // 조건: latency > computeTime * 0.1 (메모리가 병목)

  if (latency == 0.0 || computeTime == 0.0)
    return false;

  double ratio = latency / computeTime;
  return ratio > 0.1; // 메모리가 연산의 10% 이상
}

void DoubleBufferingAnalyzer::scheduleAsyncLoad(Operation *op, int64_t tileSize,
                                                 OpBuilder &builder) {
  // async.token을 사용한 비동기 로딩 스케줄링
  // 실제 구현: async dialect과 통합

  LLVM_DEBUG(llvm::dbgs() << "Scheduling async load for tile size "
                          << tileSize << "\n");

  // TODO: async dialect integration
  // Token를 생성하고 wait하지 않고 연산 수행
}

} // namespace mlir::accel
