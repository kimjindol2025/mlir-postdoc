#ifndef ACCEL_HARDWARE_AWARE_TILING_PASS_H
#define ACCEL_HARDWARE_AWARE_TILING_PASS_H

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include <cstdint>
#include <vector>

namespace mlir::accel {

/// ============================================================
/// HardwareSpec: 하드웨어 명세서
/// ============================================================
struct HardwareSpec {
  // 메모리 제약
  int64_t sramBytes;              // SRAM 크기 (bytes)
  int64_t l1CacheBytes;           // L1 캐시 크기
  int64_t l2CacheBytes;           // L2 캐시 크기

  // 성능 특성
  int64_t memoryBandwidth;        // DRAM 대역폭 (GB/s)
  int64_t computeUnits;           // 연산 유닛 개수
  int64_t peakFLOPS;              // 피크 성능 (GFLOPS)

  // 정렬 요구사항
  unsigned requiredAlignment;     // 메모리 정렬 (64, 128 등)

  // 생성자: 기본값 (NVIDIA GPU 기준)
  HardwareSpec()
      : sramBytes(128 * 1024),    // 128KB (V100 기준)
        l1CacheBytes(32 * 1024),  // 32KB
        l2CacheBytes(2 * 1024 * 1024), // 2MB
        memoryBandwidth(256),      // 256 GB/s
        computeUnits(80),          // 80 SMs
        peakFLOPS(7000),           // 7 TFLOPS
        requiredAlignment(64) {}

  // 다른 하드웨어로 커스터마이징
  static HardwareSpec getAMDGPU() {
    HardwareSpec spec;
    spec.sramBytes = 96 * 1024;    // 96KB
    spec.memoryBandwidth = 484;    // 484 GB/s (RDNA2)
    return spec;
  }

  static HardwareSpec getTPU() {
    HardwareSpec spec;
    spec.sramBytes = 64 * 1024;    // 64KB SRAM per core
    spec.memoryBandwidth = 50;     // 50 GB/s local
    return spec;
  }

  static HardwareSpec getCPU() {
    HardwareSpec spec;
    spec.sramBytes = 32 * 1024;    // 32KB L1
    spec.memoryBandwidth = 20;     // 20 GB/s
    return spec;
  }
};

/// ============================================================
/// TileAnalysis: 타일링 분석 결과
/// ============================================================
struct TileAnalysis {
  // 입력
  int64_t tensorDim0;             // 첫 번째 차원 크기
  int64_t tensorDim1;             // 두 번째 차원 크기
  int64_t elementSize;            // 요소 크기 (bytes)

  // 계산 결과
  int64_t optimalTileSize;        // 최적 타일 크기
  int64_t memorySramUsage;        // SRAM 사용량
  int64_t numTiles;               // 타일 개수
  double memoryReuseFactor;       // 메모리 재사용 비율
  double cacheHitRate;            // 캐시 히트율
  double expectedLatency;         // 예상 지연시간 (ms)

  // 안전성 검증
  bool isSafeForSRAM;             // SRAM에 안전한가?
  bool needsDoubleBuffering;      // 더블 버퍼링 필요?
  bool isOptimal;                 // 최적 타일 크기인가?

  TileAnalysis()
      : tensorDim0(0), tensorDim1(0), elementSize(4),
        optimalTileSize(0), memorySramUsage(0), numTiles(0),
        memoryReuseFactor(0.0), cacheHitRate(0.0),
        expectedLatency(0.0), isSafeForSRAM(false),
        needsDoubleBuffering(false), isOptimal(false) {}

  /// 이 분석이 유효한가?
  bool isValid() const {
    return optimalTileSize > 0 && isSafeForSRAM;
  }

  /// 분석 결과를 텍스트로 출력
  void print() const;
};

/// ============================================================
/// TileCalculator: 최적 타일 크기 계산기
/// ============================================================
class TileCalculator {
public:
  /// 메모리 제약 기반 최적 타일 크기 계산
  /// @param sramBytes SRAM 크기
  /// @param elementSize 요소 크기
  /// @param numInputs 입력 텐서 개수 (행렬곱은 3)
  /// @return 최적 타일 크기 (2의 거듭제곱)
  static int64_t calculateOptimalTileSize(int64_t sramBytes,
                                          int64_t elementSize,
                                          int numInputs);

  /// 타일 크기에 따른 메모리 사용량 계산
  /// @param tileSize 타일 크기
  /// @param elementSize 요소 크기
  /// @param numInputs 입력 개수
  /// @return 메모리 사용량 (bytes)
  static int64_t calculateMemoryUsage(int64_t tileSize, int64_t elementSize,
                                       int numInputs);

  /// 메모리 재사용 비율 계산
  /// @param tileSize 타일 크기
  /// @param tensorSize 전체 텐서 크기
  /// @return 재사용 비율 (0.0 ~ 1.0)
  static double calculateMemoryReuse(int64_t tileSize, int64_t tensorSize);

  /// 캐시 히트율 예측
  /// @param tileSize 타일 크기
  /// @param l1CacheSize L1 캐시 크기
  /// @return 예상 히트율 (0.0 ~ 1.0)
  static double predictCacheHitRate(int64_t tileSize, int64_t l1CacheSize);

  /// 예상 지연시간 계산 (ms)
  static double estimateLatency(const TileAnalysis &analysis,
                                const HardwareSpec &spec);
};

/// ============================================================
/// HardwareAwareTilingPass: 메인 Pass
/// ============================================================
class HardwareAwareTilingPass
    : public PassWrapper<HardwareAwareTilingPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HardwareAwareTilingPass)

  StringRef getArgument() const final { return "accel-hardware-aware-tiling"; }
  StringRef getDescription() const final {
    return "Apply hardware-aware tiling based on SRAM constraints";
  }

  // 옵션: 하드웨어 타입 선택 (기본: NVIDIA GPU)
  Option<std::string> hwType{
      *this, "hw-type",
      llvm::cl::desc("Hardware type (nvidia, amd, tpu, cpu)"),
      llvm::cl::init("nvidia")};

  void runOnOperation() override;

private:
  // 하드웨어 명세서
  HardwareSpec hwSpec;

  /// Phase 1: 모든 linalg operations 탐지
  void findLinalgOps(func::FuncOp func,
                     std::vector<Operation *> &linalgOps);

  /// Phase 2: 각 operation의 타일링 분석
  TileAnalysis analyzeOperation(Operation *op);

  /// Phase 3: 타일 크기 최적화
  TileAnalysis optimizeTileSize(Operation *op, const HardwareSpec &spec);

  /// Phase 4: 실제 타일링 적용
  LogicalResult applyTiling(Operation *op, const TileAnalysis &analysis,
                            OpBuilder &builder);

  /// 유틸리티: 텐서 차원 추출
  std::vector<int64_t> extractTensorDimensions(Value tensor);

  /// 유틸리티: 요소 크기 추출
  int64_t getElementSize(Type elementType);

  /// 유틸리티: 보고서 출력
  void printAnalysisReport(Operation *op, const TileAnalysis &analysis);
};

/// ============================================================
/// DoubleBufferingAnalyzer: 더블 버퍼링 분석
/// ============================================================
class DoubleBufferingAnalyzer {
public:
  /// 더블 버퍼링이 필요한가?
  /// @param tileSize 타일 크기
  /// @param latency 메모리 지연
  /// @param computeTime 연산 시간
  /// @return true if beneficial
  static bool shouldApplyDoubleBuffering(int64_t tileSize, double latency,
                                          double computeTime);

  /// async.token을 사용한 비동기 스케줄링
  static void scheduleAsyncLoad(Operation *op, int64_t tileSize,
                                OpBuilder &builder);
};

} // namespace mlir::accel

#endif // ACCEL_HARDWARE_AWARE_TILING_PASS_H
