#ifndef ACCEL_MEMORY_UTILS_H
#define ACCEL_MEMORY_UTILS_H

#include "mlir/IR/Value.h"
#include "mlir/IR/MemRefType.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>

namespace mlir::accel {

/// ============================================================
/// MemoryLayoutInfo: 메모리 레이아웃 분석 결과
/// ============================================================
struct MemoryLayoutInfo {
  /// 전체 메모리 크기 (bytes)
  int64_t totalBytes;

  /// Row-Major 레이아웃일 때 한 행의 stride
  int64_t rowMajorStride;

  /// 현재 메모리의 실제 정렬 상태
  /// (base_ptr + offset) % alignment == 0 인지 확인
  unsigned currentAlignment;

  /// 해당 하드웨어가 요구하는 정렬 기준
  unsigned requiredAlignment;

  /// 메모리가 연속적(contiguous)인가?
  bool isContiguous;

  /// 패딩이 필요한가?
  bool needsPadding;

  /// 정렬되지 않은 부분의 바이트 수
  int64_t misalignedBytes;

  // Constructor
  MemoryLayoutInfo()
      : totalBytes(0), rowMajorStride(0), currentAlignment(0),
        requiredAlignment(64), isContiguous(false), needsPadding(false),
        misalignedBytes(0) {}

  /// 이 레이아웃이 안전한가? (정렬되었고 연속적인가)
  bool isSafe() const {
    return isContiguous && (currentAlignment % requiredAlignment == 0);
  }

  /// 정렬에 필요한 패딩 크기
  int64_t getPaddingSize() const {
    if (currentAlignment % requiredAlignment == 0)
      return 0;
    unsigned remainder = currentAlignment % requiredAlignment;
    return requiredAlignment - remainder;
  }
};

/// ============================================================
/// AlignmentAnalyzer: 메모리 정렬 분석기
/// ============================================================
class AlignmentAnalyzer {
public:
  /// MemRefType으로부터 메모리 레이아웃 분석
  /// @param memref 분석할 MemRef 타입
  /// @return 분석 결과
  static MemoryLayoutInfo analyzeMemRefType(mlir::MemRefType memref);

  /// 값(Value)으로부터 메모리 레이아웃 분석
  /// @param value 분석할 MLIR value
  /// @return 분석 결과 (실패하면 needsPadding=true 반환)
  static MemoryLayoutInfo analyzeValue(mlir::Value value);

  /// 타일 크기에 따른 총 메모리 크기 계산
  /// @param tileHeight 타일의 높이
  /// @param tileWidth 타일의 너비
  /// @param elementSize 요소의 크기 (예: f32는 4)
  /// @return 총 메모리 크기 (bytes)
  static int64_t calculateTotalBytes(int64_t tileHeight, int64_t tileWidth,
                                      int64_t elementSize);

  /// Row-Major stride 계산
  /// @param tileWidth 타일의 너비
  /// @param elementSize 요소의 크기
  /// @return 한 행의 stride (bytes)
  static int64_t calculateRowMajorStride(int64_t tileWidth,
                                          int64_t elementSize);

  /// 하드웨어 스펙에 맞는 정렬 기준 반환
  /// @return 필요한 정렬 바이트 (보통 64, GPU 캐시 라인)
  static unsigned getRequiredAlignment();

  /// 연속성(contiguity) 검증
  /// @param memref MemRef 타입
  /// @return 연속적인 메모리인가?
  static bool isContiguous(mlir::MemRefType memref);
};

/// ============================================================
/// AddressAlignmentHelper: 주소 정렬 유틸리티
/// ============================================================
class AddressAlignmentHelper {
public:
  /// 두 주소의 정렬 상태를 비교
  /// @param addr1 첫 번째 주소
  /// @param addr2 두 번째 주소
  /// @param requiredAlignment 요구 정렬 기준
  /// @return 모두 정렬되었는가?
  static bool areAddressesAligned(uint64_t addr1, uint64_t addr2,
                                   unsigned requiredAlignment);

  /// 주소를 정렬된 주소로 올림(ceiling)
  /// @param addr 원본 주소
  /// @param alignment 정렬 기준
  /// @return 정렬된 주소 (>= addr)
  static uint64_t alignAddress(uint64_t addr, unsigned alignment);

  /// 주소의 정렬 여부 확인
  /// @param addr 확인할 주소
  /// @param alignment 정렬 기준
  /// @return 정렬되었는가?
  static bool isAddressAligned(uint64_t addr, unsigned alignment);

  /// 정렬되지 않은 경우 필요한 오프셋
  /// @param addr 원본 주소
  /// @param alignment 정렬 기준
  /// @return 필요한 오프셋 (0이면 이미 정렬됨)
  static uint64_t getAlignmentOffset(uint64_t addr, unsigned alignment);
};

/// ============================================================
/// TileMemoryLayout: 타일 메모리 계획
/// ============================================================
struct TileMemoryLayout {
  /// 타일 크기 (높이 x 너비)
  int64_t tileHeight;
  int64_t tileWidth;

  /// 요소 크기 (bytes)
  int64_t elementSize;

  /// 계산된 메모리 정보
  MemoryLayoutInfo layout;

  /// 패딩이 필요한 경우, 원본 메모리 오프셋
  int64_t originalOffset;

  /// 정렬된 메모리의 오프셋
  int64_t alignedOffset;

  TileMemoryLayout(int64_t h, int64_t w, int64_t elSize)
      : tileHeight(h), tileWidth(w), elementSize(elSize), originalOffset(0),
        alignedOffset(0) {
    layout.totalBytes = calculateTotalBytes(h, w, elSize);
    layout.rowMajorStride = calculateRowMajorStride(w, elSize);
    layout.requiredAlignment = AlignmentAnalyzer::getRequiredAlignment();
  }

  static int64_t calculateTotalBytes(int64_t h, int64_t w, int64_t es) {
    return h * w * es;
  }

  static int64_t calculateRowMajorStride(int64_t w, int64_t es) {
    return w * es;
  }
};

} // namespace mlir::accel

#endif // ACCEL_MEMORY_UTILS_H
