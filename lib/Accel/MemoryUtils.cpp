// ===== MemoryUtils.cpp: 메모리 분석 구현 =====

#include "Accel/MemoryUtils.h"
#include "mlir/IR/MemRefType.h"
#include "mlir/IR/Types.h"
#include "llvm/Support/MathExtras.h"

namespace mlir::accel {

// ===== AlignmentAnalyzer 구현 =====

MemoryLayoutInfo AlignmentAnalyzer::analyzeMemRefType(MemRefType memref) {
  MemoryLayoutInfo info;

  // 1. 메모리 크기 계산
  int64_t numElements = 1;
  for (auto dim : memref.getShape()) {
    if (dim == ShapedType::kDynamic) {
      // 동적 크기는 보수적으로 처리
      numElements = -1;
      break;
    }
    numElements *= dim;
  }

  auto elementType = memref.getElementType();
  int64_t elementSize = 0;
  if (auto floatType = dyn_cast<FloatType>(elementType)) {
    elementSize = floatType.getWidth() / 8; // bits to bytes
  } else if (auto intType = dyn_cast<IntegerType>(elementType)) {
    elementSize = intType.getWidth() / 8;
  } else {
    elementSize = 8; // default
  }

  if (numElements > 0) {
    info.totalBytes = numElements * elementSize;
  } else {
    info.totalBytes = -1; // unknown
  }

  // 2. Row-Major stride 계산 (2D의 경우)
  auto shape = memref.getShape();
  if (shape.size() == 2) {
    info.rowMajorStride = shape.back() * elementSize;
  }

  // 3. 연속성 검증
  auto layout = memref.getLayout();
  if (auto identity = dyn_cast<AffineMapAttr>(layout)) {
    // Identity map = contiguous
    info.isContiguous = true;
  } else {
    // 복잡한 레이아웃 = 비연속적일 가능성
    info.isContiguous = false;
  }

  // 4. 정렬 요구사항
  info.requiredAlignment = getRequiredAlignment();
  info.currentAlignment = 0; // 분석 단계에서는 알 수 없음

  // 5. 패딩 필요성 판단
  info.needsPadding = !info.isContiguous;

  return info;
}

MemoryLayoutInfo AlignmentAnalyzer::analyzeValue(Value value) {
  if (auto memrefType = dyn_cast<MemRefType>(value.getType())) {
    return analyzeMemRefType(memrefType);
  }

  // 타입 추론 실패
  MemoryLayoutInfo info;
  info.needsPadding = true;
  info.isContiguous = false;
  return info;
}

int64_t AlignmentAnalyzer::calculateTotalBytes(int64_t tileHeight,
                                                int64_t tileWidth,
                                                int64_t elementSize) {
  return tileHeight * tileWidth * elementSize;
}

int64_t AlignmentAnalyzer::calculateRowMajorStride(int64_t tileWidth,
                                                    int64_t elementSize) {
  return tileWidth * elementSize;
}

unsigned AlignmentAnalyzer::getRequiredAlignment() {
  // GPU의 일반적인 캐시 라인 크기
  // NVIDIA: 128 bytes, 하지만 보수적으로 64로 설정
  return 64;
}

bool AlignmentAnalyzer::isContiguous(MemRefType memref) {
  // MemRef가 연속적인 메모리 배치인가?
  // 이상적으로는 strides를 확인해야 함

  auto shape = memref.getShape();
  if (shape.empty())
    return false;

  // 단순화: 일반적인 경우만 처리
  // 정확한 구현은 stride 정보 필요
  return true; // for now, assume contiguous
}

// ===== AddressAlignmentHelper 구현 =====

bool AddressAlignmentHelper::areAddressesAligned(uint64_t addr1, uint64_t addr2,
                                                  unsigned requiredAlignment) {
  return isAddressAligned(addr1, requiredAlignment) &&
         isAddressAligned(addr2, requiredAlignment);
}

uint64_t AddressAlignmentHelper::alignAddress(uint64_t addr,
                                               unsigned alignment) {
  // addr를 alignment의 배수로 올림
  return llvm::alignTo(addr, alignment);
}

bool AddressAlignmentHelper::isAddressAligned(uint64_t addr,
                                               unsigned alignment) {
  return (addr % alignment) == 0;
}

uint64_t AddressAlignmentHelper::getAlignmentOffset(uint64_t addr,
                                                     unsigned alignment) {
  if (isAddressAligned(addr, alignment))
    return 0;
  return alignment - (addr % alignment);
}

} // namespace mlir::accel
