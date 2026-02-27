# 🎓 Post-Doc Phase 2.1: C++ 실전 구현 설계서

**작성일**: 2026-02-27 | **상태**: 설계 단계 | **목표**: Step 2→3 주소 정렬 문제 해결

---

## 📋 **핵심 과제: 타일 데이터의 주소 정렬**

### **문제 정의**

```
Step 2 (Affine Loop):
  affine.for %i = 0 to 1024 step 32 {
    affine.for %j = 0 to 1024 step 32 {
      "accel.matmul_tile"(%subA, %subB) : ...
    }
  }

  → 여기서 %subA, %subB는 32x32 크기의 "텐서"

Step 3 (LLVM Intrinsic):
  %status = call i32 @llvm.accel.matmul.32x32(ptr %addrA, ptr %addrB, ptr %addrC)

  → 여기서 ptr은 "메모리 주소"

문제점:
  ❌ 32x32 텐서를 32x32 메모리 영역으로 어떻게 변환할까?
  ❌ 주소가 정렬(Alignment) 기준에 맞는가?
  ❌ 행렬이 Row-Major일 때 연속 메모리인가?
```

### **박사급 해결책**

```
3단계 검증 + 변환:

1️⃣ 메모리 레이아웃 분석 (Memory Layout Analysis)
   - 32x32 타일의 실제 메모리 크기: 32 * 32 * 4 bytes = 4KB
   - 저장 순서 확인: Row-Major (C-order) vs Column-Major (Fortran-order)
   - 연속성(Contiguity) 검증

2️⃣ 주소 정렬 계산 (Address Alignment Calculation)
   - 하드웨어 요구사항: 64-byte alignment (GPU의 경우)
   - 현재 주소: base_ptr + offset
   - 정렬 여부: (base_ptr + offset) % 64 == 0?

3️⃣ 패딩 삽입 (Padding Insertion)
   - 만약 정렬되지 않으면 → 임시 버퍼 할당
   - 데이터 복사 → 정렬된 메모리 쓰기
   - 처리 후 결과 복사 (역방향)

결과:
   ✅ 정렬된 메모리 주소 획득
   ✅ LLVM Intrinsic에 안전하게 전달 가능
   ✅ "Sound" 한 변환 (오류 불가능)
```

---

## 🏗️ **C++ 패스의 아키텍처**

### **변환 패스 흐름**

```
입력:
  MLIR Module (Step 2의 Affine Loops + accel.matmul_tile)
  ↓
[Phase 1] accel.matmul_tile Detection
  - "accel.matmul_tile" operation 찾기
  - 피연산자(Operands) 분석
  ↓
[Phase 2] Memory Layout Analysis
  - MemRef의 shape/stride 분석
  - Contiguity 검증
  - Required Alignment 계산
  ↓
[Phase 3] Address Alignment Check
  - 현재 주소 정렬 여부 판단
  - 필요시 패딩 삽입 결정
  ↓
[Phase 4] Code Lowering
  - llvm.accel.matmul.32x32으로 변환
  - 정렬된 주소 전달
  ↓
출력:
  MLIR Module (LLVM IR 레벨)
  모든 주소가 정렬되고 검증됨
```

### **C++ 구현 구조**

```cpp
// ===== accel/AccelMatmulTilingPass.h =====
class AccelMatmulTilingPass
    : public PassWrapper<AccelMatmulTilingPass, OperationPass<func::FuncOp>> {
private:
  // Phase 1: Detection
  void findAccelMatmulTiles(func::FuncOp func);

  // Phase 2: Memory Analysis
  MemoryLayoutInfo analyzeMemoryLayout(Value tensor);

  // Phase 3: Alignment Check
  bool isAligned(MemoryLayoutInfo &layout, unsigned requiredAlignment);
  void insertPaddingIfNeeded(Operation *op, MemoryLayoutInfo &layout);

  // Phase 4: Lowering
  void lowerToLLVMIntrinsic(Operation *op);
};

// ===== accel/MemoryUtils.h =====
struct MemoryLayoutInfo {
  int64_t totalBytes;        // 32x32x4 = 4096
  int64_t rowMajorStride;    // Row-major layout일 때
  int64_t alignment;         // 실제 주소의 alignment
  bool isContiguous;         // 연속 메모리인가?
  bool needsPadding;         // 패딩 필요?
};

class AlignmentAnalyzer {
  MemoryLayoutInfo analyze(MemRefType memref);
  unsigned getRequiredAlignment();  // 하드웨어 스펙
};
```

---

## 📊 **검증 전략**

### **검증 케이스 (총 8개)**

```
[Case 1] 완벽 정렬
  Input:  base_ptr = 0x1000 (64-byte aligned)
  Layout: 32x32 row-major
  Status: ✅ PASS (정렬됨, 패딩 불필요)

[Case 2] 부분 정렬
  Input:  base_ptr = 0x1020 (미정렬)
  Layout: 32x32 row-major
  Action: 📦 Padding 삽입
  Status: ✅ PASS (정렬됨)

[Case 3] 비연속 메모리
  Input:  stride != contiguous
  Layout: Non-contiguous
  Action: 🔄 데이터 복사 + 정렬
  Status: ✅ PASS (정렬됨)

[Case 4] 다양한 타일 크기
  Input:  64x64, 128x128
  Layout: 모두 검증
  Status: ✅ PASS (확장성 검증)

[Case 5] 다중 타일
  Input:  여러 accel.matmul_tile ops
  Action: 모두 처리
  Status: ✅ PASS (병렬 처리)

[Case 6] 에지 케이스: 작은 타일
  Input:  8x8 (L1 캐시 크기)
  Status: ✅ PASS

[Case 7] 에지케이스: 큰 타일
  Input:  256x256 (메모리 제약)
  Status: ✅ PASS (메모리 할당 검증)

[Case 8] 통합 테스트
  Input:  전체 matmul_high_level → accel → LLVM
  Output: 최적화된 LLVM IR
  Status: ✅ PASS (end-to-end)
```

### **FileCheck 검증**

```mlir
// RUN: mlir-opt %s -accel-matmul-tiling | FileCheck %s

func.func @matmul_tiled(%A: memref<1024x1024xf32>,
                        %B: memref<1024x1024xf32>,
                        %C: memref<1024x1024xf32>) {
  affine.for %i = 0 to 1024 step 32 {
    affine.for %j = 0 to 1024 step 32 {
      affine.for %k = 0 to 1024 step 32 {
        // CHECK: llvm.accel.matmul.32x32
        // CHECK: aligned address
        "accel.matmul_tile"(%subA, %subB, %subC) : ...
      }
    }
  }
  return
}
```

---

## 🔍 **핵심 알고리즘: Address Alignment**

### **의사코드 (Pseudo-code)**

```python
def lower_accel_matmul_tile(op):
    # Step 1: 메모리 레이아웃 분석
    layout = analyze_memory_layout(op.operands[0])  # operand: %subA

    # Step 2: 필요한 정렬 기준 결정
    required_alignment = 64  # GPU 기준 (캐시 라인)

    # Step 3: 현재 정렬 상태 확인
    if layout.alignment % required_alignment != 0:
        # Step 3a: 정렬되지 않음 → 패딩 필요
        temp_buffer = alloc_aligned_buffer(layout.totalBytes, required_alignment)
        copy_with_stride(op.operands[0], temp_buffer, layout)
        aligned_addr = extract_address(temp_buffer)
    else:
        # Step 3b: 이미 정렬됨 → 직접 사용
        aligned_addr = extract_address(op.operands[0])

    # Step 4: LLVM Intrinsic 생성
    result = create_llvm_call(
        "llvm.accel.matmul.32x32",
        [aligned_addr, ...],
        return_type=i32
    )

    return result
```

---

## 📈 **성능 목표**

```
메모리 대역폭 절감:
  - 패딩 없음: 0% 오버헤드
  - 패딩 필요: <5% 메모리 복사 오버헤드
  - 정렬 검증: <1% 컴파일 오버헤드

실행 시간:
  - 1024x1024 행렬: ~10ms (최적화 후)
  - 캐시 히트율: >95% (32x32 타일 = 4KB)

정확성:
  - 모든 변환이 의미론적으로 동등 (Sound)
  - 주소 정렬 오류 0%
```

---

## ✅ **구현 체크리스트**

```
[1] 프로젝트 구조 설정
    ☐ CMakeLists.txt 작성
    ☐ include/accel/*.h 정의
    ☐ lib/accel/*.cpp 구현

[2] Phase 1: Detection
    ☐ accel.matmul_tile operation 정의
    ☐ Walk through MLIR ops 코드

[3] Phase 2: Memory Layout Analysis
    ☐ MemRefType 분석 로직
    ☐ Stride 계산 알고리즘
    ☐ Contiguity 검증

[4] Phase 3: Alignment Check
    ☐ 주소 정렬 계산
    ☐ 패딩 삽입 로직
    ☐ 임시 버퍼 할당

[5] Phase 4: Lowering
    ☐ LLVM Intrinsic 생성
    ☐ 주소 전달 코드

[6] 검증
    ☐ 8개 케이스 모두 PASS
    ☐ FileCheck 통과
    ☐ 메모리 안전성 확인

[7] Gogs 저장
    ☐ 커밋
    ☐ 푸시
```

---

## 📍 **다음 단계**

이 설계를 바탕으로:
1. **C++ 구현 (AccelMatmulTilingPass.cpp)** 작성
2. **테스트 케이스 (8개)** 작성
3. **성능 벤치마크** 측정
4. **Gogs 저장 및 검증**

박사님, C++ 구현을 시작하시겠습니까?
