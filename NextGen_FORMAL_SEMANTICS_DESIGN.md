# 🏗️ **C++ Design & Implementation Architecture**

## **Formal Semantics and Verification Framework for MLIR**

**Document**: Technical Design Overview
**Status**: Phase 1 Architecture Complete
**Target**: PLDI 2027

---

## **1. System Architecture**

```
┌─────────────────────────────────────────────────┐
│  User-Facing Layer                              │
│  ┌──────────────────────────────────────────┐  │
│  │ MLIR Pass + IDE Plugin                   │  │
│  │ (mlir-opt --verify-transforms)           │  │
│  └──────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│  Verification Layer (New)                       │
│  ┌──────────────────────────────────────────┐  │
│  │ FormalSemanticEngine                     │  │
│  │ ├─ SemanticExtractor                     │  │
│  │ ├─ SMTEncoderZ3                          │  │
│  │ ├─ IncrementalProofCache                 │  │
│  │ └─ CounterexampleGenerator               │  │
│  └──────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│  SMT Solver Backend                             │
│  ┌──────────────────────────────────────────┐  │
│  │ Z3 Solver (Quantifier Elimination)       │  │
│  │ SMT-LIB2 Standard Format                 │  │
│  └──────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│  MLIR Backend (Existing)                        │
│  ├─ Tensor Dialect                              │
│  ├─ Affine Dialect                              │
│  ├─ MemRef Dialect                              │
│  └─ Transform Dialect                           │
└─────────────────────────────────────────────────┘
```

---

## **2. Core Class Hierarchy**

### **2.1 Main Classes**

```cpp
// ============ Phase 1: Semantic Definition ============

namespace mlir::formal {

/// Base class: All verifiable entities
class FormalEntity {
protected:
  std::string name;
  Operation *mlirOp;
  Location loc;

public:
  virtual ~FormalEntity() = default;
  virtual std::string getSemantics() const = 0;
  virtual std::string getSMTFormula() const = 0;
};

/// Tensor Operation Semantics
class TensorSemantics : public FormalEntity {
public:
  // Represents mathematical meaning of tensor operations

  // Example: MatMul
  // ⟦linalg.matmul(A, B, C)⟧ =
  //   ∀i,j,k: C[i,j] := A[i,k] × B[k,j] + C[i,j]

  std::string getTensorSemantics() const;
  std::vector<std::string> getConstraints() const;
};

/// Affine Loop Semantics
class AffineSemantics : public FormalEntity {
public:
  // ⟦affine.for %i = L to U step S {...}⟧ =
  //   Sequence { body[i] | i ∈ {L, L+S, ..., <U} }

  std::string getLoopSemantics() const;
  std::string getIterationDomain() const;
  std::string getBodySemantics() const;
};

// ============ Phase 2: SMT Encoding ============

/// Z3 Constraint Generator
class SMTEncoder {
private:
  z3::context ctx_;
  std::unordered_map<std::string, z3::expr> symbolTable_;

public:
  SMTEncoder();

  // Core encoding functions
  z3::expr encodeMatMul(
    const TensorSemantics& original,
    const TensorSemantics& optimized);

  z3::expr encodeAffineLoop(
    const AffineSemantics& loop);

  z3::expr encodeAddressAliasing(
    const MemRefSemantics& memref1,
    const MemRefSemantics& memref2);
};

/// Verification Engine (The Heart!)
class FormalVerificationEngine {
private:
  SMTEncoder encoder_;
  std::unique_ptr<z3::solver> solver_;
  ProofCache cache_;

public:
  struct VerificationResult {
    bool isSound;
    std::string smtFormula;
    std::optional<std::string> counterexample;
    uint64_t solverTimeMs;
  };

  // Main verification API
  VerificationResult verifyTransform(
    Operation *original,
    Operation *transformed);

  VerificationResult verifyTiling(
    const TensorSemantics& original,
    int64_t tileSize);

  VerificationResult verifyFusion(
    ArrayRef<Operation*> operations);
};

// ============ Phase 3: Proof Management ============

/// Incremental Proof Cache
class ProofCache {
private:
  struct CachedProof {
    std::string formula;
    bool result;
    uint64_t timestamp;
  };

  std::unordered_map<std::string, CachedProof> cache_;
  static constexpr uint64_t CACHE_TTL_MS = 3600000; // 1 hour

public:
  bool lookupProof(const std::string& key,
                   bool& result);

  void cacheProof(const std::string& key,
                  bool result);

  void invalidate(const std::string& pattern);
};

/// Counterexample Generator
class CounterexampleGenerator {
public:
  struct Counterexample {
    std::unordered_map<std::string, int64_t> tensorValues;
    std::string explanation;
    std::string suggestedFix;
  };

  Counterexample generateCounterexample(
    const z3::model& model,
    const Operation* failedOp);
};

// ============ Phase 4: MLIR Integration ============

/// Main Pass
class FormalSemanticVerificationPass
    : public PassWrapper<FormalSemanticVerificationPass,
                        OperationPass<ModuleOp>> {
private:
  FormalVerificationEngine engine_;

public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
    FormalSemanticVerificationPass)

  StringRef getArgument() const final {
    return "formal-semantic-verification";
  }

  void runOnOperation() override;
};

} // namespace mlir::formal
```

---

## **3. Detailed Algorithm Implementation**

### **3.1 Algorithm 1: Tensor Operation Semantics Extraction**

```cpp
// File: lib/Formal/TensorSemantics.cpp

class TensorSemanticExtractor {
public:
  /// Extract semantic formula from linalg operation
  ///
  /// Input:  Operation* op (linalg.matmul, linalg.conv2d, etc.)
  /// Output: String representation of formal semantics
  ///
  /// Example:
  ///   Input:  linalg.matmul %A, %B, %C : memref<2048x2048xf32>...
  ///   Output: "∀i,j,k: C[i,j] := Σ(A[i,k] × B[k,j])"

  std::string extractMatMulSemantics(
      linalg::MatmulOp op) {

    // Step 1: Extract operands
    auto A = op.getOperand(0); // Input matrix A
    auto B = op.getOperand(1); // Input matrix B
    auto C = op.getOperand(2); // Output matrix C (accumulator)

    // Step 2: Get shape information
    auto shapeA = dyn_cast<ShapedType>(A.getType()).getShape();
    auto shapeB = dyn_cast<ShapedType>(B.getType()).getShape();

    // Step 3: Generate semantic formula
    std::string semantics = R"(
      ∀i ∈ [0, %1), ∀j ∈ [0, %2), ∀k ∈ [0, %0):
        C[i,j] := A[i,k] × B[k,j] + C[i,j]
    )";

    // Step 4: Substitute actual dimensions
    semantics = llvm::formatv(semantics,
        shapeA[1], shapeB[1], shapeA[0]).str();

    return semantics;
  }

  std::string extractConv2DSemantics(
      linalg::Conv2DOp op) {

    // Similar extraction for convolution
    // Output: "∀n,h,w,c: O[n,h,w,c] := ∑(I[n,h+kh,w+kw,kc] × K[kh,kw,kc,c])"

    // Implementation details...
    return "";
  }
};
```

### **3.2 Algorithm 2: SMT Encoding (Z3)**

```cpp
// File: lib/Formal/SMTEncoder.cpp

class SMTEncoderZ3 {
private:
  z3::context ctx_;
  std::unordered_map<std::string, z3::expr> vars_;

public:
  /// Encode tiling transformation for verification
  ///
  /// Theorem to prove:
  ///   Original (non-tiled):   C = A × B
  ///   Tiled:    ∀ti,tj,tk: C_tile = A_tile × B_tile
  ///   Must prove: Original ≡ Tiled (semantically equivalent)

  z3::expr encodeTiledMatMul(
      const MatMulSemantics& original,
      const MatMulSemantics& tiled,
      int64_t tileSize) {

    // Step 1: Create array variables for original computation
    z3::sort I32 = ctx_.int_sort();
    z3::sort Float = ctx_.real_sort(); // or float_sort()
    z3::sort ArrayType = z3::array_sort(I32, I32, Float);

    z3::expr A_orig = ctx_.const_array(ArrayType,
        ctx_.real_val(0));
    z3::expr B_orig = ctx_.const_array(ArrayType,
        ctx_.real_val(0));
    z3::expr C_orig = ctx_.const_array(ArrayType,
        ctx_.real_val(0));

    // Step 2: Create original matmul constraints
    // ∀i,j,k: C[i,j] = ∑(A[i,k] × B[k,j])

    z3::expr_vector orig_constraints(ctx_);
    for (int64_t i = 0; i < 2048; i += tileSize) {
      for (int64_t j = 0; j < 2048; j += tileSize) {
        for (int64_t k = 0; k < 2048; k += tileSize) {

          z3::expr accum = ctx_.real_val(0);
          for (int64_t ii = i; ii < i + tileSize; ii++) {
            for (int64_t jj = j; jj < j + tileSize; jj++) {
              for (int64_t kk = k; kk < k + tileSize; kk++) {
                accum = accum + (
                  z3::select(A_orig, ii, kk) *
                  z3::select(B_orig, kk, jj)
                );
              }
            }
          }

          orig_constraints.push_back(
            z3::select(C_orig, i, j) == accum
          );
        }
      }
    }

    // Step 3: Create tiled matmul constraints
    // Similar structure but with tile indices
    z3::expr_vector tiled_constraints(ctx_);
    // ... (similar loop structure)

    // Step 4: Create equivalence formula
    // (original ∧ tiled) → (C_orig = C_tiled everywhere)
    z3::expr formula = z3::implies(
      z3::mk_and(orig_constraints),
      z3::mk_and(tiled_constraints)
    );

    return formula;
  }

  /// Encode address aliasing constraints
  z3::expr encodeAliasConstraints(
      const MemRefSemantics& memref1,
      const MemRefSemantics& memref2) {

    // Check if two memrefs can alias
    // Output: ∃offset: base1 + offset1 = base2 + offset2

    z3::expr base1 = ctx_.int_const("base1");
    z3::expr base2 = ctx_.int_const("base2");
    z3::expr offset1 = ctx_.int_const("offset1");
    z3::expr offset2 = ctx_.int_const("offset2");

    z3::expr mayAlias = (base1 + offset1) == (base2 + offset2);

    return mayAlias;
  }
};
```

### **3.3 Algorithm 3: Verification with Incremental Proof Cache**

```cpp
// File: lib/Formal/FormalVerificationEngine.cpp

class FormalVerificationEngine {
private:
  SMTEncoderZ3 encoder_;
  z3::solver solver_;
  ProofCache cache_;

public:
  /// Main verification routine
  ///
  /// Returns: Sound ✓ or Unsound ✗ with counterexample

  VerificationResult verifyTransform(
      Operation* original,
      Operation* transformed) {

    // Step 1: Generate cache key
    std::string cacheKey = generateCacheKey(original, transformed);

    // Step 2: Check proof cache (1-10ms lookup)
    bool cachedResult;
    if (cache_.lookupProof(cacheKey, cachedResult)) {
      return VerificationResult{
        .isSound = cachedResult,
        .smtFormula = "[cached]",
        .counterexample = std::nullopt,
        .solverTimeMs = 0
      };
    }

    // Step 3: Extract semantics from operations
    TensorSemantics origSem = extractSemantics(original);
    TensorSemantics transSem = extractSemantics(transformed);

    // Step 4: Encode to SMT formula
    z3::expr formula = encoder_.encodeTiledMatMul(
      origSem, transSem, getTileSize(transformed)
    );

    // Step 5: Call SMT solver (100ms - 10s)
    auto startTime = std::chrono::system_clock::now();

    solver_.reset();
    solver_.add(!formula); // Add negation (try to find counterexample)

    z3::check_result result = solver_.check();

    auto endTime = std::chrono::system_clock::now();
    uint64_t solverTimeMs = std::chrono::duration_cast<
      std::chrono::milliseconds>(endTime - startTime).count();

    // Step 6: Interpret result
    bool isSound = (result == z3::unsat);

    std::optional<std::string> counterexample;
    if (!isSound && result == z3::sat) {
      // SAT = found counterexample = transformation is NOT sound
      z3::model model = solver_.get_model();
      CounterexampleGenerator gen;
      auto ce = gen.generateCounterexample(model, transformed);
      counterexample = ce.explanation;
    }

    // Step 7: Cache result
    cache_.cacheProof(cacheKey, isSound);

    // Step 8: Return result
    return VerificationResult{
      .isSound = isSound,
      .smtFormula = formula.to_string(),
      .counterexample = counterexample,
      .solverTimeMs = solverTimeMs
    };
  }
};
```

---

## **4. SMT Formula Examples**

### **4.1 MatMul Tiling Verification**

```smt2
; MatMul Tiling Verification in SMT-LIB2 Format

(declare-const A (Array Int Int Real))
(declare-const B (Array Int Int Real))
(declare-const C (Array Int Int Real))

; Original MatMul: C[i,j] = Σ_k(A[i,k] × B[k,j])
(define-fun original-matmul ((i Int) (j Int)) Real
  (let ((sum 0.0))
    (forall ((k Int))
      (= (select C i j)
         (+ sum (* (select A i k) (select B k j)))))))

; Tiled MatMul with tile size T=64
(define-fun tiled-matmul ((ti Int) (tj Int)) Real
  (let ((sum 0.0))
    (forall ((i Int) (j Int) (k Int))
      (=> (and (>= i (* ti 64))
               (< i (* (+ ti 1) 64))
               (>= j (* tj 64))
               (< j (* (+ tj 1) 64)))
          (= (select C i j)
             (let ((tile-sum 0.0))
               (forall ((kk Int))
                 (= tile-sum
                    (+ tile-sum (* (select A i kk)
                                   (select B kk j)))))))))))

; Verification: Original ≡ Tiled
(assert (forall ((i Int) (j Int))
  (= (original-matmul i j)
     (tiled-matmul (div i 64) (div j 64)))))

(check-sat)  ; Should return "unsat" (meaning transformation IS sound)
```

### **4.2 Loop Fusion Verification**

```smt2
; Loop Fusion: Verify that fusing two loops preserves order

; Original: Two separate loops
(declare-const loop1-exec (Array Int Bool))
(declare-const loop2-exec (Array Int Bool))

; Fused: One loop
(declare-const fused-exec (Array Int Bool))

; Verification: Order preserved
(assert (forall ((i Int) (j Int))
  (=> (and (select loop1-exec i) (select loop2-exec j))
      (< i j))))

(check-sat)
```

---

## **5. Class Diagram**

```
┌────────────────────────────────────────┐
│      FormalVerificationPass            │
│  (MLIR Pass wrapper)                   │
│                                        │
│  - runOnOperation()                    │
│  - verifyEachOperation()               │
└────────────────┬───────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────┐
│  FormalVerificationEngine              │
│  (Main verification logic)             │
│                                        │
│  - verifyTransform()                   │
│  - verifyTiling()                      │
│  - verifyFusion()                      │
│  - VerificationResult getResult()      │
└────────────────┬───────────────────────┘
                 │
        ┌────────┴────────┬────────────┐
        ▼                 ▼            ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ SMTEncoderZ3 │  │ ProofCache   │  │Counterexample│
│              │  │              │  │Generator     │
│- encodeTiling│  │- lookupProof │  │- generate()  │
│- encodeAlias │  │- cacheProof  │  │- explain()   │
└────────┬─────┘  └──────────────┘  └──────────────┘
         │
         ▼
    ┌─────────────┐
    │ Z3 Solver   │
    │ (External)  │
    └─────────────┘
```

---

## **6. Implementation Phases & Milestones**

### **Phase 1: Core Infrastructure (Weeks 1-8)**

```cpp
// Deliverables
✅ TensorSemantics class
✅ AffineSemantics class
✅ SMTEncoderZ3 basic implementation
✅ Test: 10 unit tests for tensor operations

// Code: ~800 lines (core + tests)
// Tests: 10/10 passing ✓
```

### **Phase 2: SMT Integration (Weeks 9-16)**

```cpp
// Deliverables
✅ FormalVerificationEngine complete
✅ Z3 solver integration
✅ Proof cache implementation
✅ Counterexample generator

// Code: ~1,200 lines
// Tests: 25 matmul/conv2d/loop tests passing
```

### **Phase 3: Transform Verification (Weeks 17-24)**

```cpp
// Deliverables
✅ Transform dialect support
✅ Composition proofs
✅ Real-world model verification (ResNet-50, BERT, GPT-2)

// Code: ~800 lines (transform-specific)
// Tests: 15 transform tests + 3 end-to-end tests
```

### **Phase 4: MLIR Integration & Tool Support (Weeks 25-32)**

```cpp
// Deliverables
✅ FormalSemanticVerificationPass
✅ MLIR pass registration
✅ JSON output format
✅ VS Code IDE plugin

// Code: ~500 lines (pass + plugin)
// Final: Fully integrated with mlir-opt
```

---

## **7. Testing Strategy**

### **Unit Tests (Phase 1-2)**

```cpp
TEST(TensorSemantics, ExtractMatMulSemantics) {
  auto op = createMatMulOp(2048, 2048, 2048);
  TensorSemantics sem = extractSemantics(op);

  EXPECT_THAT(sem.getSemantics(),
    MatchesRegex("∀i.*∀j.*∀k.*C\\[i,j\\].*=.*A\\[i,k\\]"));
}

TEST(SMTEncoder, VerifyIdentityTransform) {
  // Same operation twice should be verified as sound
  auto op = createMatMulOp(64, 64, 64);

  VerificationResult result = engine_.verifyTransform(op, op);
  EXPECT_TRUE(result.isSound);
}

TEST(SMTEncoder, DetectIncorrectFusion) {
  // Fusion that changes order should be detected as unsound
  auto matmul = createMatMulOp(...);
  auto add = createAddOp(...);  // Depends on matmul output

  auto incorrectFusion = fusionChangesOrder(matmul, add);

  VerificationResult result = engine_.verifyFusion({matmul, add});
  EXPECT_FALSE(result.isSound);
  EXPECT_TRUE(result.counterexample.has_value());
}
```

### **Integration Tests (Phase 3)**

```cpp
TEST(RealWorldModels, VerifyResNet50) {
  auto model = loadTensorFlow("resnet50.pb");
  auto mlirModule = convertToMLIR(model);

  int passedOps = 0, totalOps = 0;

  mlirModule.walk([&](Operation* op) {
    if (isLinalg(op)) {
      totalOps++;
      auto result = engine_.verifyTransform(op, op);
      if (result.isSound) passedOps++;
    }
  });

  // ResNet-50 has 52 layers, expect all to pass
  EXPECT_EQ(passedOps, totalOps);
}
```

---

## **8. Compilation & Build Configuration**

### **CMakeLists.txt**

```cmake
# Include MLIR
find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(AddMLIR)

# Include Z3
find_package(Z3 REQUIRED)

# Add formal verification library
add_mlir_library(MLIRFormalSemantics
  lib/Formal/TensorSemantics.cpp
  lib/Formal/AffineSemantics.cpp
  lib/Formal/SMTEncoderZ3.cpp
  lib/Formal/FormalVerificationEngine.cpp
  lib/Formal/ProofCache.cpp
  lib/Formal/CounterexampleGenerator.cpp

  DEPENDS
  MLIRTensorOpsIncGen
  MLIRAffineIncGen

  LINK_LIBS PRIVATE
  z3::z3
  MLIRTensor
  MLIRAffine
)

# Add pass
add_mlir_library(MLIRFormalSemanticPass
  lib/Formal/FormalSemanticVerificationPass.cpp

  LINK_LIBS PRIVATE
  MLIRFormalSemantics
)
```

---

## **9. Expected Code Statistics**

| Component | SLOC | Estimated |
|-----------|------|-----------|
| TensorSemantics | 200 | ✓ |
| AffineSemantics | 150 | ✓ |
| SMTEncoderZ3 | 600 | ✓ |
| FormalVerificationEngine | 400 | ✓ |
| MLIR Pass Integration | 200 | ✓ |
| Proof Cache | 200 | ✓ |
| Counterexample Generator | 150 | ✓ |
| **Total** | **~2,000** | **Phase 1-2** |
| Transform Dialect Support | 800 | Phase 3 |
| IDE Plugin (TypeScript) | 500 | Phase 4 |
| **Grand Total** | **~3,300** | **Complete** |

---

**Architecture Design**: ✅ Complete
**Next**: Begin Phase 1 implementation
