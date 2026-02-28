# 📋 Research Grant Proposal

## **Title: Formal Semantics and Verification of High-Level Tensor Dialects in MLIR**

**Principal Investigator**: Ph.D. (Post-Doctoral Researcher)
**Research Period**: 24 months (2026-03 ~ 2028-02)
**Estimated Budget**: $380,000 USD
**Target Venues**: PLDI 2027, OOPSLA 2026

---

## **1. Executive Summary**

### **The Problem**

MLIR (Multi-Level Intermediate Representation) has revolutionized compiler design by enabling hierarchical, extensible intermediate representations. The Tensor Dialect provides high-level tensor operations (matmul, conv2d, etc.), and the Transform Dialect allows developers to express complex optimization transformations.

**However, a critical gap exists:**

- **No formal semantics**: Tensor operations lack mathematical definitions
- **No verification**: Transform operations have no correctness guarantees
- **Transform correctness unverified**: Developers cannot prove their optimizations are semantically equivalent
- **Manual verification**: All soundness checking is done by hand (error-prone)

**Consequence**: While ARIA's MLIR compiler achieves 8.2× speedup, there is **no mathematical proof** that these transformations preserve program semantics.

### **Our Solution: Sound MLIR Verification**

We propose to:

1. **Define formal semantics** for Tensor Dialect using denotational semantics
2. **Encode transformations** as first-order logic constraints
3. **Use SMT solvers** (Z3) to verify Sound equivalence
4. **Extend MLIR Transform Dialect** with verification annotations
5. **Provide tool support** for automatic verification

### **Expected Impact**

```
Before (ARIA):
  ✅ Performance: 8.2× speedup (experimental)
  ❌ Correctness: "Our transformation looks right" (manual review)
  ❌ Confidence: Medium (bugs in hand-written proofs)

After (This Work):
  ✅ Performance: 8.2× speedup (same or better)
  ✅ Correctness: Mathematically proven by SMT solver
  ✅ Confidence: Maximum (0 semantic bugs)
  ✅ Reproducibility: Every transformation can be independently verified
```

**Novelty**:
- First formal semantics of MLIR Tensor Dialect ⭐
- First SMT-based verification of MLIR transformations ⭐
- First Sound verification at scale (on real models) ⭐

---

## **2. Research Problem & Motivation**

### **2.1 The Compiler Verification Crisis**

In traditional compilers (GCC, LLVM), bugs can cause:
- **Silent correctness**: Program runs but produces wrong results
- **Compliance failures**: Output violates IEEE-754 floating-point semantics
- **Security vulnerabilities**: Optimization removes safety checks

For AI compilers (TensorFlow XLA, PyTorch, TVM):
- **Correctness is critical**: Trained models are valuable IP
- **Semantic preservation matters**: 1% accuracy loss = $M in value lost
- **Hand verification fails**: 1000s of lines of transformation code

**Example**: A single bug in ARIA's tiling transformation could:
- Change computation order (violates commutativity)
- Produce NaN from overflow (changes output semantics)
- Corrupt memory (changes subsequent operations)

### **2.2 Why Formal Verification?**

Current approaches:
- ❌ **Unit testing**: Catches known bugs, misses edge cases
- ❌ **Code review**: Doesn't scale (1000s of transformations)
- ❌ **Fuzzing**: Finds crashes, not semantic bugs
- ✅ **Formal verification**: Proves correctness for ALL inputs

### **2.3 Why MLIR?**

MLIR is the ideal target because:

1. **Hierarchical structure**: Operations have well-defined semantics
2. **Transform Dialect**: Transformations are first-class constructs
3. **Growing adoption**: Google (TensorFlow), NVIDIA, AMD, Apple all use MLIR
4. **Extensibility**: New dialects (Tensor, Vector, Affine) can be formally defined
5. **Research opportunity**: No one has done this yet

### **2.4 Key Challenges**

| Challenge | Solution |
|-----------|----------|
| Tensor Dialect has no formal semantics | Define using denotational semantics + type theory |
| SMT solvers don't understand floating-point | Use IEEE-754 bit-precise semantics (SMT-FPA) |
| Verification at scale is expensive | Use abstraction + caching (incremental SMT) |
| Transformations are complex (100s of lines) | Decompose into atomic verifiable steps |
| False positives waste developer time | Provide counterexamples for debugging |

---

## **3. Proposed Approach**

### **3.1 Phase 1: Formal Semantics of Tensor Dialect (6 months)**

#### **Objective**: Define mathematical meaning of tensor operations

**Step 1.1: Denotational Semantics (Weeks 1-8)**

Define each Tensor operation formally:

```
⟦linalg.matmul(A, B, C)⟧ =
  ∀i,j,k: C[i,j] := A[i,k] × B[k,j] + C[i,j]

⟦linalg.conv2d(input, kernel, output)⟧ =
  ∀n,h,w,c: output[n,h,w,c] :=
    ∑_{kh,kw,kc} input[n,h+kh,w+kw,kc] × kernel[kh,kw,kc,c]

⟦affine.for %i = L to U step S {...}⟧ =
  Sequence { body[i] | i ∈ {L, L+S, L+2S, ..., <U} }
```

**Output**:
- Formal semantics document (LaTeX, 30 pages)
- Type theory formalization (Coq/Isabelle, optional)

**Step 1.2: Type System & Affine Analysis (Weeks 9-12)**

Formalize type system:
- Tensor<shape, dtype>
- MemRef<shape, layout, storage_class>
- Index & affine expressions

**Step 1.3: Floating-Point Semantics (Weeks 13-16)**

Define IEEE-754 semantics:
```
⟦arith.addf(a, b)⟧ = round(a + b) with rounding mode
⟦overflow handling⟧ = ±∞ or NaN per IEEE-754
```

**Step 1.4: Memory Model (Weeks 17-24)**

Formalize:
- Address aliasing
- Memory ordering
- Cache coherence (at MLIR level)

---

### **3.2 Phase 2: SMT Encoding & Solver Integration (6 months)**

#### **Objective**: Convert formal semantics to SMT formulas

**Step 2.1: SMT Formula Generation (Weeks 1-8)**

For each tensor operation, generate Z3 constraints:

```python
# Example: MatMul verification

def verify_tiled_matmul():
    # Original: C = A × B
    original = [
        C_original[i,j] == Sum([
            A[i,k] * B[k,j] for k in range(K)
        ]) for i,j in indices
    ]

    # Tiled: C_tile = A_tile × B_tile
    tiled = [
        C_tiled[ti,tj,i',j'] == Sum([
            A[ti,tk,i',k'] * B[tk,tj,k',j']
            for tk in tile_range_k
            for k' in tile_range_k_inner
        ]) for all indices
    ]

    # Verify equivalence: original ≡ tiled
    return z3.prove(
        z3.And(original) == z3.And(tiled)
    )
```

**Step 2.2: Incremental Verification (Weeks 9-12)**

Cache decomposition results:
- Prove atomic transformations once
- Compose proofs (transitivity)
- 10-100× speedup

**Step 2.3: Solver Integration in MLIR (Weeks 13-20)**

Add MLIR passes:

```cpp
class FormalVerificationPass :
    public PassWrapper<FormalVerificationPass, ...> {

  // For each Transform operation:
  // 1. Extract operation semantics
  // 2. Generate SMT formula
  // 3. Call Z3 solver
  // 4. Report result (PASS/FAIL + counterexample)
};
```

**Step 2.4: Counterexample & Debugging (Weeks 21-24)**

When verification fails, provide:
- Minimal counterexample (concrete values)
- Visualization of difference
- Suggested fixes

---

### **3.3 Phase 3: Transform Dialect Verification (4 months)**

#### **Objective**: Verify all MLIR transformations

**Step 3.1: Transform Operations Formalization (Weeks 1-4)**

For Transform Dialect operations:
- `transform.tile` → Verify loop tiling preserves semantics
- `transform.fuse` → Verify operation fusion is correct
- `transform.loop_invariant_code_motion` → Verify it's safe
- `transform.parallel` → Verify parallelization is valid

**Step 3.2: Composition & Soundness (Weeks 5-12)**

Prove that:
```
If T1 is sound (A ≡ T1(A))
And T2 is sound (B ≡ T2(B))
Then T2∘T1 is sound (A ≡ T2(T1(A)))
```

**Step 3.3: Real-World Validation (Weeks 13-16)**

Test on actual models:
- ResNet-50 (52 layers)
- BERT (12 layers × 2 passes)
- GPT-2 (48 layers)

Verify all transformations on real code.

---

### **3.4 Phase 4: Tool Support & Deployment (4 months)**

#### **Objective**: Make verification practical for developers

**Step 4.1: MLIR Integration (Weeks 1-6)**

Add pass to MLIR:
```bash
mlir-opt input.mlir -verify-transforms -verify-report out.json
```

Output: JSON with per-transformation verification result

**Step 4.2: IDE Integration (Weeks 7-10)**

VS Code extension:
- Real-time verification as user writes
- Green checkmark (Sound) / Red X (Unsound)
- Inline hints: "This transformation adds 0.2% overhead"

**Step 4.3: Documentation & Examples (Weeks 11-16)**

Write:
- Formal semantics tutorial (10 pages)
- Case study: How to verify custom transformation (5 pages)
- API reference for SMT encoding (10 pages)
- Troubleshooting guide (5 pages)

---

## **4. Technical Innovation**

### **4.1 Novel Contributions**

| Contribution | Significance |
|--------------|--------------|
| **First formal semantics of MLIR Tensor Dialect** | Enables mathematical reasoning about MLIR |
| **SMT-based verification framework** | Automates soundness proofs |
| **Incremental verification with proof caching** | Makes verification practical at scale |
| **IEEE-754 bit-precise semantics** | Handles floating-point subtleties |
| **Transform composition theory** | Proves multi-pass optimizers are sound |

### **4.2 Technical Depth**

This work combines:
- **Type theory** (algebraic semantics of operations)
- **Formal logic** (first-order + SMT)
- **Compiler theory** (polyhedral model + affine analysis)
- **Floating-point theory** (IEEE-754 semantics)
- **Implementation** (C++, Z3 API, MLIR passes)

**Research difficulty**: ⭐⭐⭐⭐⭐ (Highest)
- Not just engineering: Requires new theory
- Not just theory: Must implement at scale

---

## **5. Expected Outcomes**

### **5.1 Deliverables**

**Academic (Publications)**:
- ✅ Main paper: "Formal Semantics and Sound Verification of MLIR Transformations" (PLDI 2027)
- ✅ Tool paper: "Z3-MLIR: Automatic Verification Framework" (OOPSLA 2026)
- ✅ Theory paper: "Composition and Soundness of Hierarchical Transformations" (POPL 2027, optional)

**Code (Open-source)**:
- 📦 mlir-z3-verifier: Full verification framework (~5,000 lines C++)
- 📦 tensor-semantics: Formal definitions (~2,000 lines LaTeX/Coq)
- 📦 transform-library: Verified transformations (~3,000 lines MLIR)
- 📦 ide-plugin: VS Code extension (~500 lines TypeScript)

**Documentation**:
- 📖 Formal Semantics Manual (50 pages, PDF)
- 📖 API Reference (30 pages)
- 📖 Case Studies (15 pages)

### **5.2 Impact**

**Immediate (2026-2027)**:
- Provide developers with confidence in MLIR optimizations
- Establish MLIR as safer alternative to XLA/TVM
- Set standard for compiler verification research

**Long-term (2027-2028)**:
- Influence MLIR community adoption
- Enable formal verification as standard practice
- Prevent bugs in production AI systems

---

## **6. Timeline & Milestones**

| Phase | Duration | Key Milestones |
|-------|----------|-----------------|
| **1: Formal Semantics** | 6 months | M1: Tensor semantics done; M2: Affine analysis; M3: IEEE-754 model |
| **2: SMT Encoding** | 6 months | M4: SMT generation working; M5: Incremental verification; M6: MLIR integration |
| **3: Transform Verification** | 4 months | M7: Transform ops verified; M8: Real-world validation on 3 models |
| **4: Deployment** | 4 months | M9: IDE plugin; M10: Documentation complete; M11-M12: Paper writing |

**Milestones with papers**:
- **Month 4 (2026-07)**: OOPSLA 2026 tool paper
- **Month 8 (2026-11)**: Internal tech report
- **Month 12 (2027-03)**: PLDI 2027 main paper
- **Month 20 (2028-02)**: Completion + final documentation

---

## **7. Budget Estimate**

```
Personnel:
  PI (36 months × 60% effort)          $180,000
  Senior PhD Student (24 months)        $120,000
  Graduate Research Assistant (24 mo)    $60,000

Equipment & Software:
  Computing resources (GPU servers)      $10,000
  Z3/SMT solver licenses (if needed)     $5,000

Travel:
  PLDI 2027 + OOPSLA 2026                 $5,000

Total: $380,000 (estimated)
```

---

## **8. Related Work**

### **Compiler Verification**
- CompCert (Leroy et al.) - Verified C compiler in Coq
- **This work**: First verified MLIR compiler

### **Semantics of Optimizations**
- Polly + isl (Grosser et al.) - Polyhedral model
- **This work**: Formal semantics + SMT verification

### **SMT-based Verification**
- Alive (Lopes et al.) - Verify LLVM optimizations
- SeaHorn (Gurfinkel et al.) - Software verification
- **This work**: Applied to MLIR at compiler level

### **MLIR Research**
- MLIR itself (Lattner et al.) - IR design
- Affine dialect (previous work) - Loop optimization
- **This work**: Formal semantics + verification

---

## **9. Risk Assessment & Mitigation**

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| SMT solver timeout on large programs | Medium | High | Use abstraction + caching + slicing |
| IEEE-754 complexity | Low | Medium | Simplify to FP32 initially, extend later |
| MLIR API changes | Low | Medium | Follow MLIR releases closely |
| Limited adoption of Transform Dialect | Medium | Low | Target ARIA use cases first |

---

## **10. Expected Citations & Venue Acceptance**

### **PLDI 2027 (Main Paper)**
- Acceptance rate: ~20% (very competitive)
- Our strengths: Novel theory + practical implementation
- Expected impact: 50-100 citations in 5 years

### **OOPSLA 2026 (Tool Paper)**
- Acceptance rate: ~25% (tool track is easier)
- Expected impact: Community adoption, GitHub stars

### **Follow-up Work**
- POPL 2027: Composition theory (optional)
- ASPLOS 2027: Performance on distributed systems (future work)

---

## **References**

1. **MLIR**: Lattner et al., "MLIR: A Compiler Infrastructure for the End of Moore's Law"
2. **CompCert**: Leroy et al., "Formal Verification of a Realistic Compiler"
3. **Alive**: Lopes et al., "Alive: New Verification Tool for LLVM Optimizations"
4. **ARIA**: Our previous work on Hardware-Aware Tiling
5. **Z3**: De Moura & Bjørner, "Z3: An Efficient SMT Solver"

---

**Proposal Status**: ✅ Complete draft
**Next Step**: Convert to formal proposal (10 pages) for NSF/DOE/DARPA submission
