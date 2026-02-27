// ===== Accel Dialect ODS Definition =====
// accel.matmul_tile operation definition

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/CommonAttrConstraints.td"

def Accel_Dialect : Dialect {
  let name = "accel";
  let description = [{
    The Accel dialect provides operations for accelerator-specific
    optimizations, particularly for tiled matrix operations.
  }];
  let cppNamespace = "::mlir::accel";
}

class Accel_Op<string mnemonic, list<Trait> traits = []> :
  Op<Accel_Dialect, mnemonic, traits>;

// ===== accel.matmul_tile operation =====
def Accel_MatmulTileOp : Accel_Op<"matmul_tile", []> {
  let summary = "Tiled matrix multiplication operation";
  let description = [{
    Represents a tiled matrix multiplication for a single tile.
    This operation is produced by tiling a larger matmul operation
    and will be lowered to hardware-specific intrinsics.

    Example:
      %C = accel.matmul_tile %A, %B : memref<32x32xf32>
  }];

  let arguments = (ins
    AnyMemRef:$lhs,           // Left operand (A)
    AnyMemRef:$rhs,           // Right operand (B)
    AnyMemRef:$result         // Output (C)
  );

  let results = (outs);

  let assemblyFormat = [{
    $lhs `,` $rhs `:` type($lhs)
    attr-dict
  }];

  let hasVerifier = 1;
}

// ===== accel.aligned_alloc operation =====
def Accel_AlignedAllocOp : Accel_Op<"aligned_alloc", []> {
  let summary = "Allocate aligned memory buffer";
  let description = [{
    Allocates a memory buffer with specified alignment.
    Used for padding when data is not naturally aligned.

    Example:
      %buffer = accel.aligned_alloc %size, %alignment :
                memref<?xf32>
  }];

  let arguments = (ins
    Index:$size,
    Index:$alignment
  );

  let results = (outs AnyMemRef:$result);

  let assemblyFormat = [{
    $size `,` $alignment attr-dict `:` type($result)
  }];
}

// ===== accel.copy_tile operation =====
def Accel_CopyTileOp : Accel_Op<"copy_tile", []> {
  let summary = "Copy tile data with stride awareness";
  let description = [{
    Copies tile data from source to destination,
    handling strides and alignment correctly.
  }];

  let arguments = (ins
    AnyMemRef:$src,
    AnyMemRef:$dst
  );

  let results = (outs);

  let assemblyFormat = [{
    $src `to` $dst attr-dict `:` type($src) `->` type($dst)
  }];
}
