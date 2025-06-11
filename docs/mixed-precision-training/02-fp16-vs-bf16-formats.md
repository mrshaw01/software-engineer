# Part 2: FP16 and BF16 Formats

Mixed precision training leverages low-precision formats to improve memory efficiency and computation speed. Two widely used 16-bit formats are **FP16 (IEEE 754 half precision)** and **BF16 (bfloat16)**. While both use 16 bits, they allocate those bits differently, resulting in distinct trade-offs between dynamic range and precision.

## FP16: IEEE 754 Half-Precision Format

The **FP16** format (also known as IEEE 754 binary16) uses:

- **1 sign bit**
- **5 exponent bits**
- **10 fraction (mantissa) bits**

This yields a total of 11 bits of precision due to the implicit leading 1 in normalized numbers.

### Exponent Range

- **Exponent bias**: 15
- **Unbiased exponent range**: −14 to +15
- Exponent values 0 and 31 are reserved for special cases (e.g., zero, infinity, NaN)

### Value Types

| Type              | Range        | Notes                                    |
| ----------------- | ------------ | ---------------------------------------- |
| Normalized values | 2⁻¹⁴ to 2¹⁵  | Implicit leading 1; 11 bits of precision |
| Subnormal values  | 2⁻²⁴ to 2⁻¹⁵ | No implicit 1; precision decreases       |

### Example Magnitudes

- **Maximum normalized**: 65,504
- **Minimum normalized**: ≈ 6.10 × 10⁻⁵
- **Minimum subnormal**: ≈ 5.96 × 10⁻⁸

### Summary

- **Dynamic range**: ≈ 10⁻⁴ to 10⁴ (~40 powers of 2)
- **Precision**: ~3.3 decimal digits
- **Supports subnormals** and special values
- **Use case**: Suitable when compact size and higher precision per bit are needed

## BF16: Brain Floating Point Format

The **BF16** (bfloat16) format is designed specifically for deep learning workloads. It uses:

- **1 sign bit**
- **8 exponent bits**
- **7 fraction (mantissa) bits**

The exponent field is the same as FP32, allowing for identical dynamic range.

### Exponent Range

- **Exponent bias**: 127
- **Unbiased exponent range**: −126 to +127
- Values 0 and 255 are reserved for zero, infinity, and NaN

### Value Types

| Type              | Range                         | Notes                                       |
| ----------------- | ----------------------------- | ------------------------------------------- |
| Normalized values | ≈ 1.18 × 10⁻³⁸ to 3.39 × 10³⁸ | Same dynamic range as float32               |
| Subnormal values  | Rarely implemented            | Defined in spec (~2⁻¹³³), often unsupported |

### Summary

- **Dynamic range**: Same as FP32 (~10⁻³⁸ to 10³⁸)
- **Precision**: ~2.2 decimal digits (due to only 8 bits of precision)
- **Simplifies conversion** from FP32 (same exponent field)
- **Use case**: Preferred for training with wide-ranging values and stable gradients

## FP16 vs BF16: Comparison Table

| Feature                 | FP16                     | BF16                        |
| ----------------------- | ------------------------ | --------------------------- |
| Total bits              | 16                       | 16                          |
| Exponent bits           | 5                        | 8                           |
| Fraction bits           | 10 (11 with implicit)    | 7 (8 with implicit)         |
| Exponent bias           | 15                       | 127                         |
| Exponent range          | –14 to +15               | –126 to +127                |
| Dynamic range           | ≈ 10⁻⁴ to 10⁴            | ≈ 10⁻³⁸ to 10³⁸             |
| Decimal precision       | ~3.3 digits              | ~2.2 digits                 |
| Subnormals supported    | Yes                      | Rarely (hardware-dependent) |
| Compatibility with FP32 | Requires conversion      | Easy (same exponent)        |
| Typical hardware usage  | GPUs (e.g. Tensor Cores) | TPUs, CPUs, some GPUs       |

## When to Use FP16 vs BF16

**Use FP16 when:**

- You need better precision per bit
- Your hardware supports subnormals and loss scaling
- You want high throughput on Tensor Cores

**Use BF16 when:**

- You want to maintain the same dynamic range as FP32
- You work with models prone to underflow/overflow in FP16
- You prefer easy conversion from FP32 with minimal logic
