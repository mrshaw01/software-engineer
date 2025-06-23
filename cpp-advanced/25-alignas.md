## `alignas` in C++

Understanding memory alignment is fundamental for writing performant and portable code. The `alignas` specifier—introduced in C++11—provides a standardized, portable way to control the memory alignment of types and objects, superseding compiler-specific extensions like `__declspec(align(...))` or `__attribute__((aligned(...)))`.

## Purpose of `alignas`

The primary purpose of `alignas` is to _specify alignment requirements_ for user-defined types and variables. This is crucial in contexts such as:

- SIMD programming (e.g., aligning data to 16 or 32 bytes)
- Cache optimization
- Hardware interfacing or memory-mapped I/O
- Interop with C APIs requiring specific alignment

## Syntax

```cpp
alignas(expression)
alignas(type-id)
alignas(pack...)
```

- `expression`: must be an integral constant and a power of 2.
- `type-id`: aligns the object to the natural alignment of the specified type.
- `pack...`: a parameter pack, with alignment chosen as the largest alignment among types in the pack.

## Usage Examples

### Aligning a Struct to 8 Bytes

```cpp
struct alignas(8) S1 {
    int x;
};

static_assert(alignof(S1) == 8, "S1 should be 8-byte aligned");
```

### Multiple `alignas` — Largest One Wins

```cpp
class alignas(4) alignas(16) C1 {};

static_assert(alignof(C1) == 16, "C1 should be aligned to 16 bytes");
```

### `alignas(0)` is Ignored

```cpp
union alignas(0) U1 {
    int i;
    float f;
};

union U2 {
    int i;
    float f;
};

static_assert(alignof(U1) == alignof(U2), "U1 and U2 have the same alignment");
```

### Using a Type as the Alignment Value

```cpp
struct alignas(double) S2 {
    int x;
};

static_assert(alignof(S2) == alignof(double), "S2 should be aligned as double");
```

### Using a Template Pack

```cpp
template <typename... Ts>
class alignas(Ts...) C2 {
    char c;
};

static_assert(alignof(C2<>) == 1, "No types = default alignment");
static_assert(alignof(C2<short, int>) == 4, "Largest alignment is 4");
static_assert(alignof(C2<int, float, double>) == 8, "Largest alignment is 8");
```

## Important Rules

- **Alignment must match between declaration and definition**:

  ```cpp
  class alignas(16) C3;
  class alignas(32) C3 {}; // Error: alignment mismatch
  ```

- **Cannot reduce natural alignment**:

  ```cpp
  alignas(2) int x; // Ill-formed if int requires alignment > 2
  ```

- **Multiple `alignas`** are permitted, but only the _largest_ value takes effect.

## Best Practices

1. **Portability First**: Always prefer `alignas` over compiler-specific alignment keywords.
2. **Minimize Use**: Only use explicit alignment where performance or correctness demands it.
3. **Verify with `static_assert`**: Use `alignof` and `static_assert` to ensure alignment at compile time.
4. **Avoid Undefined Behavior**: Don’t attempt to use alignments smaller than the natural alignment of the type.
5. **Use with `std::aligned_storage` or `std::align`** when managing raw memory.

By controlling memory layout deterministically, `alignas` enables better cache alignment, hardware compatibility, and performance tuning—all critical for low-level systems programming.
