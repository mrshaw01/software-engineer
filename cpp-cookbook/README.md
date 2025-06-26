# cpp-cookbook

This project is a collection of standalone C++ example programs that demonstrate the usage of modern C++ features and standard library utilities. Each `.cpp` file is compiled independently into its own executable.

## Directory Structure

```
cpp-cookbook/
├── build.sh           # Shell script to configure and build the project
├── CMakeLists.txt     # CMake configuration to build all .cpp files
├── README.md          # Project documentation
├── file1.cpp          # Example source file
├── file2.cpp          # Another example
└── ...                # Additional C++ examples
```

## Requirements

- CMake version 3.10 or higher
- A C++17-compliant compiler (such as g++ or clang++)

## Build Instructions

To build the project:

1. Navigate to the `cpp-cookbook` directory.
2. Run the provided build script:

```bash
./build.sh
```

This will:

- Create a `build/` directory (if it does not exist)
- Run `cmake` to configure the build system
- Compile all `.cpp` files into individual executables

## Running the Programs

After a successful build, the executables will be located in the `build/` directory. Each can be executed independently:

```bash
./build/file1
./build/file2
```

## Adding New Examples

To add a new program:

1. Create a new `.cpp` file in the `cpp-cookbook` directory.
2. No changes to `CMakeLists.txt` are required. It automatically includes all `.cpp` files.
