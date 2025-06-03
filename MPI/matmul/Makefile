TARGET=main
OBJECTS=util.o matmul.o

CPPFLAGS=-std=c++11 -O3 -Wall -march=znver2 -mavx2 -mfma -fopenmp
LDLIBS=-lm -lmpi -lmpi_cxx -lnuma

all: $(TARGET)

$(TARGET): $(OBJECTS)

clean:
	rm -rf $(TARGET) $(OBJECTS)
