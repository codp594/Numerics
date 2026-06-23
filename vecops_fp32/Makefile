TARGET := vecops_fp32
SRC := main.c
HDR := ssve.h

ifeq ($(origin CC),default)
$(error Please set CC explicitly, for example: make CC=clang)
endif

CFLAGS := -O3 -Wall -march=armv9.2-a+sme2+sve+sve2
.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC) $(HDR)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)
