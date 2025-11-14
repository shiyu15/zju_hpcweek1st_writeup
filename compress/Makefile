TARGET = compress
SRC = compress.c

CC = gcc
CFLAGS = -Wall -Wextra -O3 -fopenmp
LDFLAGS = -lhts -lpthread

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
