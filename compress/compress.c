#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <omp.h>
#include <time.h>
#include "htslib/bgzf.h"

const int BUF_SIZE = 65536;

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        fprintf(stderr, "Usage: %s <infile> <outfile>\n", argv[0]);
        return 1;
    }

    FILE *fin = fopen(argv[1], "rb");
    BGZF *out = bgzf_open(argv[2], "w");
    // int threads = omp_get_max_threads();
    // bgzf_set_compression_level(out, 0);
    out->compress_level = 0;
    if (bgzf_mt(out, 52, 256) != 0)
        return 1;
    uint8_t *buf = calloc(BUF_SIZE, sizeof(uint8_t));

    size_t nread;
    while ((nread = fread(buf, 1, BUF_SIZE, fin)) > 0)
    {
        int ret = bgzf_write(out, (const uint8_t *)buf, (int)nread);
        if (ret < 0)
            return 1;
    }

    if (bgzf_flush(out) < 0)
        return 1;
    if (bgzf_close(out) < 0)
        return 1;
    fclose(fin);
    return 0;
}
