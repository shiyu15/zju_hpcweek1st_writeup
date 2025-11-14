
#pragma once
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stddef.h>

typedef struct { double  v[3]; } vec3f64;
typedef struct { float   v[2]; } vec2f32;
typedef struct { int16_t v[3]; } vec3i16;
typedef struct { int8_t  v[2]; } vec2i8;

typedef struct {
  int16_t blockcount;
  int32_t blocks_state[4096];
  int32_t biomes[64];

  int8_t  sky_light[2048];
  int8_t  block_light[2048];
} Section;

typedef struct {
    int32_t last_update;
    Section sections[24];
} Chunk;

enum {
  Section_size = sizeof(Section),
  Section_align = _Alignof(Section),
  Section_off_blockcount  = offsetof(Section, blockcount),
  Section_off_blocks_state= offsetof(Section, blocks_state),
  Section_off_biomes      = offsetof(Section, biomes),
  Section_off_sky_light   = offsetof(Section, sky_light),
  Section_off_block_light = offsetof(Section, block_light),

  Chunk_size = sizeof(Chunk),
  Chunk_align = _Alignof(Chunk),
  Chunk_off_sections = offsetof(Chunk, sections),
};

Chunk* load_chunk(int32_t x, int32_t z);
void tick_chunk();

void setblock(int32_t x, int32_t y, int32_t z, int32_t state_id);

void clear_ticks();

typedef struct {
    int32_t x;
    int32_t y;
    int32_t z;
    int32_t state_id;
}SetblockRequest;

void batch_setblock(int32_t len, SetblockRequest* reqs);

void tickAftersetblock();