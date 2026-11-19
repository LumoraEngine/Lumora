// depth.hlsl for objects that blend two levels of detail: the plain prepass, plus the discard of
// the pixels the other level draws (lodDitheredOut). No material maps are read.
#define LUMORA_DITHERED 1
#include "depth.hlsl"
