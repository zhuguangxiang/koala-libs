/*
 * This file is part of the koala project with MIT License.
 * Copyright (c) zhuguangxiang <zhuguangxiang@gmail.com>.
 */

#include "storage.h"
#include <ctype.h>
#include <errno.h>
#include <stdlib.h>
#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

#define STORAGE_DEFAULT_MAX_SIZE ((size_t)4 * 1024 * 1024 * 1024)

/* bytes held by the data areas, they live outside the gc heap and its budget */
static size_t storage_used_size;
/* max of storage_used_size, set by KOALA_STORAGE_MAX, e.g. 512M, 2G */
static size_t storage_max_size = STORAGE_DEFAULT_MAX_SIZE;
static int storage_max_parsed;

/* bytes, or with K/M/G suffix */
static size_t storage_parse_size(const char *s)
{
    char *end;
    errno = 0;
    unsigned long long n = strtoull(s, &end, 10);
    if (errno || end == s) panic("storage: invalid KOALA_STORAGE_MAX '%s'", s);

    int shift = 0;
    switch (toupper((unsigned char)*end)) {
        case 'K':
            shift = 10;
            end++;
            break;
        case 'M':
            shift = 20;
            end++;
            break;
        case 'G':
            shift = 30;
            end++;
            break;
        default:
            break;
    }
    if (*end || n == 0 || n > (SIZE_MAX >> shift)) {
        panic("storage: invalid KOALA_STORAGE_MAX '%s'", s);
    }
    return (size_t)n << shift;
}

static void storage_load_max(void)
{
    if (storage_max_parsed) return;
    storage_max_parsed = 1;
    const char *max = getenv("KOALA_STORAGE_MAX");
    if (max) storage_max_size = storage_parse_size(max);
}

/* a data area is rounded up to `align`, and never smaller than one alignment unit */
static size_t storage_alloc_size(size_t nbytes, size_t align)
{
    if (nbytes <= align) return align;
    return ((nbytes + align - 1) / align) * align;
}

Storage *storage_new(size_t nbytes, size_t align, Device dev)
{
    /* the device allocators want a power of two of at least sizeof(void *) */
    ASSERT(align >= 16 && (align & (align - 1)) == 0);
    storage_load_max();

    const DeviceDriver *driver = dev.driver;
    if (!driver || !driver->mem_ops) panic("storage: no driver for device kind %d", dev.kind);

    size_t size = storage_alloc_size(nbytes, align);
    if (size > storage_max_size || storage_used_size > storage_max_size - size) {
        /* let the device hand back whatever it cached, then try once more */
        driver->mem_ops->empty_cache();
        if (size > storage_max_size || storage_used_size > storage_max_size - size) {
            panic(
                "storage: out of memory on %s, used %zu + %zu bytes exceeds max %zu bytes, "
                "call gc.collect() or free the tensors, or raise KOALA_STORAGE_MAX",
                driver->dev_ops->name(dev.index), storage_used_size, size, storage_max_size);
        }
    }

    Storage *s = malloc(sizeof(Storage));
    if (!s) panic("storage: malloc %zu bytes failed", sizeof(Storage));

    s->data = driver->mem_ops->alloc(size, align, NULL);
    s->nbytes = nbytes;
    s->align = align;
    s->device = dev;
    storage_used_size += size;
    return s;
}

void storage_free(Storage *s)
{
    if (!s) return;
    if (s->data) {
        storage_used_size -= storage_alloc_size(s->nbytes, s->align);
        s->device.driver->mem_ops->free(s->data, NULL);
        s->data = NULL;
    }
    free(s);
}

size_t storage_used(Device dev)
{
    /* V1 only has the CPU, other devices get their own counters later */
    ASSERT(dev.kind == DEVICE_CPU && dev.index == 0);
    return storage_used_size;
}

size_t storage_max(Device dev)
{
    ASSERT(dev.kind == DEVICE_CPU && dev.index == 0);
    storage_load_max();
    return storage_max_size;
}

#ifdef __cplusplus
}
#endif