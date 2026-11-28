/*
 * This file is part of the koala project with MIT License.
 * Copyright (c) zhuguangxiang <zhuguangxiang@gmail.com>.
 */

#include <stdlib.h>
#include <string.h>
#include "common.h"
#include "device.h"

#ifdef __cplusplus
extern "C" {
#endif

static int cpu_count(void) { return 1; }

static const char *cpu_name(int index)
{
    ASSERT(index == 0);
    return "cpu";
}

static DeviceCaps cpu_caps(int index)
{
    ASSERT(index == 0);
    DeviceCaps caps;
    /* the CPU holds every element type */
    caps.dtypes = UINT32_MAX;
    caps.align = 64;
    caps.unified_memory = 1;
    caps.graph_only = 0;
    return caps;
}

static void *cpu_alloc(size_t nbytes, size_t align, DeviceStream stream)
{
    /* the CPU runs synchronously, it has no streams */
    ASSERT(stream == NULL);
    ASSERT(align >= 16 && (align & (align - 1)) == 0);

    /* aligned_alloc wants a multiple of the alignment */
    size_t size = align;
    if (nbytes > align) size = ((nbytes + align - 1) / align) * align;

    void *ptr = aligned_alloc(align, size);
    if (!ptr) panic("cpu: aligned_alloc %zu bytes failed", size);
    memset(ptr, 0, size);
    return ptr;
}

static void cpu_free(void *ptr, DeviceStream stream)
{
    ASSERT(stream == NULL);
    free(ptr);
}

/* there is no cache to release yet */
static void cpu_empty_cache(void) {}

static DeviceMemStats cpu_stats(void)
{
    DeviceMemStats stats = { 0, 0, 0, 0 };
    return stats;
}

static void cpu_copy(void *dst, const void *src, size_t nbytes, DeviceCopyKind kind,
                     DeviceStream stream)
{
    ASSERT(stream == NULL);
    ASSERT(kind == DEVICE_COPY_H2H);
    memcpy(dst, src, nbytes);
}

/* streams and events are empty on the CPU: everything is already complete */
static DeviceStream cpu_stream_create(void) { return NULL; }
static void cpu_stream_sync(DeviceStream stream) { ASSERT(stream == NULL); }
static DeviceStream cpu_default_stream(void) { return NULL; }
static DeviceEvent cpu_event_record(DeviceStream stream)
{
    ASSERT(stream == NULL);
    return NULL;
}
static int cpu_event_query(DeviceEvent event)
{
    ASSERT(event == NULL);
    return 1;
}
static void cpu_event_sync(DeviceEvent event) { ASSERT(event == NULL); }

static void cpu_launch(DeviceKernel kernel, void *args, DeviceStream stream)
{
    ASSERT(stream == NULL);
    kernel(args);
}

static const DeviceOps cpu_dev_ops = {
    .count = cpu_count,
    .name = cpu_name,
    .caps = cpu_caps,
};

/* aligned_alloc(64) for now, the caching allocator of the design comes later */
static const DeviceMemOps cpu_mem_ops = {
    .alloc = cpu_alloc,
    .free = cpu_free,
    .empty_cache = cpu_empty_cache,
    .stats = cpu_stats,
};

static const DeviceCopyOps cpu_copy_ops = {
    .copy = cpu_copy,
};

static const DeviceSyncOps cpu_sync_ops = {
    .stream_create = cpu_stream_create,
    .stream_sync = cpu_stream_sync,
    .default_stream = cpu_default_stream,
    .event_record = cpu_event_record,
    .event_query = cpu_event_query,
    .event_sync = cpu_event_sync,
};

static const DeviceExecOps cpu_exec_ops = {
    .launch = cpu_launch,
    /* whole graphs come with the B1 backend in V3 */
    .graph_compile = NULL,
    .graph_run = NULL,
};

DeviceDriver cpu_driver = {
    .dev_ops = &cpu_dev_ops,
    .mem_ops = &cpu_mem_ops,
    .copy_ops = &cpu_copy_ops,
    .sync_ops = &cpu_sync_ops,
    .exec_ops = &cpu_exec_ops,
};

static DeviceDriver *device_driver(Device dev)
{
    if (dev.kind == DEVICE_CPU && dev.index == 0) return &cpu_driver;
    return NULL;
}

Device device_make(DeviceKind kind, int32_t index)
{
    Device dev;
    dev.kind = kind;
    dev.index = index;
    dev.driver = device_driver(dev);
    return dev;
}

#ifdef __cplusplus
}
#endif
