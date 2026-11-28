/*
 * This file is part of the koala project with MIT License.
 * Copyright (c) zhuguangxiang <zhuguangxiang@gmail.com>.
 */

#ifndef _KOALA_DEVICE_H_
#define _KOALA_DEVICE_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* device kinds; only cpu is implemented in V1 */
typedef enum _DeviceKind {
    DEVICE_CPU = 0,
    DEVICE_CUDA,
    DEVICE_ROCM,
    DEVICE_NPU,
    DEVICE_METAL,
    DEVICE_VULKAN,
} DeviceKind;

/* the ops tables of one backend, defined at the bottom of this file */
typedef struct _DeviceDriver DeviceDriver;

/* {kind, index} plus the driver serving it; two devices are equal when kind and index
   are, the driver is derived from them */
typedef struct _Device {
    uint8_t kind;
    int32_t index;
    /* how this device allocates, copies, synchronizes and launches, NULL when the
       backend is not built in */
    const DeviceDriver *driver;
} Device;

/* opaque handles of the backend */
typedef void *DeviceStream;
typedef void *DeviceEvent;

/* direction of DeviceBackend.copy */
typedef enum _DeviceCopyKind {
    DEVICE_COPY_H2D = 0,
    DEVICE_COPY_D2H,
    DEVICE_COPY_D2D,
    DEVICE_COPY_H2H,
} DeviceCopyKind;

/* what a device supports: a mask of element types, the alignment of its allocations,
   whether memory is unified with the host, and whether it only runs whole graphs */
typedef struct _DeviceCaps {
    uint32_t dtypes;
    size_t align;
    int unified_memory;
    int graph_only;
} DeviceCaps;

/* memory of a device, see the CPU allocator in the design */
typedef struct _DeviceMemStats {
    size_t allocated;
    size_t reserved;
    size_t peak;
    size_t num_alloc;
} DeviceMemStats;

/* a kernel the device runs */
typedef void (*DeviceKernel)(void *args);

/*
 * The ops tables of a device, frozen in V1. They are split by who holds them, the way
 * the kernel splits super_operations / inode_operations / address_space_operations:
 * a storage only ever needs mem_ops, it never sees streams, events or kernels.
 * A driver leaves the tables it does not support NULL.
 */

/* the device itself: how many, what they are called, what they can do */
typedef struct _DeviceOps {
    int (*count)(void);
    const char *(*name)(int index);
    DeviceCaps (*caps)(int index);
} DeviceOps;

/* the data area of a storage */
typedef struct _DeviceMemOps {
    void *(*alloc)(size_t nbytes, size_t align, DeviceStream stream);
    void (*free)(void *ptr, DeviceStream stream);
    void (*empty_cache)(void);
    DeviceMemStats (*stats)(void);
} DeviceMemOps;

/* moving bytes between two memories */
typedef struct _DeviceCopyOps {
    void (*copy)(void *dst, const void *src, size_t nbytes, DeviceCopyKind kind,
                 DeviceStream stream);
} DeviceCopyOps;

/* streams and events: when work is done */
typedef struct _DeviceSyncOps {
    DeviceStream (*stream_create)(void);
    void (*stream_sync)(DeviceStream stream);
    DeviceStream (*default_stream)(void);
    DeviceEvent (*event_record)(DeviceStream stream);
    int (*event_query)(DeviceEvent event);
    void (*event_sync)(DeviceEvent event);
} DeviceSyncOps;

/* running kernels and whole graphs */
typedef struct _DeviceExecOps {
    void (*launch)(DeviceKernel kernel, void *args, DeviceStream stream);
    /* optional, provided by the B1 backend in V3 */
    void *(*graph_compile)(void *tir);
    void (*graph_run)(void *graph, void **inputs, void **outputs, DeviceStream stream);
} DeviceExecOps;

/* one per device kind, carrying its ops tables the way net_device carries netdev_ops */
struct _DeviceDriver {
    const DeviceOps *dev_ops;
    const DeviceMemOps *mem_ops;
    const DeviceCopyOps *copy_ops;
    const DeviceSyncOps *sync_ops;
    const DeviceExecOps *exec_ops;
};

/* the CPU driver, the only one of V1 */
extern DeviceDriver cpu_driver;

/* a device of `kind` and `index` with its driver filled in; `driver` stays NULL when
   no driver is registered for it, so callers can probe for a backend */
Device device_make(DeviceKind kind, int32_t index);

#ifdef __cplusplus
}
#endif

#endif /* _KOALA_DEVICE_H_ */