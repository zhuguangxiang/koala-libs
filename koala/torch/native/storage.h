/*
 * This file is part of the koala project with MIT License.
 * Copyright (c) zhuguangxiang <zhuguangxiang@gmail.com>.
 */

#ifndef _KOALA_STORAGE_H_
#define _KOALA_STORAGE_H_

#include "device.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * The data block behind a tensor.
 *
 * A plain struct owned by the tensor head: the head is the gc object, and its _free
 * hook releases this block and the data area. The data area comes from the memory ops
 * of `device.driver`, lives outside the gc heap and holds no references, so the
 * collector never scans it.
 */
typedef struct _Storage {
    void *data;
    size_t nbytes;
    size_t align;
    Device device;
} Storage;

/*
 * A storage of `nbytes` zeroed bytes aligned to `align`, a power of two of at least 16,
 * on `dev`. The header is a plain malloc block; the data area comes from the device
 * memory ops and counts against storage_max(), not against the gc heap budget.
 */
Storage *storage_new(size_t nbytes, size_t align, Device dev);

/* Release the data area and the header; the caller decides whether that is allowed. */
void storage_free(Storage *s);

/* bytes held by the data areas of `dev` */
size_t storage_used(Device dev);

/* the budget of `dev`: KOALA_STORAGE_MAX for the CPU, 4G by default */
size_t storage_max(Device dev);

#ifdef __cplusplus
}
#endif

#endif /* _KOALA_STORAGE_H_ */