/*
 * This file is part of the koala project with MIT License.
 * Copyright (c) zhuguangxiang <zhuguangxiang@gmail.com>.
 */

#ifndef _KOALA_TENSOR_OBJECT_H_
#define _KOALA_TENSOR_OBJECT_H_

#include "object.h"
#include "storage.h"

#ifdef __cplusplus
extern "C" {
#endif

/* at most 64 dimensions, as NumPy 2; the first 6 of them are stored inline */
#define TENSOR_MAX_NDIM 64
#define TENSOR_INLINE_DIMS 6

/* alignment of a storage data area on the CPU */
#define TENSOR_ALIGN 64

/* element types; V1-min only has float32 */
#define TENSOR_F32 0

typedef struct _TensorObject {
    OBJECT_HEAD
    /* NULL once free() released it */
    Storage *storage;
    /* sizes and strides, in elements; they point into dims or ext_dims */
    int64_t *shape;
    int64_t *strides;
    /* index of the first element in the storage, in elements */
    int64_t offset;
    int64_t numel;
    int32_t ndim;
    /* TENSOR_XXX */
    uint8_t dtype;
    /* the first 6 dimensions: the sizes, then the strides */
    int64_t dims[2 * TENSOR_INLINE_DIMS];
    /* all dimensions when ndim > 6, NULL otherwise */
    int64_t *ext_dims;
} TensorObject;

extern TypeObject tensor_type;
#define IS_TENSOR(ob) IS_TYPE((ob), &tensor_type)

/* A contiguous, zeroed float32 tensor of the given shape. */
Object *kl_new_tensor(int ndim, const int64_t *shape);

/* Release the storage when it is not shared, then the head; `t` must not be used afterwards. */
void kl_tensor_free(TensorObject *t);

/* the first element of the tensor */
static inline float *tensor_data(TensorObject *t)
{
    return (float *)t->storage->data + t->offset;
}

/* register the factories (zeros, ones, ...) as top-level functions of koala/torch */
void tensor_reg_factories(NativeLib *lib);

#ifdef __cplusplus
}
#endif

#endif /* _KOALA_TENSOR_OBJECT_H_ */