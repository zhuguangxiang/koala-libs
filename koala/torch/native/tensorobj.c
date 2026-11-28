/*
 * This file is part of the koala project with MIT License.
 * Copyright (c) zhuguangxiang <zhuguangxiang@gmail.com>.
 */

#include "tensorobj.h"
#include <math.h>
#include "excobj.h"
#include "gc.h"
#include "listobj.h"
#include "tupleobj.h"

#ifdef __cplusplus
extern "C" {
#endif

/* V1-min has a single element type, float32 (TENSOR_F32), so the element size is a
 * constant. It becomes a table lookup when the other element types arrive. */
#define TENSOR_ITEMSIZE ((int64_t)sizeof(float))

/* every entry point starts here: a freed tensor must not be read */
static int tensor_check(TensorObject *t, const char *who)
{
    if (t->storage) return 1;
    raise_exc_fmt("%s: the tensor was freed", who);
    return 0;
}

/* row-major strides, counted in elements */
static void tensor_set_row_major(TensorObject *t)
{
    int64_t acc = 1;
    for (int i = t->ndim - 1; i >= 0; --i) {
        t->strides[i] = acc;
        acc *= t->shape[i];
    }
}

/* the element `pos` elements past the first one of the tensor */
static float *tensor_elem(TensorObject *t, int64_t pos)
{
    return (float *)t->storage->data + t->offset + pos;
}

void kl_tensor_free(TensorObject *t)
{
    ASSERT(IS_TENSOR(t));
    if (!t->storage) return;

    /* the head gives up its reference; ext_dims is a gc block, the collector frees it */
    storage_free(t->storage);
    t->storage = NULL;
}

static void tensor_free_hook(Object *obj) { kl_tensor_free((TensorObject *)obj); }

/*---------------------------------------------------------------------------+
 |  Creation                                                                 |
 +---------------------------------------------------------------------------*/

/* The element count of a shape. Raises on a bad rank, a negative size or an overflow. */
static int tensor_shape_numel(int ndim, const int64_t *shape, int64_t *numel, const char *who)
{
    if (ndim < 0 || ndim > TENSOR_MAX_NDIM) {
        raise_exc_fmt("%s: ndim %d is out of range [0, %d]", who, ndim, TENSOR_MAX_NDIM);
        return 0;
    }

    int64_t n = 1;
    for (int i = 0; i < ndim; ++i) {
        if (shape[i] < 0) {
            raise_exc_fmt("%s: dim %d is negative (%ld)", who, i, (long)shape[i]);
            return 0;
        }
        if (shape[i] != 0 && n > INT64_MAX / shape[i]) {
            raise_exc_fmt("%s: the shape holds too many elements", who);
            return 0;
        }
        n *= shape[i];
    }

    if (n > INT64_MAX / TENSOR_ITEMSIZE) {
        raise_exc_fmt("%s: the tensor is too large", who);
        return 0;
    }

    *numel = n;
    return 1;
}

/* The storage and the head of a contiguous zeroed tensor. `shape` is copied. Raises
 * and returns NULL on an invalid shape. */
static TensorObject *tensor_alloc(int ndim, const int64_t *shape, const char *who)
{
    int64_t numel;
    if (!tensor_shape_numel(ndim, shape, &numel, who)) return NULL;

    Device dev = device_make(DEVICE_CPU, 0);
    Storage *s = storage_new((size_t)(numel * TENSOR_ITEMSIZE), TENSOR_ALIGN, dev);

    TensorObject *t = gc_alloc_obj(t);
    INIT_OBJECT_HEAD(t, &tensor_type, 0);
    t->storage = s;
    t->offset = 0;
    t->numel = numel;
    t->ndim = ndim;
    t->dtype = TENSOR_F32;
    t->ext_dims = NULL;

    if (ndim <= TENSOR_INLINE_DIMS) {
        t->shape = t->dims;
        t->strides = t->dims + TENSOR_INLINE_DIMS;
    } else {
        /* one gc byte block holding the sizes then the strides; it has no reference */
        t->ext_dims = gc_alloc_bytes((int)(2 * ndim * sizeof(int64_t)));
        t->shape = t->ext_dims;
        t->strides = t->ext_dims + ndim;
    }

    for (int i = 0; i < ndim; ++i) {
        t->shape[i] = shape[i];
    }
    tensor_set_row_major(t);

    return t;
}

Object *kl_new_tensor(int ndim, const int64_t *shape)
{
    return (Object *)tensor_alloc(ndim, shape, "Tensor");
}

/* A variadic `shape ...int` arrives as one tuple argument. */
static int tensor_shape_arg(TValue *args, int nargs, int index, int64_t *shape, int *ndim,
                            const char *who)
{
    Object *ob = kl_arg_tuple(index);
    TValue *items = TUPLE_ITEMS(ob);
    int n = TUPLE_SIZE(ob);

    if (n > TENSOR_MAX_NDIM) {
        raise_exc_fmt("%s: ndim %d is out of range [0, %d]", who, n, TENSOR_MAX_NDIM);
        return 0;
    }

    for (int i = 0; i < n; ++i) {
        int64_t d = to_int64(items + i);
        if (d < 0) {
            raise_exc_fmt("%s: dim %d is negative (%ld)", who, i, (long)d);
            return 0;
        }
        shape[i] = d;
    }

    *ndim = n;
    return 1;
}

/* pub func zeros[U](shape ...int) Tensor[U] */
static TValue _tensor_zeros(TValue *self, TValue *args, int nargs)
{
    ASSERT(nargs == 1);

    int64_t shape[TENSOR_MAX_NDIM];
    int ndim;
    if (!tensor_shape_arg(args, nargs, 0, shape, &ndim, "torch.zeros")) return error_value;

    /* the storage comes zeroed */
    TensorObject *t = tensor_alloc(ndim, shape, "torch.zeros");
    if (!t) return error_value;
    return obj_value(t);
}

/* pub func ones[U](shape ...int) Tensor[U] */
static TValue _tensor_ones(TValue *self, TValue *args, int nargs)
{
    ASSERT(nargs == 1);

    int64_t shape[TENSOR_MAX_NDIM];
    int ndim;
    if (!tensor_shape_arg(args, nargs, 0, shape, &ndim, "torch.ones")) return error_value;

    TensorObject *t = tensor_alloc(ndim, shape, "torch.ones");
    if (!t) return error_value;

    for (int64_t i = 0; i < t->numel; ++i) {
        *tensor_elem(t, i) = 1.0f;
    }
    return obj_value(t);
}

/* pub func full[U](v float64, shape ...int) Tensor[U] */
static TValue _tensor_full(TValue *self, TValue *args, int nargs)
{
    ASSERT(nargs == 2);

    float v = (float)kl_arg_float64(0);
    int64_t shape[TENSOR_MAX_NDIM];
    int ndim;
    if (!tensor_shape_arg(args, nargs, 1, shape, &ndim, "torch.full")) return error_value;

    TensorObject *t = tensor_alloc(ndim, shape, "torch.full");
    if (!t) return error_value;

    for (int64_t i = 0; i < t->numel; ++i) {
        *tensor_elem(t, i) = v;
    }
    return obj_value(t);
}

/* pub func scalar[U](v float64) Tensor[U] */
static TValue _tensor_scalar(TValue *self, TValue *args, int nargs)
{
    ASSERT(nargs == 1);

    float v = (float)kl_arg_float64(0);
    TensorObject *t = tensor_alloc(0, NULL, "torch.scalar");
    if (!t) return error_value;

    *tensor_elem(t, 0) = v;
    return obj_value(t);
}

/* pub func arange[U](start float64, stop float64, step = 1.0) Tensor[U] */
static TValue _tensor_arange(TValue *self, TValue *args, int nargs)
{
    ASSERT(nargs == 2 || nargs == 3);

    double start = kl_arg_float64(0);
    double stop = kl_arg_float64(1);
    double step = 1.0;
    if (nargs == 3) step = kl_arg_float64(2);

    if (step == 0.0) {
        raise_exc_str("torch.arange: step must not be 0");
        return error_value;
    }

    int64_t n = (int64_t)ceil((stop - start) / step);
    if (n < 0) n = 0;

    int64_t shape[1] = { n };
    TensorObject *t = tensor_alloc(1, shape, "torch.arange");
    if (!t) return error_value;

    for (int64_t i = 0; i < n; ++i) {
        *tensor_elem(t, i) = (float)(start + (double)i * step);
    }
    return obj_value(t);
}

/* pub func linspace[U](start float64, stop float64, num = 50) Tensor[U] */
static TValue _tensor_linspace(TValue *self, TValue *args, int nargs)
{
    ASSERT(nargs == 2 || nargs == 3);

    double start = kl_arg_float64(0);
    double stop = kl_arg_float64(1);
    int64_t num = 50;
    if (nargs == 3) num = kl_arg_int64(2);

    if (num < 0) {
        raise_exc_fmt("torch.linspace: num %ld is negative", (long)num);
        return error_value;
    }

    int64_t shape[1] = { num };
    TensorObject *t = tensor_alloc(1, shape, "torch.linspace");
    if (!t) return error_value;

    if (num == 1) {
        *tensor_elem(t, 0) = (float)start;
        return obj_value(t);
    }

    double step = (stop - start) / (double)(num - 1);
    for (int64_t i = 0; i < num; ++i) {
        *tensor_elem(t, i) = (float)(start + (double)i * step);
    }
    /* the last value is exactly `stop`, not the accumulated rounding of the steps */
    if (num > 1) *tensor_elem(t, num - 1) = (float)stop;

    return obj_value(t);
}

/* pub func from_list[U](data list[float64], shape ...int) Tensor[U] */
static TValue _tensor_from_list(TValue *self, TValue *args, int nargs)
{
    ASSERT(nargs == 2);

    Object *data = kl_arg_obj_as(0, list_type);
    int len = kl_list_len(data);

    int64_t shape[TENSOR_MAX_NDIM];
    int ndim;
    if (!tensor_shape_arg(args, nargs, 1, shape, &ndim, "torch.from_list")) return error_value;

    /* without a shape the result is 1-D */
    if (ndim == 0) {
        shape[0] = len;
        ndim = 1;
    }

    /* check the count before allocating, a wrong shape must not take any memory */
    int64_t numel;
    if (!tensor_shape_numel(ndim, shape, &numel, "torch.from_list")) return error_value;
    if (numel != len) {
        raise_exc_fmt("torch.from_list: the shape holds %ld values, but data has %ld", (long)numel,
                      (long)len);
        return error_value;
    }

    TensorObject *t = tensor_alloc(ndim, shape, "torch.from_list");
    if (!t) return error_value;

    for (int i = 0; i < len; ++i) {
        TValue *item = kl_list_get(data, i);
        *tensor_elem(t, i) = (float)to_float64(item);
    }
    return obj_value(t);
}

/*---------------------------------------------------------------------------+
 |  Attributes                                                               |
 +---------------------------------------------------------------------------*/

/* pub func shape() list[int] */
static TValue _tensor_shape(TValue *self, TValue *args, int nargs)
{
    TensorObject *t = SELF_AS(tensor_type);
    ASSERT(nargs == 0);
    if (!tensor_check(t, "Tensor.shape")) return error_value;

    Object *list = kl_new_list();
    for (int i = 0; i < t->ndim; ++i) {
        kl_list_append(list, int64_value(t->shape[i]));
    }
    return obj_value(list);
}

/* pub func ndim() int */
static TValue _tensor_ndim(TValue *self, TValue *args, int nargs)
{
    TensorObject *t = SELF_AS(tensor_type);
    ASSERT(nargs == 0);
    if (!tensor_check(t, "Tensor.ndim")) return error_value;

    return int64_value(t->ndim);
}

/* pub func numel() int */
static TValue _tensor_numel(TValue *self, TValue *args, int nargs)
{
    TensorObject *t = SELF_AS(tensor_type);
    ASSERT(nargs == 0);
    if (!tensor_check(t, "Tensor.numel")) return error_value;

    return int64_value(t->numel);
}

/* pub func stride() list[int] */
static TValue _tensor_stride(TValue *self, TValue *args, int nargs)
{
    TensorObject *t = SELF_AS(tensor_type);
    ASSERT(nargs == 0);
    if (!tensor_check(t, "Tensor.stride")) return error_value;

    Object *list = kl_new_list();
    for (int i = 0; i < t->ndim; ++i) {
        kl_list_append(list, int64_value(t->strides[i]));
    }
    return obj_value(list);
}

/* pub func is_contiguous() bool */
static TValue _tensor_is_contiguous(TValue *self, TValue *args, int nargs)
{
    TensorObject *t = SELF_AS(tensor_type);
    ASSERT(nargs == 0);
    if (!tensor_check(t, "Tensor.is_contiguous")) return error_value;

    /* a dimension of size 1 is walked once, its stride does not matter */
    int64_t acc = 1;
    for (int i = t->ndim - 1; i >= 0; --i) {
        if (t->shape[i] != 1 && t->strides[i] != acc) return bool_value(0);
        acc *= t->shape[i];
    }
    return bool_value(1);
}

/*---------------------------------------------------------------------------+
 |  Scalars and output                                                       |
 +---------------------------------------------------------------------------*/

/* pub func item() float64 */
static TValue _tensor_item(TValue *self, TValue *args, int nargs)
{
    TensorObject *t = SELF_AS(tensor_type);
    ASSERT(nargs == 0);
    if (!tensor_check(t, "Tensor.item")) return error_value;

    if (t->numel != 1) {
        raise_exc_fmt("Tensor.item: the tensor holds %ld elements, not 1", (long)t->numel);
        return error_value;
    }
    return float64_value(*tensor_elem(t, 0));
}

/* pub func at(idx ...int) float64 */
static TValue _tensor_at(TValue *self, TValue *args, int nargs)
{
    TensorObject *t = SELF_AS(tensor_type);
    ASSERT(nargs == 1);
    if (!tensor_check(t, "Tensor.at")) return error_value;

    Object *ob = kl_arg_tuple(0);
    TValue *items = TUPLE_ITEMS(ob);
    int n = TUPLE_SIZE(ob);

    if (n != t->ndim) {
        raise_exc_fmt("Tensor.at: got %d indices for a tensor of %d dimensions", n, t->ndim);
        return error_value;
    }

    int64_t pos = 0;
    for (int i = 0; i < n; ++i) {
        int64_t idx = to_int64(items + i);
        /* a negative index counts from the end, as in a list */
        if (idx < 0) idx += t->shape[i];
        if (idx < 0 || idx >= t->shape[i]) {
            raise_exc_fmt("Tensor.at: index %ld is out of range for dim %d of size %ld",
                          (long)to_int64(items + i), i, (long)t->shape[i]);
            return error_value;
        }
        pos += idx * t->strides[i];
    }

    return float64_value(*tensor_elem(t, pos));
}

/* pub func tolist() list[float64] */
static TValue _tensor_tolist(TValue *self, TValue *args, int nargs)
{
    TensorObject *t = SELF_AS(tensor_type);
    ASSERT(nargs == 0);
    if (!tensor_check(t, "Tensor.tolist")) return error_value;

    Object *list = kl_new_list();
    int64_t idx[TENSOR_MAX_NDIM] = { 0 };

    /* row-major order of the view, so a strided tensor reads the same as a dense one */
    for (int64_t n = 0; n < t->numel; ++n) {
        int64_t pos = 0;
        for (int i = 0; i < t->ndim; ++i) {
            pos += idx[i] * t->strides[i];
        }
        kl_list_append(list, float64_value(*tensor_elem(t, pos)));

        for (int i = t->ndim - 1; i >= 0; --i) {
            idx[i]++;
            if (idx[i] < t->shape[i]) break;
            idx[i] = 0;
        }
    }

    return obj_value(list);
}

/* pub func free() */
static TValue _tensor_free(TValue *self, TValue *args, int nargs)
{
    TensorObject *t = SELF_AS(tensor_type);
    ASSERT(nargs == 0);

    /* No view shares a storage yet: every tensor of V1-min owns its own. The sharing
     * rule must be settled before reshape/transpose/narrow/select land, otherwise
     * freeing a view would release the data of its base. */
    kl_tensor_free(t);
    return no_value;
}

/* Fixed print options mirroring the torch defaults; there is no
 * set_printoptions yet. */
#define TENSOR_PRINT_PRECISION 4
#define TENSOR_PRINT_THRESHOLD 1000
#define TENSOR_PRINT_EDGEITEMS 3

static const char *tensor_dtype_name(uint8_t dtype)
{
    if (dtype == TENSOR_F32) return "f32";
    UNREACHABLE();
    return "?";
}

/* Render one element the torch way: an integer-valued float keeps a trailing
 * dot (`1.`), otherwise fixed precision (`1.5000`); NaN/Inf become `nan`/`inf`/
 * `-inf`. `int_mode` is decided over the whole tensor beforehand. The value is
 * a float32 widened only for snprintf, so %.4f hides its binary noise (0.1 ->
 * 0.1000, not 0.1000000014901161). */
static int tensor_elem_str(char *buf, float v, int int_mode)
{
    if (isnan(v)) {
        buf[0] = 'n';
        buf[1] = 'a';
        buf[2] = 'n';
        buf[3] = '\0';
        return 3;
    }
    if (isinf(v)) {
        if (v > 0) {
            buf[0] = 'i';
            buf[1] = 'n';
            buf[2] = 'f';
            buf[3] = '\0';
            return 3;
        }
        buf[0] = '-';
        buf[1] = 'i';
        buf[2] = 'n';
        buf[3] = 'f';
        buf[4] = '\0';
        return 4;
    }
    if (int_mode) {
        int n = snprintf(buf, 64, "%.0f", (double)v);
        buf[n] = '.';
        buf[n + 1] = '\0';
        return n + 1;
    }
    return snprintf(buf, 64, "%.*f", TENSOR_PRINT_PRECISION, (double)v);
}

typedef struct {
    /* every finite value is integer-valued: print `1.` instead of `1.0000` */
    int int_mode;
    /* widest element string; elements are right-aligned to it */
    int max_width;
} TensorPrintFmt;

/* Two row-major passes over the view: first decide int_mode, then measure the
 * widest rendered element with that mode fixed. */
static void tensor_compute_fmt(TensorObject *t, TensorPrintFmt *pf)
{
    pf->int_mode = 1;
    pf->max_width = 0;

    int64_t idx[TENSOR_MAX_NDIM];
    for (int i = 0; i < TENSOR_MAX_NDIM; ++i) idx[i] = 0;
    char buf[64];

    for (int64_t n = 0; n < t->numel; ++n) {
        int64_t pos = 0;
        for (int i = 0; i < t->ndim; ++i) pos += idx[i] * t->strides[i];
        float v = *tensor_elem(t, pos);
        if (!isnan(v) && !isinf(v) && v != floorf(v)) pf->int_mode = 0;
        for (int i = t->ndim - 1; i >= 0; --i) {
            idx[i]++;
            if (idx[i] < t->shape[i]) break;
            idx[i] = 0;
        }
    }

    for (int i = 0; i < TENSOR_MAX_NDIM; ++i) idx[i] = 0;
    for (int64_t n = 0; n < t->numel; ++n) {
        int64_t pos = 0;
        for (int i = 0; i < t->ndim; ++i) pos += idx[i] * t->strides[i];
        float v = *tensor_elem(t, pos);
        int len = tensor_elem_str(buf, v, pf->int_mode);
        if (len > pf->max_width) pf->max_width = len;
        for (int i = t->ndim - 1; i >= 0; --i) {
            idx[i]++;
            if (idx[i] < t->shape[i]) break;
            idx[i] = 0;
        }
    }
}

typedef struct {
    TensorObject *t;
    Formatter *fmt;
    TensorPrintFmt pf;
    /* numel exceeds the threshold: large dimensions collapse to `...` */
    int summarize;
} TensorPrintCtx;

static void tensor_fmt_elem(TensorPrintCtx *ctx, int64_t pos)
{
    char buf[64];
    float v = *tensor_elem(ctx->t, pos);
    int len = tensor_elem_str(buf, v, ctx->pf.int_mode);
    for (int i = len; i < ctx->pf.max_width; ++i) buf_write_char(&ctx->fmt->buf, ' ');
    buf_write_nstr(&ctx->fmt->buf, buf, len);
}

/* the inline `...` of a summarized last dimension, aligned like an element */
static void tensor_fmt_ellipsis(TensorPrintCtx *ctx)
{
    int w = ctx->pf.max_width;
    if (w < 3) w = 3;
    for (int i = 3; i < w; ++i) buf_write_char(&ctx->fmt->buf, ' ');
    buf_write_str(&ctx->fmt->buf, "...");
}

/* One dimension per level. The last dimension stays inline (`[1., 2.]`); higher
 * ones put each sub-block on its own line, indented to align under the opening
 * bracket, with a blank line between the blocks of a 3-D+ tensor. `indent` is
 * the column of the continuation lines at this level. A 0-dim tensor (dim ==
 * ndim at the top) is its bare value. */
static void tensor_fmt_dim(TensorPrintCtx *ctx, int dim, int64_t pos, int indent)
{
    TensorObject *t = ctx->t;
    Formatter *fmt = ctx->fmt;

    if (dim == t->ndim) {
        tensor_fmt_elem(ctx, pos);
        return;
    }

    int64_t n = t->shape[dim];
    int last = (dim == t->ndim - 1);
    int sum = ctx->summarize && n > 2 * TENSOR_PRINT_EDGEITEMS;
    int nitems = n;
    if (sum) nitems = 2 * TENSOR_PRINT_EDGEITEMS + 1;

    buf_write_char(&fmt->buf, '[');
    for (int k = 0; k < nitems; ++k) {
        if (k > 0) {
            if (last) {
                buf_write_str(&fmt->buf, ", ");
            } else {
                buf_write_str(&fmt->buf, ",\n");
                if (t->ndim - dim >= 3) buf_write_char(&fmt->buf, '\n');
                for (int s = 0; s < indent; ++s) buf_write_char(&fmt->buf, ' ');
            }
        }

        if (sum && k == TENSOR_PRINT_EDGEITEMS) {
            if (last) {
                tensor_fmt_ellipsis(ctx);
            } else {
                buf_write_str(&fmt->buf, "...");
            }
            continue;
        }

        int64_t i = k;
        if (sum && k > TENSOR_PRINT_EDGEITEMS) {
            i = n - TENSOR_PRINT_EDGEITEMS + (k - TENSOR_PRINT_EDGEITEMS - 1);
        }
        tensor_fmt_dim(ctx, dim + 1, pos + i * t->strides[dim], indent + 1);
    }
    buf_write_char(&fmt->buf, ']');
}

/* pub func fmt(f Formatter) */
static TValue _tensor_fmt(TValue *self, TValue *args, int nargs)
{
    TensorObject *t = SELF_AS(tensor_type);
    ASSERT(nargs == 1);

    Formatter *fmt = kl_arg_fmt(0);

    /* printing a freed tensor says so instead of raising: it is what one does while
     * looking for the very bug that freed it */
    if (!t->storage) {
        buf_write_str(&fmt->buf, "Tensor[freed]");
        return no_value;
    }

    /* prefix: Tensor[dtype](shape)( */
    buf_write_str(&fmt->buf, "Tensor[");
    buf_write_str(&fmt->buf, tensor_dtype_name(t->dtype));
    buf_write_str(&fmt->buf, "](");
    for (int i = 0; i < t->ndim; ++i) {
        if (i > 0) buf_write_str(&fmt->buf, ", ");
        buf_write_int64_str(&fmt->buf, t->shape[i]);
    }
    buf_write_str(&fmt->buf, ")(");

    TensorPrintCtx ctx;
    ctx.t = t;
    ctx.fmt = fmt;
    tensor_compute_fmt(t, &ctx.pf);
    ctx.summarize = (t->numel > TENSOR_PRINT_THRESHOLD);

    /* continuation lines align one column past the opening bracket; the column
     * is measured from the last newline so an embedded `format("a={}", t)`
     * aligns to where the tensor actually starts */
    int col = 0;
    char *s = BUF_STR(fmt->buf);
    int len = BUF_LEN(fmt->buf);
    for (int i = len - 1; i >= 0; --i) {
        if (s[i] == '\n') break;
        col++;
    }

    tensor_fmt_dim(&ctx, 0, 0, col + 1);
    buf_write_char(&fmt->buf, ')');

    return no_value;
}

static MethodDef _tensor_methods[] = {
    { "shape", _tensor_shape },
    { "ndim", _tensor_ndim },
    { "numel", _tensor_numel },
    { "stride", _tensor_stride },
    { "is_contiguous", _tensor_is_contiguous },
    { "item", _tensor_item },
    { "at", _tensor_at },
    { "tolist", _tensor_tolist },
    { "free", _tensor_free },
    { "fmt", _tensor_fmt },
    { NULL },
};

/* pub class Tensor[T] { ... }, the class of torch/tensor.kl binds here */
TypeObject tensor_type = {
    ._type = &type_type,
    .name = "Tensor",
    .flags = TP_FLAGS_CLASS,
    .objsize = sizeof(TensorObject),
    .methdefs = _tensor_methods,
    ._free = tensor_free_hook,
};

/* the factories are top-level functions of koala/torch, not methods of Tensor */
void tensor_reg_factories(NativeLib *lib)
{
    kl_reg_func(lib, "zeros", _tensor_zeros);
    kl_reg_func(lib, "ones", _tensor_ones);
    kl_reg_func(lib, "full", _tensor_full);
    kl_reg_func(lib, "arange", _tensor_arange);
    kl_reg_func(lib, "linspace", _tensor_linspace);
    kl_reg_func(lib, "from_list", _tensor_from_list);
    kl_reg_func(lib, "scalar", _tensor_scalar);
}

#ifdef __cplusplus
}
#endif