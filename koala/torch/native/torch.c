/*
 * This file is part of the koala project with MIT License.
 * Copyright (c) zhuguangxiang <zhuguangxiang@gmail.com>.
 */

#include "tensorobj.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Entry point called by the runtime after `dlopen("libkoala_torch.so")`.
///
/// The name must be `<last segment of the package path>_native_lib_init`,
/// so `koala/torch` becomes `torch_native_lib_init`.
void torch_native_lib_init(NativeLib *lib)
{
    kl_reg_type(lib, &tensor_type);
    tensor_reg_factories(lib);
}

#ifdef __cplusplus
}
#endif
