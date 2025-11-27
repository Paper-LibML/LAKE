/*
 * Part of LAKE: Towards a Machine Learning-Assisted Kernel with LAKE
 * Copyright (C) 2022-2024 Henrique Fingler
 * Copyright (C) 2022-2024 Isha Tarte
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */


#include <linux/types.h>
#include <linux/module.h>
#include <linux/vmalloc.h>
#include "commands.h"
#include "lake_kapi.h"
#include "lake_shm.h"
#include "kargs.h"

/*
 *
 *   Functions in this file export CUDA symbols.
 *   In general they fill a struct and send it through netlink.
 *   They also choose if they are sync or async calls.
 *   Some have special handling, such as memcpys
 * 
 *   TODO: support netlink copies (not urgent)
 *   TODO: accumulate errors
 */

CUresult CUDAAPI cuInit(unsigned int flags) {
    struct lake_cmd_ret ret;
	struct lake_cmd_cuInit cmd = {
        .API_ID = LAKE_API_cuInit, .flags = flags,
    };
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC, &ret);
	return ret.res;
}
EXPORT_SYMBOL(cuInit);

CUresult CUDAAPI cuDeviceGet(CUdevice *device, int ordinal) {
    struct lake_cmd_ret ret;
	struct lake_cmd_cuDeviceGet cmd = {
        .API_ID = LAKE_API_cuDeviceGet, .ordinal = ordinal,
    };
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC, &ret);
    *device = ret.device;
	return ret.res;
}
EXPORT_SYMBOL(cuDeviceGet);

CUresult CUDAAPI cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev) {
    struct lake_cmd_ret ret;
	struct lake_cmd_cuCtxCreate cmd = {
        .API_ID = LAKE_API_cuCtxCreate, .flags = flags, .dev = dev
    };
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC, &ret);
    *pctx = ret.pctx;
	return ret.res;
}
EXPORT_SYMBOL(cuCtxCreate);


CUresult CUDAAPI cuCtxDestroy(CUcontext pctx) {
    struct lake_cmd_ret ret;
	struct lake_cmd_cuCtxDestroy cmd = {
        .API_ID = LAKE_API_cuCtxDestroy, .ctx = pctx,
    };
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC, &ret);
	return ret.res;
}
EXPORT_SYMBOL(cuCtxDestroy);

CUresult CUDAAPI cuModuleLoad(CUmodule *module, const char *fname) {
    struct lake_cmd_ret ret;
	struct lake_cmd_cuModuleLoad cmd = {
        .API_ID = LAKE_API_cuModuleLoad
    };
    strcpy(cmd.fname, fname);
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC, &ret);
    *module = ret.module;
	return ret.res;
}
EXPORT_SYMBOL(cuModuleLoad);

CUresult CUDAAPI cuModuleUnload(CUmodule hmod) {
    struct lake_cmd_ret ret;
	struct lake_cmd_cuModuleUnload cmd = {
        .API_ID = LAKE_API_cuModuleUnload, .hmod = hmod
    };
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC, &ret);
	return ret.res;
}
EXPORT_SYMBOL(cuModuleUnload);

CUresult CUDAAPI cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name) {
    struct kernel_args_metadata* meta;
    struct lake_cmd_ret ret;
	struct lake_cmd_cuModuleGetFunction cmd = {
        .API_ID = LAKE_API_cuModuleGetFunction, .hmod = hmod
    };
    strcpy(cmd.name, name);
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC, &ret);
    *hfunc = ret.func;

    //parse and store kargs
    meta = get_kargs(*hfunc);
    kava_parse_function_args(name, meta);

    return ret.res;
}
EXPORT_SYMBOL(cuModuleGetFunction);

CUresult CUDAAPI cuLaunchKernel(CUfunction f,
                                unsigned int gridDimX,
                                unsigned int gridDimY,
                                unsigned int gridDimZ,
                                unsigned int blockDimX,
                                unsigned int blockDimY,
                                unsigned int blockDimZ,
                                unsigned int sharedMemBytes,
                                CUstream hStream,
                                void **kernelParams,
                                void **extra) {
    struct lake_cmd_ret ret;
    struct kernel_args_metadata* meta = get_kargs(f);
    u32 tsize = sizeof(struct lake_cmd_cuLaunchKernel) + meta->total_size;
    void* cmd_and_args = vmalloc(tsize);
	struct lake_cmd_cuLaunchKernel *cmd = (struct lake_cmd_cuLaunchKernel*) cmd_and_args;
    u8 *args = cmd_and_args + sizeof(struct lake_cmd_cuLaunchKernel);

    cmd->API_ID = LAKE_API_cuLaunchKernel; cmd->f = f; 
    cmd->gridDimX = gridDimX; cmd->gridDimY = gridDimY; cmd->gridDimZ = gridDimZ;
    cmd->blockDimX = blockDimX; cmd->blockDimY = blockDimY; cmd->blockDimZ = blockDimZ;
    cmd->sharedMemBytes = sharedMemBytes; cmd->hStream = hStream; cmd->extra = 0;

    cmd->paramsSize = meta->total_size;
    serialize_args(meta, args, kernelParams);

    lake_send_cmd(cmd_and_args, tsize, CMD_ASYNC, &ret);
    vfree(cmd_and_args);
	return ret.res;
}
EXPORT_SYMBOL(cuLaunchKernel);

CUresult CUDAAPI cuMemAlloc(CUdeviceptr *dptr, size_t bytesize) {
    struct lake_cmd_ret ret;
	struct lake_cmd_cuMemAlloc cmd = {
        .API_ID = LAKE_API_cuMemAlloc, .bytesize = bytesize
    };
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC, &ret);
    *dptr = ret.ptr;
	return ret.res;
}
EXPORT_SYMBOL(cuMemAlloc);

CUresult CUDAAPI cuMemcpyHtoD(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount) {
    struct lake_cmd_ret ret;
	struct lake_cmd_cuMemcpyHtoD cmd = {
        .API_ID = LAKE_API_cuMemcpyHtoD, .dstDevice = dstDevice, .srcHost = srcHost,
        .ByteCount = ByteCount
    };

    s64 offset = kava_shm_offset(srcHost);
    if (offset < 0) {
        pr_err("srcHost in cuMemcpyHtoD is NOT a kshm pointer (use kava_alloc to fix it)\n");
        return CUDA_ERROR_INVALID_VALUE;
    }
    cmd.srcHost = (void*)offset;
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC, &ret);
	return ret.res;
}
EXPORT_SYMBOL(cuMemcpyHtoD);

CUresult CUDAAPI cuMemcpyDtoH(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
    struct lake_cmd_ret ret;
	struct lake_cmd_cuMemcpyDtoH cmd = {
        .API_ID = LAKE_API_cuMemcpyDtoH, .srcDevice = srcDevice,
        .ByteCount = ByteCount
    };

    s64 offset = kava_shm_offset(dstHost);
    if (offset < 0) {
        pr_err("dstHost in cuMemcpyDtoH is NOT a kshm pointer (use kava_alloc to fix it)\n");
        return CUDA_ERROR_INVALID_VALUE;
    }
    cmd.dstHost = (void*)offset;
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC, &ret);
	return ret.res;
}
EXPORT_SYMBOL(cuMemcpyDtoH);

CUresult CUDAAPI cuCtxSynchronize(void) {
    struct lake_cmd_ret ret;
	struct lake_cmd_cuCtxSynchronize cmd = {
        .API_ID = LAKE_API_cuCtxSynchronize,
    };
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC, &ret);
	return ret.res;
}
EXPORT_SYMBOL(cuCtxSynchronize);

CUresult CUDAAPI cuMemFree(CUdeviceptr dptr) {
    struct lake_cmd_ret ret;
	struct lake_cmd_cuMemFree cmd = {
        .API_ID = LAKE_API_cuMemFree, .dptr = dptr
    };
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC, &ret);
	return ret.res;
}
EXPORT_SYMBOL(cuMemFree);

CUresult CUDAAPI cuStreamCreate(CUstream *phStream, unsigned int Flags) {
    struct lake_cmd_ret ret;
	struct lake_cmd_cuStreamCreate cmd = {
        .API_ID = LAKE_API_cuStreamCreate, .Flags = Flags
    };
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC, &ret);
    *phStream = ret.stream;
	return ret.res;
}
EXPORT_SYMBOL(cuStreamCreate);

CUresult CUDAAPI cuStreamDestroy (CUstream hStream) {
    struct lake_cmd_ret ret;
	struct lake_cmd_cuStreamDestroy cmd = {
        .API_ID = LAKE_API_cuStreamDestroy, .hStream = hStream
    };
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC, &ret);
	return ret.res;
}
EXPORT_SYMBOL(cuStreamDestroy);

CUresult CUDAAPI cuStreamSynchronize(CUstream hStream) {
    struct lake_cmd_ret ret;
	struct lake_cmd_cuStreamSynchronize cmd = {
        .API_ID = LAKE_API_cuStreamSynchronize, .hStream = hStream
    };
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC, &ret);
	return ret.res;
}
EXPORT_SYMBOL(cuStreamSynchronize);

CUresult CUDAAPI cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount, CUstream hStream) {
    struct lake_cmd_ret ret;
	struct lake_cmd_cuMemcpyHtoDAsync cmd = {
        .API_ID = LAKE_API_cuMemcpyHtoDAsync, .dstDevice = dstDevice, .srcHost = srcHost, 
        .ByteCount = ByteCount, .hStream = hStream
    };
    s64 offset = kava_shm_offset(srcHost);
    if (offset < 0) {
        pr_err("srcHost in cuMemcpyHtoDAsync is NOT a kshm pointer (use kava_alloc to fix it)\n");
        return CUDA_ERROR_INVALID_VALUE;
    }
    cmd.srcHost = (void*)offset;

    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_ASYNC, &ret);
	return ret.res;
}
EXPORT_SYMBOL(cuMemcpyHtoDAsync);

CUresult CUDAAPI cuMemcpyDtoHAsync(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream) {
    struct lake_cmd_ret ret;
	struct lake_cmd_cuMemcpyDtoHAsync cmd = {
        .API_ID = LAKE_API_cuMemcpyDtoHAsync, .dstHost = dstHost, .srcDevice = srcDevice,
        .ByteCount = ByteCount, .hStream = hStream
    };
    
    s64 offset = kava_shm_offset(dstHost);
    if (offset < 0) {
        pr_err("dstHost in cuMemcpyDtoHAsync is NOT a kshm pointer (use kava_alloc to fix it)\n");
        return CUDA_ERROR_INVALID_VALUE;
    }
    cmd.dstHost = (void*)offset;

    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_ASYNC, &ret);
	return ret.res;
}
EXPORT_SYMBOL(cuMemcpyDtoHAsync);

CUresult CUDAAPI cuMemAllocPitch(CUdeviceptr* dptr, size_t* pPitch, 
        size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes) {
    struct lake_cmd_ret ret;
	struct lake_cmd_cuMemAllocPitch cmd = {
        .API_ID = LAKE_API_cuMemAllocPitch, .WidthInBytes = WidthInBytes,
        .Height = Height, .ElementSizeBytes = ElementSizeBytes
    };
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC, &ret);
    *dptr = ret.ptr;
    *pPitch = ret.pPitch;
	return ret.res;
}
EXPORT_SYMBOL(cuMemAllocPitch);


/*
 *  Kleio
 */

CUresult CUDAAPI kleioLoadModel(const void *srcHost, size_t len) {
    struct lake_cmd_ret ret;
	struct lake_cmd_kleioLoadModel cmd = {
        .API_ID = LAKE_API_kleioLoadModel
    };

    // s64 offset = kava_shm_offset(srcHost);
    // if (offset < 0) {
    //     pr_err("srcHost in kleioLoadModel is NOT a kshm pointer (use kava_alloc to fix it)\n");
    //     return CUDA_ERROR_INVALID_VALUE;
    // }
    // cmd.srcHost = (void*)offset;
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC, &ret);
	return ret.res;
}
EXPORT_SYMBOL(kleioLoadModel);

CUresult CUDAAPI kleioInference(const void *srcHost, size_t len, int use_gpu) {
    struct lake_cmd_ret ret;
	struct lake_cmd_kleioInference cmd = {
        .API_ID = LAKE_API_kleioInference, .len = len,
        .use_gpu = use_gpu
    };
    // s64 offset = kava_shm_offset(srcHost);
    // if (offset < 0) {
    //     pr_err("srcHost in kleioInference is NOT a kshm pointer (use kava_alloc to fix it)\n");
    //     return CUDA_ERROR_INVALID_VALUE;
    // }
    // cmd.srcHost = (void*)offset;
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC, &ret);
	return ret.res;
}
EXPORT_SYMBOL(kleioInference);

CUresult CUDAAPI kleioForceGC(void) {
    struct lake_cmd_ret ret;
	struct lake_cmd_kleioForceGC cmd = {
        .API_ID = LAKE_API_kleioForceGC,
    };
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC, &ret);
	return ret.res;
}
EXPORT_SYMBOL(kleioForceGC);

CUresult CUDAAPI nvmlRunningProcs(int* nproc) {
    struct lake_cmd_ret ret;
	struct lake_cmd_nvmlRunningProcs cmd = {
        .API_ID = LAKE_API_nvmlRunningProcs,
    };
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC, &ret);
    *nproc = (int)ret.ptr;
	return ret.res;
}
EXPORT_SYMBOL(nvmlRunningProcs);

CUresult CUDAAPI nvmlUtilRate(int* nproc) {
    struct lake_cmd_ret ret;
	struct lake_cmd_nvmlUtilRate cmd = {
        .API_ID = LAKE_API_nvmlUtilRate,
    };
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC, &ret);
    *nproc = (int)ret.ptr;
	return ret.res;
}
EXPORT_SYMBOL(nvmlUtilRate);

int dataset_from_csv(struct dataset *ds, char *filename, 
                     char *delim, int n_cols, enum type_t data_type,
                     int headers) {
    struct lake_cmd_ret ret;

    s64 ds_offset = kava_shm_offset(ds);
    if (ds_offset < 0) {
        pr_err("ds is NOT a kshm pointer (use kava_alloc to fix it)\n");
        return -1;
    }

    s64 filename_offset = kava_shm_offset(filename);
    if (filename_offset < 0) {
        pr_err("filename is NOT a kshm pointer (use kava_alloc to fix it)\n");
        return -1;
    }

    s64 delim_offset = kava_shm_offset(delim);
    if (delim_offset < 0) {
        pr_err("delim is NOT a kshm pointer (use kava_alloc to fix it)\n");
        return -1;
    }

    struct lake_cmd_libml_dataset_from_csv cmd = {
        .API_ID = LAKE_API_LIBML_dataset_from_csv,
        .ds = ds_offset,
        .filename = filename_offset,
        .delim = delim_offset,
        .n_cols = n_cols,
        .data_type = data_type,
        .headers = headers,
    };
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC, &ret);
	return (int) ret.device;
}
EXPORT_SYMBOL(dataset_from_csv);

struct norm_metadata *dataset_normalize(struct dataset *ds)
{
    struct lake_cmd_ret ret;

    s64 ds_offset = kava_shm_offset(ds);
    if (ds_offset < 0) {
        pr_err("ds is NOT a kshm pointer (use kava_alloc to fix it)\n");
        return -1;
    }

    // NOTE: We need to allocate it here because we have to copy in lakeD that
    // has no access to kava_alloc. This could be a problem if the called API
    // returns a `malloc`ed memory of runtime determined size and we have to
    // copy it back as we can't know how much we would have to allocate here.
    // Ideally we would like lakeD to reserve the space it needs and free us
    // from the check before returning.
    //
    // In truth, we could have _pointers_ to memory in user_space as long as the
    // memory accesses are resolved only in userspace.
    struct norm_metadata* aret = kava_alloc(sizeof(struct norm_metadata));

    struct lake_cmd_libml_dataset_normalize cmd = {
        .API_ID = LAKE_API_LIBML_dataset_normalize,
        .ds = ds_offset,
        .ret = kava_shm_offset(aret),
    };

    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC, &ret);

    // Rigmarole for checking for NULL return.
    if (ret.norm_metadata_ptr == NULL) {
        return NULL;
        kava_free(aret);
    } else {
        return aret;
    }
}
EXPORT_SYMBOL(dataset_normalize);

int init_layer(struct layer *l, int n_input, int n_output, enum act_func act) {
    struct lake_cmd_ret ret;

    s64 l_offset = kava_shm_offset(l);
    if (l_offset < 0) {
        pr_err("l is NOT a kshm pointer (use kava_alloc to fix it)\n");
        return -1;
    }

    struct lake_cmd_libml_init_layer cmd = {
        .API_ID = LAKE_API_LIBML_init_layer,
        .l = l_offset,
        .n_input = n_input,
        .n_output = n_output,
        .act = act,
    };

    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC, &ret);

    return ret.r_int;
}
EXPORT_SYMBOL(init_layer);
struct dataset dataset_slice(struct dataset *ds, int from_1, int to_1,
                             int from_2, int to_2)
{
    struct lake_cmd_ret ret;

    s64 ds_offset = kava_shm_offset(ds);
    if (ds_offset < 0) {
        pr_err("ds is NOT a kshm pointer (use kava_alloc to fix it)\n");
        struct dataset ret;
        return ret;
    }

    struct lake_cmd_libml_dataset_slice cmd = {
        .API_ID = LAKE_API_LIBML_init_layer,
        .ds = ds_offset,
        .from_1 = from_1,
        .to_1 = to_1,
        .from_2 = from_2,
        .to_2 = to_2,
    };

    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC, &ret);

    return ret.r_dataset;
}
EXPORT_SYMBOL(dataset_slice);

int init_matrix(struct matrix *m, int rows, int cols,
                int preset, struct matrix *existing)
{
    struct lake_cmd_ret ret;

    s64 m_offset = kava_shm_offset(m);
    if (m_offset < 0) {
        pr_err("m is NOT a kshm pointer (use kava_alloc to fix it)\n");
        return -1;
    }

    // XXX: May be bugged? Is zero a valid offset?
    s64 existing_offset = NULL;
    if (existing != NULL)
    {
        existing_offset = kava_shm_offset(existing);

        if (existing_offset < 0) {
            pr_err("existing is NOT a kshm pointer (use kava_alloc to fix it)\n");
            return -1;
        }
    }

    struct lake_cmd_libml_init_matrix cmd = {
        .API_ID = LAKE_API_LIBML_init_matrix,
        .m = m_offset,
        .rows = rows,
        .cols = cols,
        .preset = preset,
        .existing = existing_offset,
    };

    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC, &ret);

    return ret.r_int;
}
EXPORT_SYMBOL(init_matrix);

int init_model_2(struct model* m, int n_input, int n_output_hidden, int n_output, enum loss_func loss)
{
    struct lake_cmd_ret ret;

    s64 m_offset = kava_shm_offset(m);
    if (m_offset < 0) {
        pr_err("m is NOT a kshm pointer (use kava_alloc to fix it)\n");
        return -1;
    }

    struct lake_cmd_libml_init_model_2 cmd = {
        .API_ID = LAKE_API_LIBML_init_model_2,
        .m = m_offset,
        .n_input = n_input,
        .n_output_hidden = n_output_hidden,
        .n_output = n_output,
        .loss = loss,
    };

    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC, &ret);

    return ret.r_int;
}
EXPORT_SYMBOL(init_model_2);

void train(struct model *m, struct dataset *x, struct dataset *y, float lr, int epochs)
{
    struct lake_cmd_ret ret;

    s64 m_offset = kava_shm_offset(m);
    if (m_offset < 0) {
        pr_err("m is NOT a kshm pointer (use kava_alloc to fix it)\n");
        return;
    }

    s64 x_offset = kava_shm_offset(x);
    if (x_offset < 0) {
        pr_err("x is NOT a kshm pointer (use kava_alloc to fix it)\n");
        return;
    }

    s64 y_offset = kava_shm_offset(y);
    if (y_offset < 0) {
        pr_err("y is NOT a kshm pointer (use kava_alloc to fix it)\n");
        return;
    }

    struct lake_cmd_libml_train cmd = {
        .API_ID = LAKE_API_LIBML_train,
        .m = m_offset,
        .x = x_offset,
        .y = y_offset,
        .lr = lr,
        .epochs = epochs,
    };

    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC, &ret);
}
EXPORT_SYMBOL(train);

MlpiModel* mlpi_load(const char *path)
{
    struct lake_cmd_ret ret;

    s64 path_offset = kava_shm_offset(path);
    if (path_offset < 0) {
        pr_err("path is NOT a kshm pointer (use kava_alloc to fix it)\n");
        return;
    }

    struct lake_cmd_pytorch_mlpi_load cmd = {
        .API_ID = LAKE_API_PYTORCH_mlpi_load,
        .path = path_offset,
    };

    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC, &ret);

    return ret.v_ptr;
}
EXPORT_SYMBOL(mlpi_load);

int infer_on_row_floats(const MlpiModel *m,
                        const char *csv_path,
                        int target_row) {
  // NOTE: m is not kava alloced

  struct lake_cmd_ret ret;

  s64 csv_path_offset = kava_shm_offset(csv_path);
  if (csv_path_offset < 0) {
    pr_err("csv_path is NOT a kshm pointer (use kava_alloc to fix it)\n");
    return;
  }

  struct lake_cmd_pytorch_infer_on_row_floats cmd = {
      .API_ID = LAKE_API_PYTORCH_mlpi_load,
      .m = m,
      .csv_path = csv_path_offset,
      .target_row = target_row,
  };

  lake_send_cmd((void *)&cmd, sizeof(cmd), CMD_SYNC, &ret);

  return ret.r_int;
}
EXPORT_SYMBOL(infer_on_row_floats);

int pytorch_train(char* features_csv, char* labels_csv, char* hidden) {
  struct lake_cmd_ret ret;

  s64 features_csv_offset = kava_shm_offset(features_csv);
  if (features_csv_offset < 0) {
    pr_err("features_csv is NOT a kshm pointer (use kava_alloc to fix it)\n");
    return 0;
  }

  s64 labels_csv_offset = kava_shm_offset(labels_csv);
  if (labels_csv_offset < 0) {
    pr_err("labels_csv is NOT a kshm pointer (use kava_alloc to fix it)\n");
    return 0;
  }

  s64 hidden_offset = kava_shm_offset(hidden);
  if (hidden_offset < 0) {
    pr_err("hidden is NOT a kshm pointer (use kava_alloc to fix it)\n");
    return 0;
  }

  struct lake_cmd_pytorch_train cmd = {
      .API_ID = LAKE_API_pytorch_train,
      .features_csv = features_csv_offset,
      .labels_csv = labels_csv_offset,
      .hidden = hidden_offset,
  };

  lake_send_cmd((void *)&cmd, sizeof(cmd), CMD_SYNC, &ret);

  return ret.r_int;
}
EXPORT_SYMBOL(pytorch_train);

int infer_on_floats(const MlpiModel *m,
                    const float *x_f,
                    int *predicted_class_out) {
  // The MlpiModel is being kept in lakeD memory
  struct lake_cmd_ret ret;

  s64 x_f_offset = kava_shm_offset(x_f);
  if (x_f_offset < 0) {
    pr_err("x_f is NOT a kshm pointer (use kava_alloc to fix it)\n");
    return 0;
  }

  s64 predicted_class_out_offset = kava_shm_offset(predicted_class_out);
  if (predicted_class_out_offset < 0) {
    pr_err("predicted_class_out is NOT a kshm pointer (use kava_alloc to fix it)\n");
    return 0;
  }

  struct lake_cmd_infer_on_floats cmd = {
      .API_ID = LAKE_API_read_row_floats_and_quantize,
      .m = m,
      .x_f = x_f_offset,
      .predicted_class_out = predicted_class_out_offset,
  };

  lake_send_cmd((void *)&cmd, sizeof(cmd), CMD_SYNC, &ret);

  return ret.r_int;
}
EXPORT_SYMBOL(infer_on_floats);

int read_row_floats_and_quantize(const MlpiModel *m, const char *csv_path,
                                 int target_row, int8_t *x_q_out) {
  // The MlpiModel is being kept in lakeD memory
  struct lake_cmd_ret ret;

  s64 csv_path_offset = kava_shm_offset(csv_path);
  if (csv_path_offset < 0) {
    pr_err("csv_path is NOT a kshm pointer (use kava_alloc to fix it)\n");
    return 0;
  }

  s64 x_q_out_offset = kava_shm_offset(x_q_out);
  if (x_q_out_offset < 0) {
    pr_err("x_q_out is NOT a kshm pointer (use kava_alloc to fix it)\n");
    return 0;
  }

  struct lake_cmd_read_row_floats_and_quantize cmd = {
      .API_ID = LAKE_API_read_row_floats_and_quantize,
      .m = m,
      .csv_path = csv_path_offset,
      .target_row = target_row,
      .x_q_out = x_q_out_offset
  };

  lake_send_cmd((void *)&cmd, sizeof(cmd), CMD_SYNC, &ret);

  return ret.r_int;
}
EXPORT_SYMBOL(read_row_floats_and_quantize);

int mlpi_model_in_dim(const MlpiModel *m) {
  // The MlpiModel is being kept in lakeD memory
  struct lake_cmd_ret ret;

  struct lake_cmd_mlpi_model_in_dim cmd = {
      .API_ID = LAKE_API_mlpi_model_in_dim,
      .m = m,
  };

  lake_send_cmd((void *)&cmd, sizeof(cmd), CMD_SYNC, &ret);

  return ret.r_int;
}
EXPORT_SYMBOL(mlpi_model_in_dim);
