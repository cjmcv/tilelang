; ModuleID = 'TVMMod'
source_filename = "TVMMod"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

%0 = type { i32, i32, double }
%1 = type { ptr, %2, i32, %3, ptr, ptr, i64 }
%2 = type { i32, i32 }
%3 = type { i8, i8, i16 }

@__tvm_ffi__library_ctx = linkonce dllexport local_unnamed_addr global ptr null, align 8
@__TVMFFIFunctionCall = linkonce dllexport local_unnamed_addr global ptr null, align 8
@__TVMBackendGetFuncFromEnv = linkonce dllexport local_unnamed_addr global ptr null, align 8
@__TVMFFIErrorSetRaisedFromCStr = linkonce dllexport local_unnamed_addr global ptr null, align 8
@.str = private constant [55 x i8] c"Assert fail: num_args == 3, gemm: num_args should be 3\00", align 1
@.str.1 = private constant [13 x i8] c"RuntimeError\00", align 1
@.str.2 = private constant [63 x i8] c"Assert fail: not T.isnullptr(args), gemm: args pointer is NULL\00", align 1
@.str.3 = private constant [178 x i8] c"Assert fail: A_handle_type_index == 0 or A_handle_type_index == 4 or A_handle_type_index == 7 or 64 <= A_handle_type_index, kernel gemm input A expected pointer or tensor handle\00", align 1
@.str.4 = private constant [178 x i8] c"Assert fail: B_handle_type_index == 0 or B_handle_type_index == 4 or B_handle_type_index == 7 or 64 <= B_handle_type_index, kernel gemm input B expected pointer or tensor handle\00", align 1
@.str.5 = private constant [178 x i8] c"Assert fail: C_handle_type_index == 0 or C_handle_type_index == 4 or C_handle_type_index == 7 or 64 <= C_handle_type_index, kernel gemm input C expected pointer or tensor handle\00", align 1
@.str.6 = private constant [77 x i8] c"Assert fail: not gemm_A_is_null, gemm.A is expected to have non-NULL pointer\00", align 1
@.str.7 = private constant [77 x i8] c"Assert fail: not gemm_B_is_null, gemm.B is expected to have non-NULL pointer\00", align 1
@.str.8 = private constant [77 x i8] c"Assert fail: not gemm_C_is_null, gemm.C is expected to have non-NULL pointer\00", align 1
@.str.9 = private constant [5 x i8] c"gemm\00", align 1
@.str.10 = private constant [2 x i8] c"A\00", align 1
@.tvm_func.__tvm_error_ndim_mismatch = internal unnamed_addr global ptr null, align 8
@.str.11 = private constant [26 x i8] c"__tvm_error_ndim_mismatch\00", align 1
@.str.12 = private constant [2 x i8] c"B\00", align 1
@.str.13 = private constant [2 x i8] c"C\00", align 1
@.tvm_func.__tvm_error_dtype_mismatch = internal unnamed_addr global ptr null, align 8
@.str.14 = private constant [27 x i8] c"__tvm_error_dtype_mismatch\00", align 1
@.str.15 = private constant [9 x i8] c"shape[0]\00", align 1
@.tvm_func.__tvm_error_expect_eq = internal unnamed_addr global ptr null, align 8
@.str.16 = private constant [22 x i8] c"__tvm_error_expect_eq\00", align 1
@.str.17 = private constant [9 x i8] c"shape[1]\00", align 1
@.str.18 = private constant [11 x i8] c"strides[1]\00", align 1
@.str.19 = private constant [11 x i8] c"strides[0]\00", align 1
@.tvm_func.__tvm_error_byte_offset_mismatch = internal unnamed_addr global ptr null, align 8
@.str.20 = private constant [33 x i8] c"__tvm_error_byte_offset_mismatch\00", align 1
@.tvm_func.__tvm_error_device_type_mismatch = internal unnamed_addr global ptr null, align 8
@.str.21 = private constant [33 x i8] c"__tvm_error_device_type_mismatch\00", align 1
@.str.22 = private constant [13 x i8] c"data pointer\00", align 1
@.tvm_func.__tvm_error_null_ptr = internal unnamed_addr global ptr null, align 8
@.str.23 = private constant [21 x i8] c"__tvm_error_null_ptr\00", align 1
@.str.24 = private constant [10 x i8] c"device_id\00", align 1
@.tvm_func.__tvm_set_device = internal unnamed_addr global ptr null, align 8
@.str.25 = private constant [17 x i8] c"__tvm_set_device\00", align 1
@.tvm_func.gemm_kernel = internal unnamed_addr global ptr null, align 8
@.str.26 = private constant [12 x i8] c"gemm_kernel\00", align 1
@llvm.global_ctors = appending global [0 x { i32, ptr, ptr }] zeroinitializer

define dllexport i32 @gemm(ptr nocapture readnone %self_handle, ptr readonly %args, i32 %num_args, ptr nocapture readnone %result) local_unnamed_addr #0 !dbg !5 {
entry:
  %0 = alloca ptr, align 8, !dbg !15
  call void @llvm.dbg.value(metadata ptr poison, metadata !11, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.value(metadata ptr %args, metadata !12, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.value(metadata i32 %num_args, metadata !13, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.value(metadata ptr poison, metadata !14, metadata !DIExpression()), !dbg !27
  %1 = alloca ptr, align 8, !dbg !27
  %2 = alloca ptr, align 8, !dbg !27
  %3 = alloca ptr, align 8, !dbg !27
  %4 = alloca ptr, align 8, !dbg !27
  %5 = alloca ptr, align 8, !dbg !27
  %6 = alloca ptr, align 8, !dbg !27
  %7 = alloca ptr, align 8, !dbg !27
  %8 = alloca ptr, align 8, !dbg !27
  %9 = alloca ptr, align 8, !dbg !27
  %10 = alloca ptr, align 8, !dbg !27
  %11 = alloca ptr, align 8, !dbg !27
  %12 = alloca ptr, align 8, !dbg !27
  %13 = alloca ptr, align 8, !dbg !27
  %14 = alloca ptr, align 8, !dbg !27
  %15 = alloca ptr, align 8, !dbg !27
  %16 = alloca ptr, align 8, !dbg !27
  %17 = alloca ptr, align 8, !dbg !27
  %18 = alloca ptr, align 8, !dbg !27
  %19 = alloca ptr, align 8, !dbg !27
  %20 = alloca ptr, align 8, !dbg !27
  %21 = alloca ptr, align 8, !dbg !27
  %22 = alloca ptr, align 8, !dbg !27
  %23 = alloca ptr, align 8, !dbg !27
  %24 = alloca ptr, align 8, !dbg !27
  %25 = alloca ptr, align 8, !dbg !27
  %26 = alloca ptr, align 8, !dbg !27
  %27 = alloca ptr, align 8, !dbg !27
  %28 = alloca ptr, align 8, !dbg !27
  %29 = alloca ptr, align 8, !dbg !27
  %30 = alloca ptr, align 8, !dbg !27
  %stack_ffi_any375 = alloca [10 x %0], align 8, !dbg !27
  call void @llvm.dbg.declare(metadata ptr %stack_ffi_any375, metadata !28, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.declare(metadata ptr %stack_ffi_any375, metadata !28, metadata !DIExpression()), !dbg !27
  %31 = icmp eq i32 %num_args, 3, !dbg !27
  br i1 %31, label %assert_end, label %assert_fail, !dbg !27, !prof !29

common.ret:                                       ; preds = %gemm_compute_.exit, %handle_init_end277, %handle_init276, %handle_init_end271, %handle_init270, %handle_init_end263, %handle_init262, %handle_init_end255, %handle_init254, %handle_init_end247, %handle_init246, %handle_init_end239, %handle_init238, %handle_init_end225, %handle_init224, %handle_init_end211, %handle_init210, %handle_init_end203, %handle_init202, %handle_init_end195, %handle_init194, %handle_init_end187, %handle_init186, %handle_init_end179, %handle_init178, %handle_init_end171, %handle_init170, %handle_init_end163, %handle_init162, %handle_init_end155, %handle_init154, %handle_init_end141, %handle_init140, %handle_init_end127, %handle_init126, %handle_init_end119, %handle_init118, %handle_init_end111, %handle_init110, %handle_init_end103, %handle_init102, %handle_init_end95, %handle_init94, %handle_init_end87, %handle_init86, %handle_init_end79, %handle_init78, %handle_init_end65, %handle_init64, %handle_init_end52, %handle_init51, %handle_init_end44, %handle_init43, %handle_init_end36, %handle_init35, %handle_init_end28, %handle_init27, %handle_init_end20, %handle_init19, %handle_init_end, %handle_init, %assert_fail13, %assert_fail11, %assert_fail9, %assert_fail7, %assert_fail5, %assert_fail3, %assert_fail1, %assert_fail
  %common.ret.op = phi i32 [ -1, %assert_fail ], [ -1, %assert_fail1 ], [ -1, %assert_fail3 ], [ -1, %assert_fail5 ], [ -1, %assert_fail7 ], [ -1, %assert_fail9 ], [ -1, %assert_fail11 ], [ -1, %assert_fail13 ], [ %77, %handle_init ], [ %80, %handle_init_end ], [ %100, %handle_init19 ], [ %103, %handle_init_end20 ], [ %132, %handle_init27 ], [ %135, %handle_init_end28 ], [ %166, %handle_init35 ], [ %169, %handle_init_end36 ], [ %191, %handle_init43 ], [ %194, %handle_init_end44 ], [ %214, %handle_init51 ], [ %217, %handle_init_end52 ], [ %239, %handle_init64 ], [ %242, %handle_init_end65 ], [ %266, %handle_init78 ], [ %269, %handle_init_end79 ], [ %287, %handle_init86 ], [ %290, %handle_init_end87 ], [ %308, %handle_init94 ], [ %311, %handle_init_end95 ], [ %335, %handle_init102 ], [ %338, %handle_init_end103 ], [ %369, %handle_init110 ], [ %372, %handle_init_end111 ], [ %394, %handle_init118 ], [ %397, %handle_init_end119 ], [ %417, %handle_init126 ], [ %420, %handle_init_end127 ], [ %442, %handle_init140 ], [ %445, %handle_init_end141 ], [ %469, %handle_init154 ], [ %472, %handle_init_end155 ], [ %491, %handle_init162 ], [ %494, %handle_init_end163 ], [ %517, %handle_init170 ], [ %520, %handle_init_end171 ], [ %538, %handle_init178 ], [ %541, %handle_init_end179 ], [ %565, %handle_init186 ], [ %568, %handle_init_end187 ], [ %599, %handle_init194 ], [ %602, %handle_init_end195 ], [ %624, %handle_init202 ], [ %627, %handle_init_end203 ], [ %647, %handle_init210 ], [ %650, %handle_init_end211 ], [ %672, %handle_init224 ], [ %675, %handle_init_end225 ], [ %699, %handle_init238 ], [ %702, %handle_init_end239 ], [ %721, %handle_init246 ], [ %724, %handle_init_end247 ], [ %747, %handle_init254 ], [ %750, %handle_init_end255 ], [ %768, %handle_init262 ], [ %771, %handle_init_end263 ], [ %795, %handle_init270 ], [ %798, %handle_init_end271 ], [ %803, %handle_init276 ], [ %806, %handle_init_end277 ], [ %common.ret.op.i, %gemm_compute_.exit ]
  ret i32 %common.ret.op, !dbg !27

assert_fail:                                      ; preds = %entry
  %32 = load ptr, ptr @__TVMFFIErrorSetRaisedFromCStr, align 8, !dbg !27, !tbaa !30
  tail call void %32(ptr nonnull @.str.1, ptr nonnull @.str), !dbg !27
  br label %common.ret

assert_end:                                       ; preds = %entry
  %.not = icmp eq ptr %args, null, !dbg !27
  br i1 %.not, label %assert_fail1, label %assert_end2, !dbg !27, !prof !33

assert_fail1:                                     ; preds = %assert_end
  %33 = load ptr, ptr @__TVMFFIErrorSetRaisedFromCStr, align 8, !dbg !27, !tbaa !30
  tail call void %33(ptr nonnull @.str.1, ptr nonnull @.str.2), !dbg !27
  br label %common.ret

assert_end2:                                      ; preds = %assert_end
  %A_handle.type_index = load i32, ptr %args, align 4, !dbg !27
  %A_handle.type_index.fr = freeze i32 %A_handle.type_index, !dbg !27
  call void @llvm.dbg.declare(metadata i32 %A_handle.type_index, metadata !34, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.declare(metadata i32 %A_handle.type_index, metadata !34, metadata !DIExpression()), !dbg !27
  %34 = icmp sgt i32 %A_handle.type_index.fr, 63, !dbg !27
  br i1 %34, label %assert_end4, label %switch.early.test, !dbg !27

switch.early.test:                                ; preds = %assert_end2
  switch i32 %A_handle.type_index.fr, label %assert_fail3 [
    i32 7, label %assert_end4
    i32 4, label %assert_end4
    i32 0, label %assert_end4
  ], !dbg !27

assert_fail3:                                     ; preds = %switch.early.test
  %35 = load ptr, ptr @__TVMFFIErrorSetRaisedFromCStr, align 8, !dbg !27, !tbaa !30
  tail call void %35(ptr nonnull @.str.1, ptr nonnull @.str.3), !dbg !27
  br label %common.ret

assert_end4:                                      ; preds = %switch.early.test, %switch.early.test, %switch.early.test, %assert_end2
  %36 = getelementptr inbounds %0, ptr %args, i64 1, i32 0, !dbg !27
  %B_handle.type_index = load i32, ptr %36, align 4, !dbg !27
  %B_handle.type_index.fr = freeze i32 %B_handle.type_index, !dbg !27
  call void @llvm.dbg.declare(metadata i32 %B_handle.type_index, metadata !35, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.declare(metadata i32 %B_handle.type_index, metadata !35, metadata !DIExpression()), !dbg !27
  %37 = icmp sgt i32 %B_handle.type_index.fr, 63, !dbg !27
  br i1 %37, label %assert_end6, label %switch.early.test284, !dbg !27

switch.early.test284:                             ; preds = %assert_end4
  switch i32 %B_handle.type_index.fr, label %assert_fail5 [
    i32 7, label %assert_end6
    i32 4, label %assert_end6
    i32 0, label %assert_end6
  ], !dbg !27

assert_fail5:                                     ; preds = %switch.early.test284
  %38 = load ptr, ptr @__TVMFFIErrorSetRaisedFromCStr, align 8, !dbg !27, !tbaa !30
  tail call void %38(ptr nonnull @.str.1, ptr nonnull @.str.4), !dbg !27
  br label %common.ret

assert_end6:                                      ; preds = %switch.early.test284, %switch.early.test284, %switch.early.test284, %assert_end4
  %39 = getelementptr inbounds %0, ptr %args, i64 2, i32 0, !dbg !27
  %C_handle.type_index = load i32, ptr %39, align 4, !dbg !27
  %C_handle.type_index.fr = freeze i32 %C_handle.type_index, !dbg !27
  call void @llvm.dbg.declare(metadata i32 %C_handle.type_index, metadata !36, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.declare(metadata i32 %C_handle.type_index, metadata !36, metadata !DIExpression()), !dbg !27
  %40 = icmp sgt i32 %C_handle.type_index.fr, 63, !dbg !27
  br i1 %40, label %assert_end8, label %switch.early.test285, !dbg !27

switch.early.test285:                             ; preds = %assert_end6
  switch i32 %C_handle.type_index.fr, label %assert_fail7 [
    i32 7, label %assert_end8
    i32 4, label %assert_end8
    i32 0, label %assert_end8
  ], !dbg !27

assert_fail7:                                     ; preds = %switch.early.test285
  %41 = load ptr, ptr @__TVMFFIErrorSetRaisedFromCStr, align 8, !dbg !27, !tbaa !30
  tail call void %41(ptr nonnull @.str.1, ptr nonnull @.str.5), !dbg !27
  br label %common.ret

assert_end8:                                      ; preds = %switch.early.test285, %switch.early.test285, %switch.early.test285, %assert_end6
  %42 = getelementptr inbounds %0, ptr %args, i64 0, i32 2, !dbg !27
  %43 = load ptr, ptr %42, align 8, !dbg !27
  %44 = icmp eq i32 %A_handle.type_index.fr, 70, !dbg !27
  %A_handle.idx = select i1 %44, i64 24, i64 0, !dbg !27
  %A_handle = getelementptr i8, ptr %43, i64 %A_handle.idx, !dbg !27
  call void @llvm.dbg.declare(metadata ptr %A_handle, metadata !37, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.declare(metadata ptr %A_handle, metadata !37, metadata !DIExpression()), !dbg !27
  %45 = getelementptr inbounds %0, ptr %args, i64 1, i32 2, !dbg !27
  %46 = load ptr, ptr %45, align 8, !dbg !27
  %47 = icmp eq i32 %B_handle.type_index.fr, 70, !dbg !27
  %B_handle.idx = select i1 %47, i64 24, i64 0, !dbg !27
  %B_handle = getelementptr i8, ptr %46, i64 %B_handle.idx, !dbg !27
  call void @llvm.dbg.declare(metadata ptr %B_handle, metadata !38, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.declare(metadata ptr %B_handle, metadata !38, metadata !DIExpression()), !dbg !27
  %48 = getelementptr inbounds %0, ptr %args, i64 2, i32 2, !dbg !27
  %49 = load ptr, ptr %48, align 8, !dbg !27
  %50 = icmp eq i32 %C_handle.type_index.fr, 70, !dbg !27
  %C_handle.idx = select i1 %50, i64 24, i64 0, !dbg !27
  %C_handle = getelementptr i8, ptr %49, i64 %C_handle.idx, !dbg !27
  call void @llvm.dbg.declare(metadata ptr %C_handle, metadata !39, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.declare(metadata ptr %C_handle, metadata !39, metadata !DIExpression()), !dbg !27
  %gemm.A_is_null.not = icmp eq ptr %A_handle, null, !dbg !27
  call void @llvm.dbg.declare(metadata i1 %gemm.A_is_null.not, metadata !40, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.declare(metadata i1 %gemm.A_is_null.not, metadata !40, metadata !DIExpression()), !dbg !27
  br i1 %gemm.A_is_null.not, label %assert_fail9, label %assert_end10, !dbg !27, !prof !33

assert_fail9:                                     ; preds = %assert_end8
  %51 = load ptr, ptr @__TVMFFIErrorSetRaisedFromCStr, align 8, !dbg !27, !tbaa !30
  tail call void %51(ptr nonnull @.str.1, ptr nonnull @.str.6), !dbg !27
  br label %common.ret

assert_end10:                                     ; preds = %assert_end8
  %gemm.B_is_null.not = icmp eq ptr %B_handle, null, !dbg !27
  call void @llvm.dbg.declare(metadata i1 %gemm.B_is_null.not, metadata !42, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.declare(metadata i1 %gemm.B_is_null.not, metadata !42, metadata !DIExpression()), !dbg !27
  br i1 %gemm.B_is_null.not, label %assert_fail11, label %assert_end12, !dbg !27, !prof !33

assert_fail11:                                    ; preds = %assert_end10
  %52 = load ptr, ptr @__TVMFFIErrorSetRaisedFromCStr, align 8, !dbg !27, !tbaa !30
  tail call void %52(ptr nonnull @.str.1, ptr nonnull @.str.7), !dbg !27
  br label %common.ret

assert_end12:                                     ; preds = %assert_end10
  %gemm.C_is_null.not = icmp eq ptr %C_handle, null, !dbg !27
  call void @llvm.dbg.declare(metadata i1 %gemm.C_is_null.not, metadata !43, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.declare(metadata i1 %gemm.C_is_null.not, metadata !43, metadata !DIExpression()), !dbg !27
  br i1 %gemm.C_is_null.not, label %assert_fail13, label %assert_end14, !dbg !27, !prof !33

assert_fail13:                                    ; preds = %assert_end12
  %53 = load ptr, ptr @__TVMFFIErrorSetRaisedFromCStr, align 8, !dbg !27, !tbaa !30
  tail call void %53(ptr nonnull @.str.1, ptr nonnull @.str.8), !dbg !27
  br label %common.ret

assert_end14:                                     ; preds = %assert_end12
  %54 = getelementptr inbounds %1, ptr %A_handle, i64 0, i32 4, !dbg !27
  %gemm.A.shape = load ptr, ptr %54, align 8, !dbg !27
  call void @llvm.dbg.declare(metadata ptr %gemm.A.shape, metadata !44, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.declare(metadata ptr %gemm.A.shape, metadata !44, metadata !DIExpression()), !dbg !27
  %55 = getelementptr inbounds %1, ptr %B_handle, i64 0, i32 4, !dbg !27
  %gemm.B.shape = load ptr, ptr %55, align 8, !dbg !27
  call void @llvm.dbg.declare(metadata ptr %gemm.B.shape, metadata !47, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.declare(metadata ptr %gemm.B.shape, metadata !47, metadata !DIExpression()), !dbg !27
  %56 = getelementptr inbounds %1, ptr %C_handle, i64 0, i32 4, !dbg !27
  %gemm.C.shape = load ptr, ptr %56, align 8, !dbg !27
  call void @llvm.dbg.declare(metadata ptr %gemm.C.shape, metadata !48, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.declare(metadata ptr %gemm.C.shape, metadata !48, metadata !DIExpression()), !dbg !27
  %57 = getelementptr inbounds %1, ptr %A_handle, i64 0, i32 2, !dbg !27
  %58 = load i32, ptr %57, align 4, !dbg !27
  %.not376 = icmp eq i32 %58, 2, !dbg !27
  br i1 %.not376, label %if_end, label %if_then, !dbg !27, !prof !33

if_then:                                          ; preds = %assert_end14
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any375, align 8, !dbg !27
  %59 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 0, i32 2, !dbg !27
  store ptr @.str.9, ptr %59, align 8, !dbg !27
  %60 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %60, align 8, !dbg !27
  %61 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 2, !dbg !27
  store ptr @.str.10, ptr %61, align 8, !dbg !27
  %62 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %62, align 8, !dbg !27
  %63 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 2, !dbg !27
  store i64 2, ptr %63, align 8, !dbg !27
  %64 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %64, align 8, !dbg !27
  %65 = load i32, ptr %57, align 4, !dbg !27
  %66 = sext i32 %65 to i64, !dbg !27
  %67 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 2, !dbg !27
  store i64 %66, ptr %67, align 8, !dbg !27
  %68 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 0, !dbg !27
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %68, i8 0, i64 16, i1 false), !dbg !27
  %69 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !27, !tbaa !30
  %70 = load ptr, ptr @.tvm_func.__tvm_error_ndim_mismatch, align 8, !dbg !27
  %.not439 = icmp eq ptr %70, null, !dbg !27
  br i1 %.not439, label %handle_init, label %handle_init_end, !dbg !27, !prof !33

if_end:                                           ; preds = %handle_init_end, %assert_end14
  %71 = getelementptr inbounds %1, ptr %A_handle, i64 0, i32 5, !dbg !27
  %gemm.A.strides = load ptr, ptr %71, align 8, !dbg !27
  call void @llvm.dbg.declare(metadata ptr %gemm.A.strides, metadata !49, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.declare(metadata ptr %gemm.A.strides, metadata !49, metadata !DIExpression()), !dbg !27
  %72 = getelementptr inbounds %1, ptr %A_handle, i64 0, i32 1, i32 1, !dbg !27
  %dev_id = load i32, ptr %72, align 4, !dbg !27
  call void @llvm.dbg.declare(metadata i32 %dev_id, metadata !50, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.declare(metadata i32 %dev_id, metadata !50, metadata !DIExpression()), !dbg !27
  %A = load ptr, ptr %A_handle, align 8, !dbg !27
  call void @llvm.dbg.declare(metadata ptr %A, metadata !51, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.declare(metadata ptr %A, metadata !51, metadata !DIExpression()), !dbg !27
  call void @llvm.assume(i1 true) [ "align"(ptr %A, i64 64) ], !dbg !27
  %73 = getelementptr inbounds %1, ptr %B_handle, i64 0, i32 2, !dbg !27
  %74 = load i32, ptr %73, align 4, !dbg !27
  %.not377 = icmp eq i32 %74, 2, !dbg !27
  br i1 %.not377, label %if_end18, label %if_then17, !dbg !27, !prof !33

handle_init:                                      ; preds = %if_then
  %75 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !27, !tbaa !30
  %76 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !27, !tbaa !30
  %77 = call i32 %76(ptr %75, ptr nonnull @.str.11, ptr nonnull %30), !dbg !27
  %78 = icmp eq i32 %77, 0, !dbg !27
  br i1 %78, label %call_end, label %common.ret, !dbg !27, !prof !29

handle_init_end:                                  ; preds = %call_end, %if_then
  %79 = phi ptr [ %70, %if_then ], [ %82, %call_end ], !dbg !27
  %80 = call i32 %69(ptr %79, ptr nonnull %stack_ffi_any375, i32 4, ptr nonnull %68), !dbg !27
  %81 = icmp eq i32 %80, 0, !dbg !27
  br i1 %81, label %if_end, label %common.ret, !dbg !27, !prof !29

call_end:                                         ; preds = %handle_init
  %82 = load ptr, ptr %30, align 8, !dbg !27
  store ptr %82, ptr @.tvm_func.__tvm_error_ndim_mismatch, align 8, !dbg !27
  br label %handle_init_end, !dbg !27

if_then17:                                        ; preds = %if_end
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any375, align 8, !dbg !27
  %83 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 0, i32 2, !dbg !27
  store ptr @.str.9, ptr %83, align 8, !dbg !27
  %84 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %84, align 8, !dbg !27
  %85 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 2, !dbg !27
  store ptr @.str.12, ptr %85, align 8, !dbg !27
  %86 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %86, align 8, !dbg !27
  %87 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 2, !dbg !27
  store i64 2, ptr %87, align 8, !dbg !27
  %88 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %88, align 8, !dbg !27
  %89 = load i32, ptr %73, align 4, !dbg !27
  %90 = sext i32 %89 to i64, !dbg !27
  %91 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 2, !dbg !27
  store i64 %90, ptr %91, align 8, !dbg !27
  %92 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 0, !dbg !27
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %92, i8 0, i64 16, i1 false), !dbg !27
  %93 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !27, !tbaa !30
  %94 = load ptr, ptr @.tvm_func.__tvm_error_ndim_mismatch, align 8, !dbg !27
  %.not438 = icmp eq ptr %94, null, !dbg !27
  br i1 %.not438, label %handle_init19, label %handle_init_end20, !dbg !27, !prof !33

if_end18:                                         ; preds = %handle_init_end20, %if_end
  %95 = getelementptr inbounds %1, ptr %B_handle, i64 0, i32 5, !dbg !27
  %gemm.B.strides = load ptr, ptr %95, align 8, !dbg !27
  call void @llvm.dbg.declare(metadata ptr %gemm.B.strides, metadata !52, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.declare(metadata ptr %gemm.B.strides, metadata !52, metadata !DIExpression()), !dbg !27
  %B = load ptr, ptr %B_handle, align 8, !dbg !27
  call void @llvm.dbg.declare(metadata ptr %B, metadata !53, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.declare(metadata ptr %B, metadata !53, metadata !DIExpression()), !dbg !27
  call void @llvm.assume(i1 true) [ "align"(ptr %B, i64 64) ], !dbg !27
  %96 = getelementptr inbounds %1, ptr %C_handle, i64 0, i32 2, !dbg !27
  %97 = load i32, ptr %96, align 4, !dbg !27
  %.not378 = icmp eq i32 %97, 2, !dbg !27
  br i1 %.not378, label %if_end26, label %if_then25, !dbg !27, !prof !33

handle_init19:                                    ; preds = %if_then17
  %98 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !27, !tbaa !30
  %99 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !27, !tbaa !30
  %100 = call i32 %99(ptr %98, ptr nonnull @.str.11, ptr nonnull %29), !dbg !27
  %101 = icmp eq i32 %100, 0, !dbg !27
  br i1 %101, label %call_end22, label %common.ret, !dbg !27, !prof !29

handle_init_end20:                                ; preds = %call_end22, %if_then17
  %102 = phi ptr [ %94, %if_then17 ], [ %105, %call_end22 ], !dbg !27
  %103 = call i32 %93(ptr %102, ptr nonnull %stack_ffi_any375, i32 4, ptr nonnull %92), !dbg !27
  %104 = icmp eq i32 %103, 0, !dbg !27
  br i1 %104, label %if_end18, label %common.ret, !dbg !27, !prof !29

call_end22:                                       ; preds = %handle_init19
  %105 = load ptr, ptr %29, align 8, !dbg !27
  store ptr %105, ptr @.tvm_func.__tvm_error_ndim_mismatch, align 8, !dbg !27
  br label %handle_init_end20, !dbg !27

if_then25:                                        ; preds = %if_end18
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any375, align 8, !dbg !27
  %106 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 0, i32 2, !dbg !27
  store ptr @.str.9, ptr %106, align 8, !dbg !27
  %107 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %107, align 8, !dbg !27
  %108 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 2, !dbg !27
  store ptr @.str.13, ptr %108, align 8, !dbg !27
  %109 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %109, align 8, !dbg !27
  %110 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 2, !dbg !27
  store i64 2, ptr %110, align 8, !dbg !27
  %111 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %111, align 8, !dbg !27
  %112 = load i32, ptr %96, align 4, !dbg !27
  %113 = sext i32 %112 to i64, !dbg !27
  %114 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 2, !dbg !27
  store i64 %113, ptr %114, align 8, !dbg !27
  %115 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 0, !dbg !27
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %115, i8 0, i64 16, i1 false), !dbg !27
  %116 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !27, !tbaa !30
  %117 = load ptr, ptr @.tvm_func.__tvm_error_ndim_mismatch, align 8, !dbg !27
  %.not437 = icmp eq ptr %117, null, !dbg !27
  br i1 %.not437, label %handle_init27, label %handle_init_end28, !dbg !27, !prof !33

if_end26:                                         ; preds = %handle_init_end28, %if_end18
  %118 = getelementptr inbounds %1, ptr %C_handle, i64 0, i32 5, !dbg !27
  %gemm.C.strides = load ptr, ptr %118, align 8, !dbg !27
  call void @llvm.dbg.declare(metadata ptr %gemm.C.strides, metadata !54, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.declare(metadata ptr %gemm.C.strides, metadata !54, metadata !DIExpression()), !dbg !27
  %C = load ptr, ptr %C_handle, align 8, !dbg !27
  call void @llvm.dbg.declare(metadata ptr %C, metadata !55, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.declare(metadata ptr %C, metadata !55, metadata !DIExpression()), !dbg !27
  call void @llvm.assume(i1 true) [ "align"(ptr %C, i64 64) ], !dbg !27
  %119 = getelementptr inbounds %1, ptr %A_handle, i64 0, i32 3, i32 2, !dbg !27
  %120 = load i16, ptr %119, align 2, !dbg !27
  %121 = icmp ne i16 %120, 1, !dbg !27
  %122 = getelementptr inbounds %1, ptr %A_handle, i64 0, i32 3, i32 1, !dbg !27
  %123 = load i8, ptr %122, align 1, !dbg !27
  %124 = icmp ne i8 %123, 16, !dbg !27
  %125 = getelementptr inbounds %1, ptr %A_handle, i64 0, i32 3, i32 0, !dbg !27
  %126 = load i8, ptr %125, align 1, !dbg !27
  %127 = icmp ne i8 %126, 4, !dbg !27
  %128 = or i1 %124, %127, !dbg !27
  %129 = or i1 %121, %128, !dbg !27
  br i1 %129, label %if_then33, label %if_end34, !dbg !27, !prof !29

handle_init27:                                    ; preds = %if_then25
  %130 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !27, !tbaa !30
  %131 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !27, !tbaa !30
  %132 = call i32 %131(ptr %130, ptr nonnull @.str.11, ptr nonnull %28), !dbg !27
  %133 = icmp eq i32 %132, 0, !dbg !27
  br i1 %133, label %call_end30, label %common.ret, !dbg !27, !prof !29

handle_init_end28:                                ; preds = %call_end30, %if_then25
  %134 = phi ptr [ %117, %if_then25 ], [ %137, %call_end30 ], !dbg !27
  %135 = call i32 %116(ptr %134, ptr nonnull %stack_ffi_any375, i32 4, ptr nonnull %115), !dbg !27
  %136 = icmp eq i32 %135, 0, !dbg !27
  br i1 %136, label %if_end26, label %common.ret, !dbg !27, !prof !29

call_end30:                                       ; preds = %handle_init27
  %137 = load ptr, ptr %28, align 8, !dbg !27
  store ptr %137, ptr @.tvm_func.__tvm_error_ndim_mismatch, align 8, !dbg !27
  br label %handle_init_end28, !dbg !27

if_then33:                                        ; preds = %if_end26
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any375, align 8, !dbg !27
  %138 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 0, i32 2, !dbg !27
  store ptr @.str.9, ptr %138, align 8, !dbg !27
  %139 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %139, align 8, !dbg !27
  %140 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 2, !dbg !27
  store ptr @.str.10, ptr %140, align 8, !dbg !27
  %141 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %141, align 8, !dbg !27
  %142 = load i8, ptr %125, align 1, !dbg !27
  %143 = zext i8 %142 to i64, !dbg !27
  %144 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 2, !dbg !27
  store i64 %143, ptr %144, align 8, !dbg !27
  %145 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %145, align 8, !dbg !27
  %146 = load i8, ptr %122, align 1, !dbg !27
  %147 = zext i8 %146 to i64, !dbg !27
  %148 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 2, !dbg !27
  store i64 %147, ptr %148, align 8, !dbg !27
  %149 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %149, align 8, !dbg !27
  %150 = load i16, ptr %119, align 2, !dbg !27
  %151 = zext i16 %150 to i64, !dbg !27
  %152 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 2, !dbg !27
  store i64 %151, ptr %152, align 8, !dbg !27
  %153 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 5, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %153, align 8, !dbg !27
  %154 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 5, i32 2, !dbg !27
  store i64 4, ptr %154, align 8, !dbg !27
  %155 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 6, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %155, align 8, !dbg !27
  %156 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 6, i32 2, !dbg !27
  store i64 16, ptr %156, align 8, !dbg !27
  %157 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 7, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %157, align 8, !dbg !27
  %158 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 7, i32 2, !dbg !27
  store i64 1, ptr %158, align 8, !dbg !27
  %159 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 8, i32 0, !dbg !27
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %159, i8 0, i64 16, i1 false), !dbg !27
  %160 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !27, !tbaa !30
  %161 = load ptr, ptr @.tvm_func.__tvm_error_dtype_mismatch, align 8, !dbg !27
  %.not436 = icmp eq ptr %161, null, !dbg !27
  br i1 %.not436, label %handle_init35, label %handle_init_end36, !dbg !27, !prof !33

if_end34:                                         ; preds = %handle_init_end36, %if_end26
  %162 = load i64, ptr %gemm.A.shape, align 8, !dbg !27, !tbaa !56
  %163 = and i64 %162, 4294967295, !dbg !27
  %.not379 = icmp eq i64 %163, 1024, !dbg !27
  br i1 %.not379, label %if_end42, label %if_then41, !dbg !27, !prof !33

handle_init35:                                    ; preds = %if_then33
  %164 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !27, !tbaa !30
  %165 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !27, !tbaa !30
  %166 = call i32 %165(ptr %164, ptr nonnull @.str.14, ptr nonnull %27), !dbg !27
  %167 = icmp eq i32 %166, 0, !dbg !27
  br i1 %167, label %call_end38, label %common.ret, !dbg !27, !prof !29

handle_init_end36:                                ; preds = %call_end38, %if_then33
  %168 = phi ptr [ %161, %if_then33 ], [ %171, %call_end38 ], !dbg !27
  %169 = call i32 %160(ptr %168, ptr nonnull %stack_ffi_any375, i32 8, ptr nonnull %159), !dbg !27
  %170 = icmp eq i32 %169, 0, !dbg !27
  br i1 %170, label %if_end34, label %common.ret, !dbg !27, !prof !29

call_end38:                                       ; preds = %handle_init35
  %171 = load ptr, ptr %27, align 8, !dbg !27
  store ptr %171, ptr @.tvm_func.__tvm_error_dtype_mismatch, align 8, !dbg !27
  br label %handle_init_end36, !dbg !27

if_then41:                                        ; preds = %if_end34
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any375, align 8, !dbg !27
  %172 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 0, i32 2, !dbg !27
  store ptr @.str.9, ptr %172, align 8, !dbg !27
  %173 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %173, align 8, !dbg !27
  %174 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 2, !dbg !27
  store ptr @.str.10, ptr %174, align 8, !dbg !27
  %175 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %175, align 8, !dbg !27
  %176 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 2, !dbg !27
  store ptr @.str.15, ptr %176, align 8, !dbg !27
  %177 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %177, align 8, !dbg !27
  %178 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 2, !dbg !27
  store i64 1024, ptr %178, align 8, !dbg !27
  %179 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %179, align 8, !dbg !27
  %180 = load i64, ptr %gemm.A.shape, align 8, !dbg !27, !tbaa !56
  %sext434 = shl i64 %180, 32, !dbg !27
  %181 = ashr exact i64 %sext434, 32, !dbg !27
  %182 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 2, !dbg !27
  store i64 %181, ptr %182, align 8, !dbg !27
  %183 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 5, i32 0, !dbg !27
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %183, i8 0, i64 16, i1 false), !dbg !27
  %184 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !27, !tbaa !30
  %185 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !27
  %.not435 = icmp eq ptr %185, null, !dbg !27
  br i1 %.not435, label %handle_init43, label %handle_init_end44, !dbg !27, !prof !33

if_end42:                                         ; preds = %handle_init_end44, %if_end34
  %186 = getelementptr inbounds i64, ptr %gemm.A.shape, i64 1, !dbg !27
  %187 = load i64, ptr %186, align 8, !dbg !27, !tbaa !56
  %188 = and i64 %187, 4294967295, !dbg !27
  %.not380 = icmp eq i64 %188, 1024, !dbg !27
  br i1 %.not380, label %if_end50, label %if_then49, !dbg !27, !prof !33

handle_init43:                                    ; preds = %if_then41
  %189 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !27, !tbaa !30
  %190 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !27, !tbaa !30
  %191 = call i32 %190(ptr %189, ptr nonnull @.str.16, ptr nonnull %26), !dbg !27
  %192 = icmp eq i32 %191, 0, !dbg !27
  br i1 %192, label %call_end46, label %common.ret, !dbg !27, !prof !29

handle_init_end44:                                ; preds = %call_end46, %if_then41
  %193 = phi ptr [ %185, %if_then41 ], [ %196, %call_end46 ], !dbg !27
  %194 = call i32 %184(ptr %193, ptr nonnull %stack_ffi_any375, i32 5, ptr nonnull %183), !dbg !27
  %195 = icmp eq i32 %194, 0, !dbg !27
  br i1 %195, label %if_end42, label %common.ret, !dbg !27, !prof !29

call_end46:                                       ; preds = %handle_init43
  %196 = load ptr, ptr %26, align 8, !dbg !27
  store ptr %196, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !27
  br label %handle_init_end44, !dbg !27

if_then49:                                        ; preds = %if_end42
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any375, align 8, !dbg !27
  %197 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 0, i32 2, !dbg !27
  store ptr @.str.9, ptr %197, align 8, !dbg !27
  %198 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %198, align 8, !dbg !27
  %199 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 2, !dbg !27
  store ptr @.str.10, ptr %199, align 8, !dbg !27
  %200 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %200, align 8, !dbg !27
  %201 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 2, !dbg !27
  store ptr @.str.17, ptr %201, align 8, !dbg !27
  %202 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %202, align 8, !dbg !27
  %203 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 2, !dbg !27
  store i64 1024, ptr %203, align 8, !dbg !27
  %204 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %204, align 8, !dbg !27
  %205 = load i64, ptr %186, align 8, !dbg !27, !tbaa !56
  %sext432 = shl i64 %205, 32, !dbg !27
  %206 = ashr exact i64 %sext432, 32, !dbg !27
  %207 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 2, !dbg !27
  store i64 %206, ptr %207, align 8, !dbg !27
  %208 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 5, i32 0, !dbg !27
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %208, i8 0, i64 16, i1 false), !dbg !27
  %209 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !27, !tbaa !30
  %210 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !27
  %.not433 = icmp eq ptr %210, null, !dbg !27
  br i1 %.not433, label %handle_init51, label %handle_init_end52, !dbg !27, !prof !33

if_end50:                                         ; preds = %handle_init_end52, %if_end42
  %211 = icmp eq ptr %gemm.A.strides, null, !dbg !27
  br i1 %211, label %if_then73, label %if_end58, !dbg !27

handle_init51:                                    ; preds = %if_then49
  %212 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !27, !tbaa !30
  %213 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !27, !tbaa !30
  %214 = call i32 %213(ptr %212, ptr nonnull @.str.16, ptr nonnull %25), !dbg !27
  %215 = icmp eq i32 %214, 0, !dbg !27
  br i1 %215, label %call_end54, label %common.ret, !dbg !27, !prof !29

handle_init_end52:                                ; preds = %call_end54, %if_then49
  %216 = phi ptr [ %210, %if_then49 ], [ %219, %call_end54 ], !dbg !27
  %217 = call i32 %209(ptr %216, ptr nonnull %stack_ffi_any375, i32 5, ptr nonnull %208), !dbg !27
  %218 = icmp eq i32 %217, 0, !dbg !27
  br i1 %218, label %if_end50, label %common.ret, !dbg !27, !prof !29

call_end54:                                       ; preds = %handle_init51
  %219 = load ptr, ptr %25, align 8, !dbg !27
  store ptr %219, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !27
  br label %handle_init_end52, !dbg !27

if_end58:                                         ; preds = %if_end50
  %220 = getelementptr inbounds i64, ptr %gemm.A.strides, i64 1, !dbg !27
  %221 = load i64, ptr %220, align 8, !dbg !27, !tbaa !56
  %222 = and i64 %221, 4294967295, !dbg !27
  %.not381 = icmp eq i64 %222, 1, !dbg !27
  br i1 %.not381, label %if_end72, label %if_end63, !dbg !27, !prof !33

if_end63:                                         ; preds = %if_end58
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any375, align 8, !dbg !27
  %223 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 0, i32 2, !dbg !27
  store ptr @.str.9, ptr %223, align 8, !dbg !27
  %224 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %224, align 8, !dbg !27
  %225 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 2, !dbg !27
  store ptr @.str.10, ptr %225, align 8, !dbg !27
  %226 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %226, align 8, !dbg !27
  %227 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 2, !dbg !27
  store ptr @.str.18, ptr %227, align 8, !dbg !27
  %228 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %228, align 8, !dbg !27
  %229 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 2, !dbg !27
  store i64 1, ptr %229, align 8, !dbg !27
  %230 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %230, align 8, !dbg !27
  %231 = load i64, ptr %220, align 8, !dbg !27, !tbaa !56
  %sext430 = shl i64 %231, 32, !dbg !27
  %232 = ashr exact i64 %sext430, 32, !dbg !27
  %233 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 2, !dbg !27
  store i64 %232, ptr %233, align 8, !dbg !27
  %234 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 5, i32 0, !dbg !27
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %234, i8 0, i64 16, i1 false), !dbg !27
  %235 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !27, !tbaa !30
  %236 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !27
  %.not431 = icmp eq ptr %236, null, !dbg !27
  br i1 %.not431, label %handle_init64, label %handle_init_end65, !dbg !27, !prof !33

handle_init64:                                    ; preds = %if_end63
  %237 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !27, !tbaa !30
  %238 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !27, !tbaa !30
  %239 = call i32 %238(ptr %237, ptr nonnull @.str.16, ptr nonnull %24), !dbg !27
  %240 = icmp eq i32 %239, 0, !dbg !27
  br i1 %240, label %call_end67, label %common.ret, !dbg !27, !prof !29

handle_init_end65:                                ; preds = %call_end67, %if_end63
  %241 = phi ptr [ %236, %if_end63 ], [ %244, %call_end67 ], !dbg !27
  %242 = call i32 %235(ptr %241, ptr nonnull %stack_ffi_any375, i32 5, ptr nonnull %234), !dbg !27
  %243 = icmp eq i32 %242, 0, !dbg !27
  br i1 %243, label %if_end72, label %common.ret, !dbg !27, !prof !29

call_end67:                                       ; preds = %handle_init64
  %244 = load ptr, ptr %24, align 8, !dbg !27
  store ptr %244, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !27
  br label %handle_init_end65, !dbg !27

if_end72:                                         ; preds = %if_end58, %handle_init_end65
  %245 = load i64, ptr %gemm.A.strides, align 8, !dbg !27, !tbaa !56
  %246 = and i64 %245, 4294967295, !dbg !27
  %.not382 = icmp eq i64 %246, 1024, !dbg !27
  br i1 %.not382, label %if_end74, label %if_then73, !dbg !27, !prof !33

if_then73:                                        ; preds = %if_end50, %if_end72
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any375, align 8, !dbg !27
  %247 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 0, i32 2, !dbg !27
  store ptr @.str.9, ptr %247, align 8, !dbg !27
  %248 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %248, align 8, !dbg !27
  %249 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 2, !dbg !27
  store ptr @.str.10, ptr %249, align 8, !dbg !27
  %250 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %250, align 8, !dbg !27
  %251 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 2, !dbg !27
  store ptr @.str.19, ptr %251, align 8, !dbg !27
  %252 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %252, align 8, !dbg !27
  %253 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 2, !dbg !27
  store i64 1024, ptr %253, align 8, !dbg !27
  %254 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %254, align 8, !dbg !27
  br i1 %211, label %if_end77, label %if_else76, !dbg !27

if_end74:                                         ; preds = %handle_init_end79, %if_end72
  %255 = getelementptr inbounds %1, ptr %A_handle, i64 0, i32 6, !dbg !27
  %256 = load i64, ptr %255, align 8, !dbg !27
  %.not383 = icmp eq i64 %256, 0, !dbg !27
  br i1 %.not383, label %if_end85, label %if_then84, !dbg !27, !prof !33

if_else76:                                        ; preds = %if_then73
  %257 = load i64, ptr %gemm.A.strides, align 8, !dbg !27, !tbaa !56
  br label %if_end77, !dbg !27

if_end77:                                         ; preds = %if_then73, %if_else76
  %258 = phi i64 [ %257, %if_else76 ], [ 1, %if_then73 ], !dbg !27
  %sext428 = shl i64 %258, 32, !dbg !27
  %259 = ashr exact i64 %sext428, 32, !dbg !27
  %260 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 2, !dbg !27
  store i64 %259, ptr %260, align 8, !dbg !27
  %261 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 5, i32 0, !dbg !27
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %261, i8 0, i64 16, i1 false), !dbg !27
  %262 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !27, !tbaa !30
  %263 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !27
  %.not429 = icmp eq ptr %263, null, !dbg !27
  br i1 %.not429, label %handle_init78, label %handle_init_end79, !dbg !27, !prof !33

handle_init78:                                    ; preds = %if_end77
  %264 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !27, !tbaa !30
  %265 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !27, !tbaa !30
  %266 = call i32 %265(ptr %264, ptr nonnull @.str.16, ptr nonnull %23), !dbg !27
  %267 = icmp eq i32 %266, 0, !dbg !27
  br i1 %267, label %call_end81, label %common.ret, !dbg !27, !prof !29

handle_init_end79:                                ; preds = %call_end81, %if_end77
  %268 = phi ptr [ %263, %if_end77 ], [ %271, %call_end81 ], !dbg !27
  %269 = call i32 %262(ptr %268, ptr nonnull %stack_ffi_any375, i32 5, ptr nonnull %261), !dbg !27
  %270 = icmp eq i32 %269, 0, !dbg !27
  br i1 %270, label %if_end74, label %common.ret, !dbg !27, !prof !29

call_end81:                                       ; preds = %handle_init78
  %271 = load ptr, ptr %23, align 8, !dbg !27
  store ptr %271, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !27
  br label %handle_init_end79, !dbg !27

if_then84:                                        ; preds = %if_end74
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any375, align 8, !dbg !27
  %272 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 0, i32 2, !dbg !27
  store ptr @.str.9, ptr %272, align 8, !dbg !27
  %273 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %273, align 8, !dbg !27
  %274 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 2, !dbg !27
  store ptr @.str.10, ptr %274, align 8, !dbg !27
  %275 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %275, align 8, !dbg !27
  %276 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 2, !dbg !27
  store i64 0, ptr %276, align 8, !dbg !27
  %277 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %277, align 8, !dbg !27
  %278 = load i64, ptr %255, align 8, !dbg !27
  %279 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 2, !dbg !27
  store i64 %278, ptr %279, align 8, !dbg !27
  %280 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 0, !dbg !27
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %280, i8 0, i64 16, i1 false), !dbg !27
  %281 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !27, !tbaa !30
  %282 = load ptr, ptr @.tvm_func.__tvm_error_byte_offset_mismatch, align 8, !dbg !27
  %.not427 = icmp eq ptr %282, null, !dbg !27
  br i1 %.not427, label %handle_init86, label %handle_init_end87, !dbg !27, !prof !33

if_end85:                                         ; preds = %handle_init_end87, %if_end74
  %283 = getelementptr inbounds %1, ptr %A_handle, i64 0, i32 1, i32 0, !dbg !27
  %284 = load i32, ptr %283, align 4, !dbg !27
  %.not384 = icmp eq i32 %284, 2, !dbg !27
  br i1 %.not384, label %if_end93, label %if_then92, !dbg !27, !prof !33

handle_init86:                                    ; preds = %if_then84
  %285 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !27, !tbaa !30
  %286 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !27, !tbaa !30
  %287 = call i32 %286(ptr %285, ptr nonnull @.str.20, ptr nonnull %22), !dbg !27
  %288 = icmp eq i32 %287, 0, !dbg !27
  br i1 %288, label %call_end89, label %common.ret, !dbg !27, !prof !29

handle_init_end87:                                ; preds = %call_end89, %if_then84
  %289 = phi ptr [ %282, %if_then84 ], [ %292, %call_end89 ], !dbg !27
  %290 = call i32 %281(ptr %289, ptr nonnull %stack_ffi_any375, i32 4, ptr nonnull %280), !dbg !27
  %291 = icmp eq i32 %290, 0, !dbg !27
  br i1 %291, label %if_end85, label %common.ret, !dbg !27, !prof !29

call_end89:                                       ; preds = %handle_init86
  %292 = load ptr, ptr %22, align 8, !dbg !27
  store ptr %292, ptr @.tvm_func.__tvm_error_byte_offset_mismatch, align 8, !dbg !27
  br label %handle_init_end87, !dbg !27

if_then92:                                        ; preds = %if_end85
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any375, align 8, !dbg !27
  %293 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 0, i32 2, !dbg !27
  store ptr @.str.9, ptr %293, align 8, !dbg !27
  %294 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %294, align 8, !dbg !27
  %295 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 2, !dbg !27
  store ptr @.str.10, ptr %295, align 8, !dbg !27
  %296 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %296, align 8, !dbg !27
  %297 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 2, !dbg !27
  store i64 2, ptr %297, align 8, !dbg !27
  %298 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %298, align 8, !dbg !27
  %299 = load i32, ptr %283, align 4, !dbg !27
  %300 = sext i32 %299 to i64, !dbg !27
  %301 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 2, !dbg !27
  store i64 %300, ptr %301, align 8, !dbg !27
  %302 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 0, !dbg !27
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %302, i8 0, i64 16, i1 false), !dbg !27
  %303 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !27, !tbaa !30
  %304 = load ptr, ptr @.tvm_func.__tvm_error_device_type_mismatch, align 8, !dbg !27
  %.not426 = icmp eq ptr %304, null, !dbg !27
  br i1 %.not426, label %handle_init94, label %handle_init_end95, !dbg !27, !prof !33

if_end93:                                         ; preds = %handle_init_end95, %if_end85
  %305 = icmp eq ptr %A, null, !dbg !27
  br i1 %305, label %if_then100, label %if_end101, !dbg !27, !prof !29

handle_init94:                                    ; preds = %if_then92
  %306 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !27, !tbaa !30
  %307 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !27, !tbaa !30
  %308 = call i32 %307(ptr %306, ptr nonnull @.str.21, ptr nonnull %21), !dbg !27
  %309 = icmp eq i32 %308, 0, !dbg !27
  br i1 %309, label %call_end97, label %common.ret, !dbg !27, !prof !29

handle_init_end95:                                ; preds = %call_end97, %if_then92
  %310 = phi ptr [ %304, %if_then92 ], [ %313, %call_end97 ], !dbg !27
  %311 = call i32 %303(ptr %310, ptr nonnull %stack_ffi_any375, i32 4, ptr nonnull %302), !dbg !27
  %312 = icmp eq i32 %311, 0, !dbg !27
  br i1 %312, label %if_end93, label %common.ret, !dbg !27, !prof !29

call_end97:                                       ; preds = %handle_init94
  %313 = load ptr, ptr %21, align 8, !dbg !27
  store ptr %313, ptr @.tvm_func.__tvm_error_device_type_mismatch, align 8, !dbg !27
  br label %handle_init_end95, !dbg !27

if_then100:                                       ; preds = %if_end93
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any375, align 8, !dbg !27
  %314 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 0, i32 2, !dbg !27
  store ptr @.str.9, ptr %314, align 8, !dbg !27
  %315 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %315, align 8, !dbg !27
  %316 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 2, !dbg !27
  store ptr @.str.10, ptr %316, align 8, !dbg !27
  %317 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %317, align 8, !dbg !27
  %318 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 2, !dbg !27
  store ptr @.str.22, ptr %318, align 8, !dbg !27
  %319 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 0, !dbg !27
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %319, i8 0, i64 16, i1 false), !dbg !27
  %320 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !27, !tbaa !30
  %321 = load ptr, ptr @.tvm_func.__tvm_error_null_ptr, align 8, !dbg !27
  %.not425 = icmp eq ptr %321, null, !dbg !27
  br i1 %.not425, label %handle_init102, label %handle_init_end103, !dbg !27, !prof !33

if_end101:                                        ; preds = %handle_init_end103, %if_end93
  %322 = getelementptr inbounds %1, ptr %B_handle, i64 0, i32 3, i32 2, !dbg !27
  %323 = load i16, ptr %322, align 2, !dbg !27
  %324 = icmp ne i16 %323, 1, !dbg !27
  %325 = getelementptr inbounds %1, ptr %B_handle, i64 0, i32 3, i32 1, !dbg !27
  %326 = load i8, ptr %325, align 1, !dbg !27
  %327 = icmp ne i8 %326, 16, !dbg !27
  %328 = getelementptr inbounds %1, ptr %B_handle, i64 0, i32 3, i32 0, !dbg !27
  %329 = load i8, ptr %328, align 1, !dbg !27
  %330 = icmp ne i8 %329, 4, !dbg !27
  %331 = or i1 %327, %330, !dbg !27
  %332 = or i1 %324, %331, !dbg !27
  br i1 %332, label %if_then108, label %if_end109, !dbg !27, !prof !29

handle_init102:                                   ; preds = %if_then100
  %333 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !27, !tbaa !30
  %334 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !27, !tbaa !30
  %335 = call i32 %334(ptr %333, ptr nonnull @.str.23, ptr nonnull %20), !dbg !27
  %336 = icmp eq i32 %335, 0, !dbg !27
  br i1 %336, label %call_end105, label %common.ret, !dbg !27, !prof !29

handle_init_end103:                               ; preds = %call_end105, %if_then100
  %337 = phi ptr [ %321, %if_then100 ], [ %340, %call_end105 ], !dbg !27
  %338 = call i32 %320(ptr %337, ptr nonnull %stack_ffi_any375, i32 3, ptr nonnull %319), !dbg !27
  %339 = icmp eq i32 %338, 0, !dbg !27
  br i1 %339, label %if_end101, label %common.ret, !dbg !27, !prof !29

call_end105:                                      ; preds = %handle_init102
  %340 = load ptr, ptr %20, align 8, !dbg !27
  store ptr %340, ptr @.tvm_func.__tvm_error_null_ptr, align 8, !dbg !27
  br label %handle_init_end103, !dbg !27

if_then108:                                       ; preds = %if_end101
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any375, align 8, !dbg !27
  %341 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 0, i32 2, !dbg !27
  store ptr @.str.9, ptr %341, align 8, !dbg !27
  %342 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %342, align 8, !dbg !27
  %343 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 2, !dbg !27
  store ptr @.str.12, ptr %343, align 8, !dbg !27
  %344 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %344, align 8, !dbg !27
  %345 = load i8, ptr %328, align 1, !dbg !27
  %346 = zext i8 %345 to i64, !dbg !27
  %347 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 2, !dbg !27
  store i64 %346, ptr %347, align 8, !dbg !27
  %348 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %348, align 8, !dbg !27
  %349 = load i8, ptr %325, align 1, !dbg !27
  %350 = zext i8 %349 to i64, !dbg !27
  %351 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 2, !dbg !27
  store i64 %350, ptr %351, align 8, !dbg !27
  %352 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %352, align 8, !dbg !27
  %353 = load i16, ptr %322, align 2, !dbg !27
  %354 = zext i16 %353 to i64, !dbg !27
  %355 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 2, !dbg !27
  store i64 %354, ptr %355, align 8, !dbg !27
  %356 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 5, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %356, align 8, !dbg !27
  %357 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 5, i32 2, !dbg !27
  store i64 4, ptr %357, align 8, !dbg !27
  %358 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 6, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %358, align 8, !dbg !27
  %359 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 6, i32 2, !dbg !27
  store i64 16, ptr %359, align 8, !dbg !27
  %360 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 7, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %360, align 8, !dbg !27
  %361 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 7, i32 2, !dbg !27
  store i64 1, ptr %361, align 8, !dbg !27
  %362 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 8, i32 0, !dbg !27
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %362, i8 0, i64 16, i1 false), !dbg !27
  %363 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !27, !tbaa !30
  %364 = load ptr, ptr @.tvm_func.__tvm_error_dtype_mismatch, align 8, !dbg !27
  %.not424 = icmp eq ptr %364, null, !dbg !27
  br i1 %.not424, label %handle_init110, label %handle_init_end111, !dbg !27, !prof !33

if_end109:                                        ; preds = %handle_init_end111, %if_end101
  %365 = load i64, ptr %gemm.B.shape, align 8, !dbg !27, !tbaa !56
  %366 = and i64 %365, 4294967295, !dbg !27
  %.not385 = icmp eq i64 %366, 1024, !dbg !27
  br i1 %.not385, label %if_end117, label %if_then116, !dbg !27, !prof !33

handle_init110:                                   ; preds = %if_then108
  %367 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !27, !tbaa !30
  %368 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !27, !tbaa !30
  %369 = call i32 %368(ptr %367, ptr nonnull @.str.14, ptr nonnull %19), !dbg !27
  %370 = icmp eq i32 %369, 0, !dbg !27
  br i1 %370, label %call_end113, label %common.ret, !dbg !27, !prof !29

handle_init_end111:                               ; preds = %call_end113, %if_then108
  %371 = phi ptr [ %364, %if_then108 ], [ %374, %call_end113 ], !dbg !27
  %372 = call i32 %363(ptr %371, ptr nonnull %stack_ffi_any375, i32 8, ptr nonnull %362), !dbg !27
  %373 = icmp eq i32 %372, 0, !dbg !27
  br i1 %373, label %if_end109, label %common.ret, !dbg !27, !prof !29

call_end113:                                      ; preds = %handle_init110
  %374 = load ptr, ptr %19, align 8, !dbg !27
  store ptr %374, ptr @.tvm_func.__tvm_error_dtype_mismatch, align 8, !dbg !27
  br label %handle_init_end111, !dbg !27

if_then116:                                       ; preds = %if_end109
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any375, align 8, !dbg !27
  %375 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 0, i32 2, !dbg !27
  store ptr @.str.9, ptr %375, align 8, !dbg !27
  %376 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %376, align 8, !dbg !27
  %377 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 2, !dbg !27
  store ptr @.str.12, ptr %377, align 8, !dbg !27
  %378 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %378, align 8, !dbg !27
  %379 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 2, !dbg !27
  store ptr @.str.15, ptr %379, align 8, !dbg !27
  %380 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %380, align 8, !dbg !27
  %381 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 2, !dbg !27
  store i64 1024, ptr %381, align 8, !dbg !27
  %382 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %382, align 8, !dbg !27
  %383 = load i64, ptr %gemm.B.shape, align 8, !dbg !27, !tbaa !56
  %sext422 = shl i64 %383, 32, !dbg !27
  %384 = ashr exact i64 %sext422, 32, !dbg !27
  %385 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 2, !dbg !27
  store i64 %384, ptr %385, align 8, !dbg !27
  %386 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 5, i32 0, !dbg !27
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %386, i8 0, i64 16, i1 false), !dbg !27
  %387 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !27, !tbaa !30
  %388 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !27
  %.not423 = icmp eq ptr %388, null, !dbg !27
  br i1 %.not423, label %handle_init118, label %handle_init_end119, !dbg !27, !prof !33

if_end117:                                        ; preds = %handle_init_end119, %if_end109
  %389 = getelementptr inbounds i64, ptr %gemm.B.shape, i64 1, !dbg !27
  %390 = load i64, ptr %389, align 8, !dbg !27, !tbaa !56
  %391 = and i64 %390, 4294967295, !dbg !27
  %.not386 = icmp eq i64 %391, 1024, !dbg !27
  br i1 %.not386, label %if_end125, label %if_then124, !dbg !27, !prof !33

handle_init118:                                   ; preds = %if_then116
  %392 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !27, !tbaa !30
  %393 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !27, !tbaa !30
  %394 = call i32 %393(ptr %392, ptr nonnull @.str.16, ptr nonnull %18), !dbg !27
  %395 = icmp eq i32 %394, 0, !dbg !27
  br i1 %395, label %call_end121, label %common.ret, !dbg !27, !prof !29

handle_init_end119:                               ; preds = %call_end121, %if_then116
  %396 = phi ptr [ %388, %if_then116 ], [ %399, %call_end121 ], !dbg !27
  %397 = call i32 %387(ptr %396, ptr nonnull %stack_ffi_any375, i32 5, ptr nonnull %386), !dbg !27
  %398 = icmp eq i32 %397, 0, !dbg !27
  br i1 %398, label %if_end117, label %common.ret, !dbg !27, !prof !29

call_end121:                                      ; preds = %handle_init118
  %399 = load ptr, ptr %18, align 8, !dbg !27
  store ptr %399, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !27
  br label %handle_init_end119, !dbg !27

if_then124:                                       ; preds = %if_end117
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any375, align 8, !dbg !27
  %400 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 0, i32 2, !dbg !27
  store ptr @.str.9, ptr %400, align 8, !dbg !27
  %401 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %401, align 8, !dbg !27
  %402 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 2, !dbg !27
  store ptr @.str.12, ptr %402, align 8, !dbg !27
  %403 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %403, align 8, !dbg !27
  %404 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 2, !dbg !27
  store ptr @.str.17, ptr %404, align 8, !dbg !27
  %405 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %405, align 8, !dbg !27
  %406 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 2, !dbg !27
  store i64 1024, ptr %406, align 8, !dbg !27
  %407 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %407, align 8, !dbg !27
  %408 = load i64, ptr %389, align 8, !dbg !27, !tbaa !56
  %sext420 = shl i64 %408, 32, !dbg !27
  %409 = ashr exact i64 %sext420, 32, !dbg !27
  %410 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 2, !dbg !27
  store i64 %409, ptr %410, align 8, !dbg !27
  %411 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 5, i32 0, !dbg !27
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %411, i8 0, i64 16, i1 false), !dbg !27
  %412 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !27, !tbaa !30
  %413 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !27
  %.not421 = icmp eq ptr %413, null, !dbg !27
  br i1 %.not421, label %handle_init126, label %handle_init_end127, !dbg !27, !prof !33

if_end125:                                        ; preds = %handle_init_end127, %if_end117
  %414 = icmp eq ptr %gemm.B.strides, null, !dbg !27
  br i1 %414, label %if_then149, label %if_end134, !dbg !27

handle_init126:                                   ; preds = %if_then124
  %415 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !27, !tbaa !30
  %416 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !27, !tbaa !30
  %417 = call i32 %416(ptr %415, ptr nonnull @.str.16, ptr nonnull %17), !dbg !27
  %418 = icmp eq i32 %417, 0, !dbg !27
  br i1 %418, label %call_end129, label %common.ret, !dbg !27, !prof !29

handle_init_end127:                               ; preds = %call_end129, %if_then124
  %419 = phi ptr [ %413, %if_then124 ], [ %422, %call_end129 ], !dbg !27
  %420 = call i32 %412(ptr %419, ptr nonnull %stack_ffi_any375, i32 5, ptr nonnull %411), !dbg !27
  %421 = icmp eq i32 %420, 0, !dbg !27
  br i1 %421, label %if_end125, label %common.ret, !dbg !27, !prof !29

call_end129:                                      ; preds = %handle_init126
  %422 = load ptr, ptr %17, align 8, !dbg !27
  store ptr %422, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !27
  br label %handle_init_end127, !dbg !27

if_end134:                                        ; preds = %if_end125
  %423 = getelementptr inbounds i64, ptr %gemm.B.strides, i64 1, !dbg !27
  %424 = load i64, ptr %423, align 8, !dbg !27, !tbaa !56
  %425 = and i64 %424, 4294967295, !dbg !27
  %.not387 = icmp eq i64 %425, 1, !dbg !27
  br i1 %.not387, label %if_end148, label %if_end139, !dbg !27, !prof !33

if_end139:                                        ; preds = %if_end134
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any375, align 8, !dbg !27
  %426 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 0, i32 2, !dbg !27
  store ptr @.str.9, ptr %426, align 8, !dbg !27
  %427 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %427, align 8, !dbg !27
  %428 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 2, !dbg !27
  store ptr @.str.12, ptr %428, align 8, !dbg !27
  %429 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %429, align 8, !dbg !27
  %430 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 2, !dbg !27
  store ptr @.str.18, ptr %430, align 8, !dbg !27
  %431 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %431, align 8, !dbg !27
  %432 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 2, !dbg !27
  store i64 1, ptr %432, align 8, !dbg !27
  %433 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %433, align 8, !dbg !27
  %434 = load i64, ptr %423, align 8, !dbg !27, !tbaa !56
  %sext418 = shl i64 %434, 32, !dbg !27
  %435 = ashr exact i64 %sext418, 32, !dbg !27
  %436 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 2, !dbg !27
  store i64 %435, ptr %436, align 8, !dbg !27
  %437 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 5, i32 0, !dbg !27
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %437, i8 0, i64 16, i1 false), !dbg !27
  %438 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !27, !tbaa !30
  %439 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !27
  %.not419 = icmp eq ptr %439, null, !dbg !27
  br i1 %.not419, label %handle_init140, label %handle_init_end141, !dbg !27, !prof !33

handle_init140:                                   ; preds = %if_end139
  %440 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !27, !tbaa !30
  %441 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !27, !tbaa !30
  %442 = call i32 %441(ptr %440, ptr nonnull @.str.16, ptr nonnull %16), !dbg !27
  %443 = icmp eq i32 %442, 0, !dbg !27
  br i1 %443, label %call_end143, label %common.ret, !dbg !27, !prof !29

handle_init_end141:                               ; preds = %call_end143, %if_end139
  %444 = phi ptr [ %439, %if_end139 ], [ %447, %call_end143 ], !dbg !27
  %445 = call i32 %438(ptr %444, ptr nonnull %stack_ffi_any375, i32 5, ptr nonnull %437), !dbg !27
  %446 = icmp eq i32 %445, 0, !dbg !27
  br i1 %446, label %if_end148, label %common.ret, !dbg !27, !prof !29

call_end143:                                      ; preds = %handle_init140
  %447 = load ptr, ptr %16, align 8, !dbg !27
  store ptr %447, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !27
  br label %handle_init_end141, !dbg !27

if_end148:                                        ; preds = %if_end134, %handle_init_end141
  %448 = load i64, ptr %gemm.B.strides, align 8, !dbg !27, !tbaa !56
  %449 = and i64 %448, 4294967295, !dbg !27
  %.not388 = icmp eq i64 %449, 1024, !dbg !27
  br i1 %.not388, label %if_end150, label %if_then149, !dbg !27, !prof !33

if_then149:                                       ; preds = %if_end125, %if_end148
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any375, align 8, !dbg !27
  %450 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 0, i32 2, !dbg !27
  store ptr @.str.9, ptr %450, align 8, !dbg !27
  %451 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %451, align 8, !dbg !27
  %452 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 2, !dbg !27
  store ptr @.str.12, ptr %452, align 8, !dbg !27
  %453 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %453, align 8, !dbg !27
  %454 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 2, !dbg !27
  store ptr @.str.19, ptr %454, align 8, !dbg !27
  %455 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %455, align 8, !dbg !27
  %456 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 2, !dbg !27
  store i64 1024, ptr %456, align 8, !dbg !27
  %457 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %457, align 8, !dbg !27
  br i1 %414, label %if_end153, label %if_else152, !dbg !27

if_end150:                                        ; preds = %handle_init_end155, %if_end148
  %458 = getelementptr inbounds %1, ptr %B_handle, i64 0, i32 6, !dbg !27
  %459 = load i64, ptr %458, align 8, !dbg !27
  %.not389 = icmp eq i64 %459, 0, !dbg !27
  br i1 %.not389, label %if_end161, label %if_then160, !dbg !27, !prof !33

if_else152:                                       ; preds = %if_then149
  %460 = load i64, ptr %gemm.B.strides, align 8, !dbg !27, !tbaa !56
  br label %if_end153, !dbg !27

if_end153:                                        ; preds = %if_then149, %if_else152
  %461 = phi i64 [ %460, %if_else152 ], [ 1, %if_then149 ], !dbg !27
  %sext416 = shl i64 %461, 32, !dbg !27
  %462 = ashr exact i64 %sext416, 32, !dbg !27
  %463 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 2, !dbg !27
  store i64 %462, ptr %463, align 8, !dbg !27
  %464 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 5, i32 0, !dbg !27
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %464, i8 0, i64 16, i1 false), !dbg !27
  %465 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !27, !tbaa !30
  %466 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !27
  %.not417 = icmp eq ptr %466, null, !dbg !27
  br i1 %.not417, label %handle_init154, label %handle_init_end155, !dbg !27, !prof !33

handle_init154:                                   ; preds = %if_end153
  %467 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !27, !tbaa !30
  %468 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !27, !tbaa !30
  %469 = call i32 %468(ptr %467, ptr nonnull @.str.16, ptr nonnull %15), !dbg !27
  %470 = icmp eq i32 %469, 0, !dbg !27
  br i1 %470, label %call_end157, label %common.ret, !dbg !27, !prof !29

handle_init_end155:                               ; preds = %call_end157, %if_end153
  %471 = phi ptr [ %466, %if_end153 ], [ %474, %call_end157 ], !dbg !27
  %472 = call i32 %465(ptr %471, ptr nonnull %stack_ffi_any375, i32 5, ptr nonnull %464), !dbg !27
  %473 = icmp eq i32 %472, 0, !dbg !27
  br i1 %473, label %if_end150, label %common.ret, !dbg !27, !prof !29

call_end157:                                      ; preds = %handle_init154
  %474 = load ptr, ptr %15, align 8, !dbg !27
  store ptr %474, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !27
  br label %handle_init_end155, !dbg !27

if_then160:                                       ; preds = %if_end150
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any375, align 8, !dbg !27
  %475 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 0, i32 2, !dbg !27
  store ptr @.str.9, ptr %475, align 8, !dbg !27
  %476 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %476, align 8, !dbg !27
  %477 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 2, !dbg !27
  store ptr @.str.12, ptr %477, align 8, !dbg !27
  %478 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %478, align 8, !dbg !27
  %479 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 2, !dbg !27
  store i64 0, ptr %479, align 8, !dbg !27
  %480 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %480, align 8, !dbg !27
  %481 = load i64, ptr %458, align 8, !dbg !27
  %482 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 2, !dbg !27
  store i64 %481, ptr %482, align 8, !dbg !27
  %483 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 0, !dbg !27
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %483, i8 0, i64 16, i1 false), !dbg !27
  %484 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !27, !tbaa !30
  %485 = load ptr, ptr @.tvm_func.__tvm_error_byte_offset_mismatch, align 8, !dbg !27
  %.not415 = icmp eq ptr %485, null, !dbg !27
  br i1 %.not415, label %handle_init162, label %handle_init_end163, !dbg !27, !prof !33

if_end161:                                        ; preds = %handle_init_end163, %if_end150
  %486 = getelementptr inbounds %1, ptr %B_handle, i64 0, i32 1, i32 1, !dbg !27
  %487 = load i32, ptr %486, align 4, !dbg !27
  %488 = load i32, ptr %72, align 4, !dbg !27
  %.not390 = icmp eq i32 %487, %488, !dbg !27
  br i1 %.not390, label %if_end169, label %if_then168, !dbg !27, !prof !33

handle_init162:                                   ; preds = %if_then160
  %489 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !27, !tbaa !30
  %490 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !27, !tbaa !30
  %491 = call i32 %490(ptr %489, ptr nonnull @.str.20, ptr nonnull %14), !dbg !27
  %492 = icmp eq i32 %491, 0, !dbg !27
  br i1 %492, label %call_end165, label %common.ret, !dbg !27, !prof !29

handle_init_end163:                               ; preds = %call_end165, %if_then160
  %493 = phi ptr [ %485, %if_then160 ], [ %496, %call_end165 ], !dbg !27
  %494 = call i32 %484(ptr %493, ptr nonnull %stack_ffi_any375, i32 4, ptr nonnull %483), !dbg !27
  %495 = icmp eq i32 %494, 0, !dbg !27
  br i1 %495, label %if_end161, label %common.ret, !dbg !27, !prof !29

call_end165:                                      ; preds = %handle_init162
  %496 = load ptr, ptr %14, align 8, !dbg !27
  store ptr %496, ptr @.tvm_func.__tvm_error_byte_offset_mismatch, align 8, !dbg !27
  br label %handle_init_end163, !dbg !27

if_then168:                                       ; preds = %if_end161
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any375, align 8, !dbg !27
  %497 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 0, i32 2, !dbg !27
  store ptr @.str.9, ptr %497, align 8, !dbg !27
  %498 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %498, align 8, !dbg !27
  %499 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 2, !dbg !27
  store ptr @.str.12, ptr %499, align 8, !dbg !27
  %500 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %500, align 8, !dbg !27
  %501 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 2, !dbg !27
  store ptr @.str.24, ptr %501, align 8, !dbg !27
  %502 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %502, align 8, !dbg !27
  %503 = load i32, ptr %72, align 4, !dbg !27
  %504 = sext i32 %503 to i64, !dbg !27
  %505 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 2, !dbg !27
  store i64 %504, ptr %505, align 8, !dbg !27
  %506 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %506, align 8, !dbg !27
  %507 = load i32, ptr %486, align 4, !dbg !27
  %508 = sext i32 %507 to i64, !dbg !27
  %509 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 2, !dbg !27
  store i64 %508, ptr %509, align 8, !dbg !27
  %510 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 5, i32 0, !dbg !27
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %510, i8 0, i64 16, i1 false), !dbg !27
  %511 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !27, !tbaa !30
  %512 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !27
  %.not414 = icmp eq ptr %512, null, !dbg !27
  br i1 %.not414, label %handle_init170, label %handle_init_end171, !dbg !27, !prof !33

if_end169:                                        ; preds = %handle_init_end171, %if_end161
  %513 = getelementptr inbounds %1, ptr %B_handle, i64 0, i32 1, i32 0, !dbg !27
  %514 = load i32, ptr %513, align 4, !dbg !27
  %.not391 = icmp eq i32 %514, 2, !dbg !27
  br i1 %.not391, label %if_end177, label %if_then176, !dbg !27, !prof !33

handle_init170:                                   ; preds = %if_then168
  %515 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !27, !tbaa !30
  %516 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !27, !tbaa !30
  %517 = call i32 %516(ptr %515, ptr nonnull @.str.16, ptr nonnull %13), !dbg !27
  %518 = icmp eq i32 %517, 0, !dbg !27
  br i1 %518, label %call_end173, label %common.ret, !dbg !27, !prof !29

handle_init_end171:                               ; preds = %call_end173, %if_then168
  %519 = phi ptr [ %512, %if_then168 ], [ %522, %call_end173 ], !dbg !27
  %520 = call i32 %511(ptr %519, ptr nonnull %stack_ffi_any375, i32 5, ptr nonnull %510), !dbg !27
  %521 = icmp eq i32 %520, 0, !dbg !27
  br i1 %521, label %if_end169, label %common.ret, !dbg !27, !prof !29

call_end173:                                      ; preds = %handle_init170
  %522 = load ptr, ptr %13, align 8, !dbg !27
  store ptr %522, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !27
  br label %handle_init_end171, !dbg !27

if_then176:                                       ; preds = %if_end169
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any375, align 8, !dbg !27
  %523 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 0, i32 2, !dbg !27
  store ptr @.str.9, ptr %523, align 8, !dbg !27
  %524 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %524, align 8, !dbg !27
  %525 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 2, !dbg !27
  store ptr @.str.12, ptr %525, align 8, !dbg !27
  %526 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %526, align 8, !dbg !27
  %527 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 2, !dbg !27
  store i64 2, ptr %527, align 8, !dbg !27
  %528 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %528, align 8, !dbg !27
  %529 = load i32, ptr %513, align 4, !dbg !27
  %530 = sext i32 %529 to i64, !dbg !27
  %531 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 2, !dbg !27
  store i64 %530, ptr %531, align 8, !dbg !27
  %532 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 0, !dbg !27
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %532, i8 0, i64 16, i1 false), !dbg !27
  %533 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !27, !tbaa !30
  %534 = load ptr, ptr @.tvm_func.__tvm_error_device_type_mismatch, align 8, !dbg !27
  %.not413 = icmp eq ptr %534, null, !dbg !27
  br i1 %.not413, label %handle_init178, label %handle_init_end179, !dbg !27, !prof !33

if_end177:                                        ; preds = %handle_init_end179, %if_end169
  %535 = icmp eq ptr %B, null, !dbg !27
  br i1 %535, label %if_then184, label %if_end185, !dbg !27, !prof !29

handle_init178:                                   ; preds = %if_then176
  %536 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !27, !tbaa !30
  %537 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !27, !tbaa !30
  %538 = call i32 %537(ptr %536, ptr nonnull @.str.21, ptr nonnull %12), !dbg !27
  %539 = icmp eq i32 %538, 0, !dbg !27
  br i1 %539, label %call_end181, label %common.ret, !dbg !27, !prof !29

handle_init_end179:                               ; preds = %call_end181, %if_then176
  %540 = phi ptr [ %534, %if_then176 ], [ %543, %call_end181 ], !dbg !27
  %541 = call i32 %533(ptr %540, ptr nonnull %stack_ffi_any375, i32 4, ptr nonnull %532), !dbg !27
  %542 = icmp eq i32 %541, 0, !dbg !27
  br i1 %542, label %if_end177, label %common.ret, !dbg !27, !prof !29

call_end181:                                      ; preds = %handle_init178
  %543 = load ptr, ptr %12, align 8, !dbg !27
  store ptr %543, ptr @.tvm_func.__tvm_error_device_type_mismatch, align 8, !dbg !27
  br label %handle_init_end179, !dbg !27

if_then184:                                       ; preds = %if_end177
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any375, align 8, !dbg !27
  %544 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 0, i32 2, !dbg !27
  store ptr @.str.9, ptr %544, align 8, !dbg !27
  %545 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %545, align 8, !dbg !27
  %546 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 2, !dbg !27
  store ptr @.str.12, ptr %546, align 8, !dbg !27
  %547 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %547, align 8, !dbg !27
  %548 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 2, !dbg !27
  store ptr @.str.22, ptr %548, align 8, !dbg !27
  %549 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 0, !dbg !27
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %549, i8 0, i64 16, i1 false), !dbg !27
  %550 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !27, !tbaa !30
  %551 = load ptr, ptr @.tvm_func.__tvm_error_null_ptr, align 8, !dbg !27
  %.not412 = icmp eq ptr %551, null, !dbg !27
  br i1 %.not412, label %handle_init186, label %handle_init_end187, !dbg !27, !prof !33

if_end185:                                        ; preds = %handle_init_end187, %if_end177
  %552 = getelementptr inbounds %1, ptr %C_handle, i64 0, i32 3, i32 2, !dbg !27
  %553 = load i16, ptr %552, align 2, !dbg !27
  %554 = icmp ne i16 %553, 1, !dbg !27
  %555 = getelementptr inbounds %1, ptr %C_handle, i64 0, i32 3, i32 1, !dbg !27
  %556 = load i8, ptr %555, align 1, !dbg !27
  %557 = icmp ne i8 %556, 16, !dbg !27
  %558 = getelementptr inbounds %1, ptr %C_handle, i64 0, i32 3, i32 0, !dbg !27
  %559 = load i8, ptr %558, align 1, !dbg !27
  %560 = icmp ne i8 %559, 4, !dbg !27
  %561 = or i1 %557, %560, !dbg !27
  %562 = or i1 %554, %561, !dbg !27
  br i1 %562, label %if_then192, label %if_end193, !dbg !27, !prof !29

handle_init186:                                   ; preds = %if_then184
  %563 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !27, !tbaa !30
  %564 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !27, !tbaa !30
  %565 = call i32 %564(ptr %563, ptr nonnull @.str.23, ptr nonnull %11), !dbg !27
  %566 = icmp eq i32 %565, 0, !dbg !27
  br i1 %566, label %call_end189, label %common.ret, !dbg !27, !prof !29

handle_init_end187:                               ; preds = %call_end189, %if_then184
  %567 = phi ptr [ %551, %if_then184 ], [ %570, %call_end189 ], !dbg !27
  %568 = call i32 %550(ptr %567, ptr nonnull %stack_ffi_any375, i32 3, ptr nonnull %549), !dbg !27
  %569 = icmp eq i32 %568, 0, !dbg !27
  br i1 %569, label %if_end185, label %common.ret, !dbg !27, !prof !29

call_end189:                                      ; preds = %handle_init186
  %570 = load ptr, ptr %11, align 8, !dbg !27
  store ptr %570, ptr @.tvm_func.__tvm_error_null_ptr, align 8, !dbg !27
  br label %handle_init_end187, !dbg !27

if_then192:                                       ; preds = %if_end185
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any375, align 8, !dbg !27
  %571 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 0, i32 2, !dbg !27
  store ptr @.str.9, ptr %571, align 8, !dbg !27
  %572 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %572, align 8, !dbg !27
  %573 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 2, !dbg !27
  store ptr @.str.13, ptr %573, align 8, !dbg !27
  %574 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %574, align 8, !dbg !27
  %575 = load i8, ptr %558, align 1, !dbg !27
  %576 = zext i8 %575 to i64, !dbg !27
  %577 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 2, !dbg !27
  store i64 %576, ptr %577, align 8, !dbg !27
  %578 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %578, align 8, !dbg !27
  %579 = load i8, ptr %555, align 1, !dbg !27
  %580 = zext i8 %579 to i64, !dbg !27
  %581 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 2, !dbg !27
  store i64 %580, ptr %581, align 8, !dbg !27
  %582 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %582, align 8, !dbg !27
  %583 = load i16, ptr %552, align 2, !dbg !27
  %584 = zext i16 %583 to i64, !dbg !27
  %585 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 2, !dbg !27
  store i64 %584, ptr %585, align 8, !dbg !27
  %586 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 5, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %586, align 8, !dbg !27
  %587 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 5, i32 2, !dbg !27
  store i64 4, ptr %587, align 8, !dbg !27
  %588 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 6, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %588, align 8, !dbg !27
  %589 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 6, i32 2, !dbg !27
  store i64 16, ptr %589, align 8, !dbg !27
  %590 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 7, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %590, align 8, !dbg !27
  %591 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 7, i32 2, !dbg !27
  store i64 1, ptr %591, align 8, !dbg !27
  %592 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 8, i32 0, !dbg !27
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %592, i8 0, i64 16, i1 false), !dbg !27
  %593 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !27, !tbaa !30
  %594 = load ptr, ptr @.tvm_func.__tvm_error_dtype_mismatch, align 8, !dbg !27
  %.not411 = icmp eq ptr %594, null, !dbg !27
  br i1 %.not411, label %handle_init194, label %handle_init_end195, !dbg !27, !prof !33

if_end193:                                        ; preds = %handle_init_end195, %if_end185
  %595 = load i64, ptr %gemm.C.shape, align 8, !dbg !27, !tbaa !56
  %596 = and i64 %595, 4294967295, !dbg !27
  %.not392 = icmp eq i64 %596, 1024, !dbg !27
  br i1 %.not392, label %if_end201, label %if_then200, !dbg !27, !prof !33

handle_init194:                                   ; preds = %if_then192
  %597 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !27, !tbaa !30
  %598 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !27, !tbaa !30
  %599 = call i32 %598(ptr %597, ptr nonnull @.str.14, ptr nonnull %10), !dbg !27
  %600 = icmp eq i32 %599, 0, !dbg !27
  br i1 %600, label %call_end197, label %common.ret, !dbg !27, !prof !29

handle_init_end195:                               ; preds = %call_end197, %if_then192
  %601 = phi ptr [ %594, %if_then192 ], [ %604, %call_end197 ], !dbg !27
  %602 = call i32 %593(ptr %601, ptr nonnull %stack_ffi_any375, i32 8, ptr nonnull %592), !dbg !27
  %603 = icmp eq i32 %602, 0, !dbg !27
  br i1 %603, label %if_end193, label %common.ret, !dbg !27, !prof !29

call_end197:                                      ; preds = %handle_init194
  %604 = load ptr, ptr %10, align 8, !dbg !27
  store ptr %604, ptr @.tvm_func.__tvm_error_dtype_mismatch, align 8, !dbg !27
  br label %handle_init_end195, !dbg !27

if_then200:                                       ; preds = %if_end193
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any375, align 8, !dbg !27
  %605 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 0, i32 2, !dbg !27
  store ptr @.str.9, ptr %605, align 8, !dbg !27
  %606 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %606, align 8, !dbg !27
  %607 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 2, !dbg !27
  store ptr @.str.13, ptr %607, align 8, !dbg !27
  %608 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %608, align 8, !dbg !27
  %609 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 2, !dbg !27
  store ptr @.str.15, ptr %609, align 8, !dbg !27
  %610 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %610, align 8, !dbg !27
  %611 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 2, !dbg !27
  store i64 1024, ptr %611, align 8, !dbg !27
  %612 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %612, align 8, !dbg !27
  %613 = load i64, ptr %gemm.C.shape, align 8, !dbg !27, !tbaa !56
  %sext409 = shl i64 %613, 32, !dbg !27
  %614 = ashr exact i64 %sext409, 32, !dbg !27
  %615 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 2, !dbg !27
  store i64 %614, ptr %615, align 8, !dbg !27
  %616 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 5, i32 0, !dbg !27
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %616, i8 0, i64 16, i1 false), !dbg !27
  %617 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !27, !tbaa !30
  %618 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !27
  %.not410 = icmp eq ptr %618, null, !dbg !27
  br i1 %.not410, label %handle_init202, label %handle_init_end203, !dbg !27, !prof !33

if_end201:                                        ; preds = %handle_init_end203, %if_end193
  %619 = getelementptr inbounds i64, ptr %gemm.C.shape, i64 1, !dbg !27
  %620 = load i64, ptr %619, align 8, !dbg !27, !tbaa !56
  %621 = and i64 %620, 4294967295, !dbg !27
  %.not393 = icmp eq i64 %621, 1024, !dbg !27
  br i1 %.not393, label %if_end209, label %if_then208, !dbg !27, !prof !33

handle_init202:                                   ; preds = %if_then200
  %622 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !27, !tbaa !30
  %623 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !27, !tbaa !30
  %624 = call i32 %623(ptr %622, ptr nonnull @.str.16, ptr nonnull %9), !dbg !27
  %625 = icmp eq i32 %624, 0, !dbg !27
  br i1 %625, label %call_end205, label %common.ret, !dbg !27, !prof !29

handle_init_end203:                               ; preds = %call_end205, %if_then200
  %626 = phi ptr [ %618, %if_then200 ], [ %629, %call_end205 ], !dbg !27
  %627 = call i32 %617(ptr %626, ptr nonnull %stack_ffi_any375, i32 5, ptr nonnull %616), !dbg !27
  %628 = icmp eq i32 %627, 0, !dbg !27
  br i1 %628, label %if_end201, label %common.ret, !dbg !27, !prof !29

call_end205:                                      ; preds = %handle_init202
  %629 = load ptr, ptr %9, align 8, !dbg !27
  store ptr %629, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !27
  br label %handle_init_end203, !dbg !27

if_then208:                                       ; preds = %if_end201
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any375, align 8, !dbg !27
  %630 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 0, i32 2, !dbg !27
  store ptr @.str.9, ptr %630, align 8, !dbg !27
  %631 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %631, align 8, !dbg !27
  %632 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 2, !dbg !27
  store ptr @.str.13, ptr %632, align 8, !dbg !27
  %633 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %633, align 8, !dbg !27
  %634 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 2, !dbg !27
  store ptr @.str.17, ptr %634, align 8, !dbg !27
  %635 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %635, align 8, !dbg !27
  %636 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 2, !dbg !27
  store i64 1024, ptr %636, align 8, !dbg !27
  %637 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %637, align 8, !dbg !27
  %638 = load i64, ptr %619, align 8, !dbg !27, !tbaa !56
  %sext407 = shl i64 %638, 32, !dbg !27
  %639 = ashr exact i64 %sext407, 32, !dbg !27
  %640 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 2, !dbg !27
  store i64 %639, ptr %640, align 8, !dbg !27
  %641 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 5, i32 0, !dbg !27
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %641, i8 0, i64 16, i1 false), !dbg !27
  %642 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !27, !tbaa !30
  %643 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !27
  %.not408 = icmp eq ptr %643, null, !dbg !27
  br i1 %.not408, label %handle_init210, label %handle_init_end211, !dbg !27, !prof !33

if_end209:                                        ; preds = %handle_init_end211, %if_end201
  %644 = icmp eq ptr %gemm.C.strides, null, !dbg !27
  br i1 %644, label %if_then233, label %if_end218, !dbg !27

handle_init210:                                   ; preds = %if_then208
  %645 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !27, !tbaa !30
  %646 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !27, !tbaa !30
  %647 = call i32 %646(ptr %645, ptr nonnull @.str.16, ptr nonnull %8), !dbg !27
  %648 = icmp eq i32 %647, 0, !dbg !27
  br i1 %648, label %call_end213, label %common.ret, !dbg !27, !prof !29

handle_init_end211:                               ; preds = %call_end213, %if_then208
  %649 = phi ptr [ %643, %if_then208 ], [ %652, %call_end213 ], !dbg !27
  %650 = call i32 %642(ptr %649, ptr nonnull %stack_ffi_any375, i32 5, ptr nonnull %641), !dbg !27
  %651 = icmp eq i32 %650, 0, !dbg !27
  br i1 %651, label %if_end209, label %common.ret, !dbg !27, !prof !29

call_end213:                                      ; preds = %handle_init210
  %652 = load ptr, ptr %8, align 8, !dbg !27
  store ptr %652, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !27
  br label %handle_init_end211, !dbg !27

if_end218:                                        ; preds = %if_end209
  %653 = getelementptr inbounds i64, ptr %gemm.C.strides, i64 1, !dbg !27
  %654 = load i64, ptr %653, align 8, !dbg !27, !tbaa !56
  %655 = and i64 %654, 4294967295, !dbg !27
  %.not394 = icmp eq i64 %655, 1, !dbg !27
  br i1 %.not394, label %if_end232, label %if_end223, !dbg !27, !prof !33

if_end223:                                        ; preds = %if_end218
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any375, align 8, !dbg !27
  %656 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 0, i32 2, !dbg !27
  store ptr @.str.9, ptr %656, align 8, !dbg !27
  %657 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %657, align 8, !dbg !27
  %658 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 2, !dbg !27
  store ptr @.str.13, ptr %658, align 8, !dbg !27
  %659 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %659, align 8, !dbg !27
  %660 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 2, !dbg !27
  store ptr @.str.18, ptr %660, align 8, !dbg !27
  %661 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %661, align 8, !dbg !27
  %662 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 2, !dbg !27
  store i64 1, ptr %662, align 8, !dbg !27
  %663 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %663, align 8, !dbg !27
  %664 = load i64, ptr %653, align 8, !dbg !27, !tbaa !56
  %sext405 = shl i64 %664, 32, !dbg !27
  %665 = ashr exact i64 %sext405, 32, !dbg !27
  %666 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 2, !dbg !27
  store i64 %665, ptr %666, align 8, !dbg !27
  %667 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 5, i32 0, !dbg !27
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %667, i8 0, i64 16, i1 false), !dbg !27
  %668 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !27, !tbaa !30
  %669 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !27
  %.not406 = icmp eq ptr %669, null, !dbg !27
  br i1 %.not406, label %handle_init224, label %handle_init_end225, !dbg !27, !prof !33

handle_init224:                                   ; preds = %if_end223
  %670 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !27, !tbaa !30
  %671 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !27, !tbaa !30
  %672 = call i32 %671(ptr %670, ptr nonnull @.str.16, ptr nonnull %7), !dbg !27
  %673 = icmp eq i32 %672, 0, !dbg !27
  br i1 %673, label %call_end227, label %common.ret, !dbg !27, !prof !29

handle_init_end225:                               ; preds = %call_end227, %if_end223
  %674 = phi ptr [ %669, %if_end223 ], [ %677, %call_end227 ], !dbg !27
  %675 = call i32 %668(ptr %674, ptr nonnull %stack_ffi_any375, i32 5, ptr nonnull %667), !dbg !27
  %676 = icmp eq i32 %675, 0, !dbg !27
  br i1 %676, label %if_end232, label %common.ret, !dbg !27, !prof !29

call_end227:                                      ; preds = %handle_init224
  %677 = load ptr, ptr %7, align 8, !dbg !27
  store ptr %677, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !27
  br label %handle_init_end225, !dbg !27

if_end232:                                        ; preds = %if_end218, %handle_init_end225
  %678 = load i64, ptr %gemm.C.strides, align 8, !dbg !27, !tbaa !56
  %679 = and i64 %678, 4294967295, !dbg !27
  %.not395 = icmp eq i64 %679, 1024, !dbg !27
  br i1 %.not395, label %if_end234, label %if_then233, !dbg !27, !prof !33

if_then233:                                       ; preds = %if_end209, %if_end232
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any375, align 8, !dbg !27
  %680 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 0, i32 2, !dbg !27
  store ptr @.str.9, ptr %680, align 8, !dbg !27
  %681 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %681, align 8, !dbg !27
  %682 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 2, !dbg !27
  store ptr @.str.13, ptr %682, align 8, !dbg !27
  %683 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %683, align 8, !dbg !27
  %684 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 2, !dbg !27
  store ptr @.str.19, ptr %684, align 8, !dbg !27
  %685 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %685, align 8, !dbg !27
  %686 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 2, !dbg !27
  store i64 1024, ptr %686, align 8, !dbg !27
  %687 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %687, align 8, !dbg !27
  br i1 %644, label %if_end237, label %if_else236, !dbg !27

if_end234:                                        ; preds = %handle_init_end239, %if_end232
  %688 = getelementptr inbounds %1, ptr %C_handle, i64 0, i32 6, !dbg !27
  %689 = load i64, ptr %688, align 8, !dbg !27
  %.not396 = icmp eq i64 %689, 0, !dbg !27
  br i1 %.not396, label %if_end245, label %if_then244, !dbg !27, !prof !33

if_else236:                                       ; preds = %if_then233
  %690 = load i64, ptr %gemm.C.strides, align 8, !dbg !27, !tbaa !56
  br label %if_end237, !dbg !27

if_end237:                                        ; preds = %if_then233, %if_else236
  %691 = phi i64 [ %690, %if_else236 ], [ 1, %if_then233 ], !dbg !27
  %sext = shl i64 %691, 32, !dbg !27
  %692 = ashr exact i64 %sext, 32, !dbg !27
  %693 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 2, !dbg !27
  store i64 %692, ptr %693, align 8, !dbg !27
  %694 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 5, i32 0, !dbg !27
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %694, i8 0, i64 16, i1 false), !dbg !27
  %695 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !27, !tbaa !30
  %696 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !27
  %.not404 = icmp eq ptr %696, null, !dbg !27
  br i1 %.not404, label %handle_init238, label %handle_init_end239, !dbg !27, !prof !33

handle_init238:                                   ; preds = %if_end237
  %697 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !27, !tbaa !30
  %698 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !27, !tbaa !30
  %699 = call i32 %698(ptr %697, ptr nonnull @.str.16, ptr nonnull %6), !dbg !27
  %700 = icmp eq i32 %699, 0, !dbg !27
  br i1 %700, label %call_end241, label %common.ret, !dbg !27, !prof !29

handle_init_end239:                               ; preds = %call_end241, %if_end237
  %701 = phi ptr [ %696, %if_end237 ], [ %704, %call_end241 ], !dbg !27
  %702 = call i32 %695(ptr %701, ptr nonnull %stack_ffi_any375, i32 5, ptr nonnull %694), !dbg !27
  %703 = icmp eq i32 %702, 0, !dbg !27
  br i1 %703, label %if_end234, label %common.ret, !dbg !27, !prof !29

call_end241:                                      ; preds = %handle_init238
  %704 = load ptr, ptr %6, align 8, !dbg !27
  store ptr %704, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !27
  br label %handle_init_end239, !dbg !27

if_then244:                                       ; preds = %if_end234
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any375, align 8, !dbg !27
  %705 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 0, i32 2, !dbg !27
  store ptr @.str.9, ptr %705, align 8, !dbg !27
  %706 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %706, align 8, !dbg !27
  %707 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 2, !dbg !27
  store ptr @.str.13, ptr %707, align 8, !dbg !27
  %708 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %708, align 8, !dbg !27
  %709 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 2, !dbg !27
  store i64 0, ptr %709, align 8, !dbg !27
  %710 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %710, align 8, !dbg !27
  %711 = load i64, ptr %688, align 8, !dbg !27
  %712 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 2, !dbg !27
  store i64 %711, ptr %712, align 8, !dbg !27
  %713 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 0, !dbg !27
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %713, i8 0, i64 16, i1 false), !dbg !27
  %714 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !27, !tbaa !30
  %715 = load ptr, ptr @.tvm_func.__tvm_error_byte_offset_mismatch, align 8, !dbg !27
  %.not403 = icmp eq ptr %715, null, !dbg !27
  br i1 %.not403, label %handle_init246, label %handle_init_end247, !dbg !27, !prof !33

if_end245:                                        ; preds = %handle_init_end247, %if_end234
  %716 = getelementptr inbounds %1, ptr %C_handle, i64 0, i32 1, i32 1, !dbg !27
  %717 = load i32, ptr %716, align 4, !dbg !27
  %718 = load i32, ptr %72, align 4, !dbg !27
  %.not397 = icmp eq i32 %717, %718, !dbg !27
  br i1 %.not397, label %if_end253, label %if_then252, !dbg !27, !prof !33

handle_init246:                                   ; preds = %if_then244
  %719 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !27, !tbaa !30
  %720 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !27, !tbaa !30
  %721 = call i32 %720(ptr %719, ptr nonnull @.str.20, ptr nonnull %5), !dbg !27
  %722 = icmp eq i32 %721, 0, !dbg !27
  br i1 %722, label %call_end249, label %common.ret, !dbg !27, !prof !29

handle_init_end247:                               ; preds = %call_end249, %if_then244
  %723 = phi ptr [ %715, %if_then244 ], [ %726, %call_end249 ], !dbg !27
  %724 = call i32 %714(ptr %723, ptr nonnull %stack_ffi_any375, i32 4, ptr nonnull %713), !dbg !27
  %725 = icmp eq i32 %724, 0, !dbg !27
  br i1 %725, label %if_end245, label %common.ret, !dbg !27, !prof !29

call_end249:                                      ; preds = %handle_init246
  %726 = load ptr, ptr %5, align 8, !dbg !27
  store ptr %726, ptr @.tvm_func.__tvm_error_byte_offset_mismatch, align 8, !dbg !27
  br label %handle_init_end247, !dbg !27

if_then252:                                       ; preds = %if_end245
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any375, align 8, !dbg !27
  %727 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 0, i32 2, !dbg !27
  store ptr @.str.9, ptr %727, align 8, !dbg !27
  %728 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %728, align 8, !dbg !27
  %729 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 2, !dbg !27
  store ptr @.str.13, ptr %729, align 8, !dbg !27
  %730 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %730, align 8, !dbg !27
  %731 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 2, !dbg !27
  store ptr @.str.24, ptr %731, align 8, !dbg !27
  %732 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %732, align 8, !dbg !27
  %733 = load i32, ptr %72, align 4, !dbg !27
  %734 = sext i32 %733 to i64, !dbg !27
  %735 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 2, !dbg !27
  store i64 %734, ptr %735, align 8, !dbg !27
  %736 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %736, align 8, !dbg !27
  %737 = load i32, ptr %716, align 4, !dbg !27
  %738 = sext i32 %737 to i64, !dbg !27
  %739 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 2, !dbg !27
  store i64 %738, ptr %739, align 8, !dbg !27
  %740 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 5, i32 0, !dbg !27
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %740, i8 0, i64 16, i1 false), !dbg !27
  %741 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !27, !tbaa !30
  %742 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !27
  %.not402 = icmp eq ptr %742, null, !dbg !27
  br i1 %.not402, label %handle_init254, label %handle_init_end255, !dbg !27, !prof !33

if_end253:                                        ; preds = %handle_init_end255, %if_end245
  %743 = getelementptr inbounds %1, ptr %C_handle, i64 0, i32 1, i32 0, !dbg !27
  %744 = load i32, ptr %743, align 4, !dbg !27
  %.not398 = icmp eq i32 %744, 2, !dbg !27
  br i1 %.not398, label %if_end261, label %if_then260, !dbg !27, !prof !33

handle_init254:                                   ; preds = %if_then252
  %745 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !27, !tbaa !30
  %746 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !27, !tbaa !30
  %747 = call i32 %746(ptr %745, ptr nonnull @.str.16, ptr nonnull %4), !dbg !27
  %748 = icmp eq i32 %747, 0, !dbg !27
  br i1 %748, label %call_end257, label %common.ret, !dbg !27, !prof !29

handle_init_end255:                               ; preds = %call_end257, %if_then252
  %749 = phi ptr [ %742, %if_then252 ], [ %752, %call_end257 ], !dbg !27
  %750 = call i32 %741(ptr %749, ptr nonnull %stack_ffi_any375, i32 5, ptr nonnull %740), !dbg !27
  %751 = icmp eq i32 %750, 0, !dbg !27
  br i1 %751, label %if_end253, label %common.ret, !dbg !27, !prof !29

call_end257:                                      ; preds = %handle_init254
  %752 = load ptr, ptr %4, align 8, !dbg !27
  store ptr %752, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !27
  br label %handle_init_end255, !dbg !27

if_then260:                                       ; preds = %if_end253
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any375, align 8, !dbg !27
  %753 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 0, i32 2, !dbg !27
  store ptr @.str.9, ptr %753, align 8, !dbg !27
  %754 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %754, align 8, !dbg !27
  %755 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 2, !dbg !27
  store ptr @.str.13, ptr %755, align 8, !dbg !27
  %756 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %756, align 8, !dbg !27
  %757 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 2, !dbg !27
  store i64 2, ptr %757, align 8, !dbg !27
  %758 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %758, align 8, !dbg !27
  %759 = load i32, ptr %743, align 4, !dbg !27
  %760 = sext i32 %759 to i64, !dbg !27
  %761 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 2, !dbg !27
  store i64 %760, ptr %761, align 8, !dbg !27
  %762 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 0, !dbg !27
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %762, i8 0, i64 16, i1 false), !dbg !27
  %763 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !27, !tbaa !30
  %764 = load ptr, ptr @.tvm_func.__tvm_error_device_type_mismatch, align 8, !dbg !27
  %.not401 = icmp eq ptr %764, null, !dbg !27
  br i1 %.not401, label %handle_init262, label %handle_init_end263, !dbg !27, !prof !33

if_end261:                                        ; preds = %handle_init_end263, %if_end253
  %765 = icmp eq ptr %C, null, !dbg !27
  br i1 %765, label %if_then268, label %if_end269, !dbg !27, !prof !29

handle_init262:                                   ; preds = %if_then260
  %766 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !27, !tbaa !30
  %767 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !27, !tbaa !30
  %768 = call i32 %767(ptr %766, ptr nonnull @.str.21, ptr nonnull %3), !dbg !27
  %769 = icmp eq i32 %768, 0, !dbg !27
  br i1 %769, label %call_end265, label %common.ret, !dbg !27, !prof !29

handle_init_end263:                               ; preds = %call_end265, %if_then260
  %770 = phi ptr [ %764, %if_then260 ], [ %773, %call_end265 ], !dbg !27
  %771 = call i32 %763(ptr %770, ptr nonnull %stack_ffi_any375, i32 4, ptr nonnull %762), !dbg !27
  %772 = icmp eq i32 %771, 0, !dbg !27
  br i1 %772, label %if_end261, label %common.ret, !dbg !27, !prof !29

call_end265:                                      ; preds = %handle_init262
  %773 = load ptr, ptr %3, align 8, !dbg !27
  store ptr %773, ptr @.tvm_func.__tvm_error_device_type_mismatch, align 8, !dbg !27
  br label %handle_init_end263, !dbg !27

if_then268:                                       ; preds = %if_end261
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any375, align 8, !dbg !27
  %774 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 0, i32 2, !dbg !27
  store ptr @.str.9, ptr %774, align 8, !dbg !27
  %775 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %775, align 8, !dbg !27
  %776 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 2, !dbg !27
  store ptr @.str.13, ptr %776, align 8, !dbg !27
  %777 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %777, align 8, !dbg !27
  %778 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 2, !dbg !27
  store ptr @.str.22, ptr %778, align 8, !dbg !27
  %779 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 0, !dbg !27
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %779, i8 0, i64 16, i1 false), !dbg !27
  %780 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !27, !tbaa !30
  %781 = load ptr, ptr @.tvm_func.__tvm_error_null_ptr, align 8, !dbg !27
  %.not400 = icmp eq ptr %781, null, !dbg !27
  br i1 %.not400, label %handle_init270, label %handle_init_end271, !dbg !27, !prof !33

if_end269:                                        ; preds = %handle_init_end271, %if_end261
  %782 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 0, i32 1, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %stack_ffi_any375, align 8, !dbg !27
  %783 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 0, i32 2, !dbg !27
  store i64 2, ptr %783, align 8, !dbg !27
  %784 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 0, !dbg !27
  %785 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 1, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %784, align 8, !dbg !27
  %786 = sext i32 %dev_id to i64, !dbg !27
  %787 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 2, !dbg !27
  store i64 %786, ptr %787, align 8, !dbg !27
  %788 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 0, !dbg !27
  %789 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 1, !dbg !27
  %790 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 2, !dbg !27
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %788, i8 0, i64 16, i1 false), !dbg !27
  %791 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !27, !tbaa !30
  %792 = load ptr, ptr @.tvm_func.__tvm_set_device, align 8, !dbg !27
  %.not399 = icmp eq ptr %792, null, !dbg !27
  br i1 %.not399, label %handle_init276, label %handle_init_end277, !dbg !27, !prof !33

handle_init270:                                   ; preds = %if_then268
  %793 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !27, !tbaa !30
  %794 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !27, !tbaa !30
  %795 = call i32 %794(ptr %793, ptr nonnull @.str.23, ptr nonnull %2), !dbg !27
  %796 = icmp eq i32 %795, 0, !dbg !27
  br i1 %796, label %call_end273, label %common.ret, !dbg !27, !prof !29

handle_init_end271:                               ; preds = %call_end273, %if_then268
  %797 = phi ptr [ %781, %if_then268 ], [ %800, %call_end273 ], !dbg !27
  %798 = call i32 %780(ptr %797, ptr nonnull %stack_ffi_any375, i32 3, ptr nonnull %779), !dbg !27
  %799 = icmp eq i32 %798, 0, !dbg !27
  br i1 %799, label %if_end269, label %common.ret, !dbg !27, !prof !29

call_end273:                                      ; preds = %handle_init270
  %800 = load ptr, ptr %2, align 8, !dbg !27
  store ptr %800, ptr @.tvm_func.__tvm_error_null_ptr, align 8, !dbg !27
  br label %handle_init_end271, !dbg !27

handle_init276:                                   ; preds = %if_end269
  %801 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !27, !tbaa !30
  %802 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !27, !tbaa !30
  %803 = call i32 %802(ptr %801, ptr nonnull @.str.25, ptr nonnull %1), !dbg !27
  %804 = icmp eq i32 %803, 0, !dbg !27
  br i1 %804, label %call_end279, label %common.ret, !dbg !27, !prof !29

handle_init_end277:                               ; preds = %call_end279, %if_end269
  %805 = phi ptr [ %792, %if_end269 ], [ %808, %call_end279 ], !dbg !27
  %806 = call i32 %791(ptr %805, ptr nonnull %stack_ffi_any375, i32 2, ptr nonnull %788), !dbg !27
  %807 = icmp eq i32 %806, 0, !dbg !27
  br i1 %807, label %call_end281, label %common.ret, !dbg !27, !prof !29

call_end279:                                      ; preds = %handle_init276
  %808 = load ptr, ptr %1, align 8, !dbg !27
  store ptr %808, ptr @.tvm_func.__tvm_set_device, align 8, !dbg !27
  br label %handle_init_end277, !dbg !27

call_end281:                                      ; preds = %handle_init_end277
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %0), !dbg !15
  call void @llvm.dbg.value(metadata ptr %A, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.value(metadata ptr %stack_ffi_any375, metadata !23, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.value(metadata ptr %B, metadata !24, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.value(metadata ptr %C, metadata !25, metadata !DIExpression()), !dbg !15
  %..i = select i1 %305, i32 0, i32 4, !dbg !15
  store i32 %..i, ptr %stack_ffi_any375, align 8, !dbg !15
  store i32 0, ptr %782, align 4, !dbg !15
  store ptr %A, ptr %783, align 8, !dbg !15
  %spec.select.i = select i1 %535, i32 0, i32 4, !dbg !15
  store i32 %spec.select.i, ptr %784, align 8, !dbg !15
  store i32 0, ptr %785, align 4, !dbg !15
  store ptr %B, ptr %787, align 8, !dbg !15
  %.sink11.i = select i1 %765, i32 0, i32 4, !dbg !15
  store i32 %.sink11.i, ptr %788, align 8, !dbg !15
  store i32 0, ptr %789, align 4, !dbg !15
  store ptr %C, ptr %790, align 8, !dbg !15
  %809 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 0, !dbg !15
  store <2 x i32> <i32 1, i32 0>, ptr %809, align 8, !dbg !15
  %810 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 2, !dbg !15
  store i64 8, ptr %810, align 8, !dbg !15
  %811 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 0, !dbg !15
  store <2 x i32> <i32 1, i32 0>, ptr %811, align 8, !dbg !15
  %812 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 2, !dbg !15
  store i64 8, ptr %812, align 8, !dbg !15
  %813 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 5, i32 0, !dbg !15
  store <2 x i32> <i32 1, i32 0>, ptr %813, align 8, !dbg !15
  %814 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 5, i32 2, !dbg !15
  store i64 256, ptr %814, align 8, !dbg !15
  %815 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 6, i32 0, !dbg !15
  store <2 x i32> <i32 1, i32 0>, ptr %815, align 8, !dbg !15
  %816 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 6, i32 2, !dbg !15
  store i64 1, ptr %816, align 8, !dbg !15
  %817 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 7, i32 0, !dbg !15
  store <2 x i32> <i32 1, i32 0>, ptr %817, align 8, !dbg !15
  %818 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 7, i32 2, !dbg !15
  store i64 1, ptr %818, align 8, !dbg !15
  %819 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 8, i32 0, !dbg !15
  store <2 x i32> <i32 1, i32 0>, ptr %819, align 8, !dbg !15
  %820 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 8, i32 2, !dbg !15
  store i64 49152, ptr %820, align 8, !dbg !15
  %821 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 9, i32 0, !dbg !15
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %821, i8 0, i64 16, i1 false), !dbg !15
  %822 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !15, !tbaa !30
  %823 = load ptr, ptr @.tvm_func.gemm_kernel, align 8, !dbg !15
  %.not.i = icmp eq ptr %823, null, !dbg !15
  br i1 %.not.i, label %handle_init.i, label %handle_init_end.i, !dbg !15, !prof !33

handle_init.i:                                    ; preds = %call_end281
  %824 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !15, !tbaa !30
  %825 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !15, !tbaa !30
  %826 = call i32 %825(ptr %824, ptr nonnull @.str.26, ptr nonnull %0), !dbg !15
  %827 = icmp eq i32 %826, 0, !dbg !15
  br i1 %827, label %call_end.i, label %gemm_compute_.exit, !dbg !15, !prof !29

handle_init_end.i:                                ; preds = %call_end.i, %call_end281
  %828 = phi ptr [ %823, %call_end281 ], [ %830, %call_end.i ], !dbg !15
  %829 = call i32 %822(ptr %828, ptr nonnull %stack_ffi_any375, i32 9, ptr nonnull %821), !dbg !15
  br label %gemm_compute_.exit, !dbg !15

call_end.i:                                       ; preds = %handle_init.i
  %830 = load ptr, ptr %0, align 8, !dbg !15
  store ptr %830, ptr @.tvm_func.gemm_kernel, align 8, !dbg !15
  br label %handle_init_end.i, !dbg !15

gemm_compute_.exit:                               ; preds = %handle_init.i, %handle_init_end.i
  %common.ret.op.i = phi i32 [ %826, %handle_init.i ], [ %829, %handle_init_end.i ]
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %0), !dbg !15
  br label %common.ret
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: inaccessiblememonly mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.assume(i1 noundef) #2

define weak dllexport i32 @__tvm_ffi_main(ptr %0, ptr %1, i32 %2, ptr %3) local_unnamed_addr {
entry:
  %4 = tail call i32 @gemm(ptr poison, ptr %1, i32 %2, ptr poison), !dbg !27
  ret i32 %4, !dbg !27
}

; Function Attrs: nofree nosync nounwind readnone
define weak dso_local half @__truncsfhf2(float %a0) local_unnamed_addr #3 section ".text.tvm.fp16.conv" {
b0:
  %v0 = bitcast float %a0 to i32
  %v1 = and i32 %v0, 2147483647
  %v2 = add nsw i32 %v1, -947912704
  %v3 = add nsw i32 %v1, -1199570944
  %v4 = icmp ult i32 %v2, %v3
  br i1 %v4, label %b1, label %b5

b1:                                               ; preds = %b0
  %v5 = lshr i32 %v0, 13
  %v6 = and i32 %v5, 65535
  %v7 = add nuw nsw i32 %v6, -114688
  %v8 = and i32 %v0, 8191
  %v9 = icmp ugt i32 %v8, 4096
  br i1 %v9, label %b2, label %b3

b2:                                               ; preds = %b1
  %v10 = add nuw nsw i32 %v6, -114687
  br label %b13

b3:                                               ; preds = %b1
  %v11 = icmp eq i32 %v8, 4096
  br i1 %v11, label %b4, label %b13

b4:                                               ; preds = %b3
  %v12 = and i32 %v7, 65535
  %v13 = and i32 %v5, 1
  %v14 = add nuw nsw i32 %v12, %v13
  br label %b13

b5:                                               ; preds = %b0
  %v15 = icmp ugt i32 %v1, 2139095040
  br i1 %v15, label %b6, label %b7

b6:                                               ; preds = %b5
  %v16 = lshr i32 %v0, 13
  %v17 = and i32 %v16, 511
  %v18 = or i32 %v17, 32256
  br label %b13

b7:                                               ; preds = %b5
  %v19 = icmp ugt i32 %v1, 1199570943
  br i1 %v19, label %b13, label %b8

b8:                                               ; preds = %b7
  %v20 = icmp ult i32 %v1, 754974720
  br i1 %v20, label %b13, label %b9

b9:                                               ; preds = %b8
  %v21 = lshr i32 %v1, 23
  %v22 = sub nsw i32 113, %v21
  %v23 = and i32 %v0, 8388607
  %v24 = or i32 %v23, 8388608
  %v25 = add nsw i32 %v21, -81
  %v26 = shl i32 %v24, %v25
  %v27 = icmp ne i32 %v26, 0
  %v28 = lshr i32 %v24, %v22
  %v29 = zext i1 %v27 to i32
  %v30 = lshr i32 %v28, 13
  %v31 = and i32 %v28, 8191
  %v32 = or i32 %v31, %v29
  %v33 = icmp ugt i32 %v32, 4096
  br i1 %v33, label %b10, label %b11

b10:                                              ; preds = %b9
  %v34 = add nuw nsw i32 %v30, 1
  br label %b13

b11:                                              ; preds = %b9
  %v35 = icmp eq i32 %v32, 4096
  br i1 %v35, label %b12, label %b13

b12:                                              ; preds = %b11
  %v36 = and i32 %v30, 1
  %v37 = add nuw nsw i32 %v36, %v30
  br label %b13

b13:                                              ; preds = %b12, %b11, %b10, %b8, %b7, %b6, %b4, %b3, %b2
  %v38 = phi i32 [ %v18, %b6 ], [ %v10, %b2 ], [ %v14, %b4 ], [ %v7, %b3 ], [ 31744, %b7 ], [ 0, %b8 ], [ %v34, %b10 ], [ %v37, %b12 ], [ %v30, %b11 ]
  %v39 = lshr i32 %v0, 16
  %v40 = and i32 %v39, 32768
  %v41 = or i32 %v38, %v40
  %vlast = trunc i32 %v41 to i16
  %vres = bitcast i16 %vlast to half
  ret half %vres
}

; Function Attrs: nofree nosync nounwind readnone
define weak dso_local float @__extendhfsf2(half %a0) local_unnamed_addr #3 section ".text.tvm.fp16.conv" {
b0:
  %vinp = bitcast half %a0 to i16
  %v1 = and i16 %vinp, 32767
  %v2 = zext i16 %v1 to i32
  %v3 = add nsw i16 %v1, -1024
  %v4 = icmp ult i16 %v3, 30720
  br i1 %v4, label %b1, label %b2

b1:                                               ; preds = %b0
  %v5 = shl nuw nsw i32 %v2, 13
  %v6 = add nuw nsw i32 %v5, 939524096
  br label %b6

b2:                                               ; preds = %b0
  %v7 = icmp ugt i16 %v1, 31743
  br i1 %v7, label %b3, label %b4

b3:                                               ; preds = %b2
  %v8 = shl nuw nsw i32 %v2, 13
  %v9 = or i32 %v8, 2139095040
  br label %b6

b4:                                               ; preds = %b2
  %v10 = icmp eq i16 %v1, 0
  br i1 %v10, label %b6, label %b5

b5:                                               ; preds = %b4
  %v11 = icmp ult i16 %v1, 256
  %v12 = lshr i32 %v2, 8
  %v13 = select i1 %v11, i32 %v2, i32 %v12
  %v14 = select i1 %v11, i32 32, i32 24
  %v15 = icmp ult i32 %v13, 16
  %v16 = lshr i32 %v13, 4
  %v17 = add nsw i32 %v14, -4
  %v18 = select i1 %v15, i32 %v13, i32 %v16
  %v19 = select i1 %v15, i32 %v14, i32 %v17
  %v20 = icmp ult i32 %v18, 4
  %v21 = lshr i32 %v18, 2
  %v22 = add nsw i32 %v19, -2
  %v23 = select i1 %v20, i32 %v18, i32 %v21
  %v24 = select i1 %v20, i32 %v19, i32 %v22
  %v25 = icmp ult i32 %v23, 2
  %v26 = sub nsw i32 0, %v23
  %v27 = select i1 %v25, i32 %v26, i32 -2
  %v28 = add nsw i32 %v27, %v24
  %v29 = add nsw i32 %v28, -8
  %v30 = shl i32 %v2, %v29
  %v31 = xor i32 %v30, 8388608
  %v32.neg = mul i32 %v28, -8388608
  %v33 = add i32 %v32.neg, 1124073472
  %v34 = or i32 %v31, %v33
  br label %b6

b6:                                               ; preds = %b5, %b4, %b3, %b1
  %v35 = phi i32 [ %v6, %b1 ], [ %v9, %b3 ], [ %v34, %b5 ], [ 0, %b4 ]
  %v36 = and i16 %vinp, -32768
  %v37 = zext i16 %v36 to i32
  %v38 = shl nuw i32 %v37, 16
  %v39 = or i32 %v35, %v38
  %v40 = bitcast i32 %v39 to float
  ret float %v40
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

; Function Attrs: argmemonly nocallback nofree nounwind willreturn writeonly
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg) #4

; Function Attrs: argmemonly nocallback nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #5

; Function Attrs: argmemonly nocallback nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #5

attributes #0 = { "target-cpu"="generic" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind readnone speculatable willreturn }
attributes #2 = { inaccessiblememonly mustprogress nocallback nofree nosync nounwind willreturn }
attributes #3 = { nofree nosync nounwind readnone "target-cpu"="generic" "target-features" }
attributes #4 = { argmemonly nocallback nofree nounwind willreturn writeonly }
attributes #5 = { argmemonly nocallback nofree nosync nounwind willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "TVM", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "IRModule.CodeGenLLVM", directory: ".")
!2 = !{i32 2, !"tvm_target", !"llvm -mtriple=x86_64-pc-linux-gnu"}
!3 = !{i32 4, !"Debug Info Version", i32 3}
!4 = !{i32 4, !"Dwarf Version", i32 4}
!5 = distinct !DISubprogram(name: "gemm", scope: !1, file: !1, type: !6, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !10)
!6 = !DISubroutineType(types: !7)
!7 = !{!8, !9, !9, !8, !9}
!8 = !DIBasicType(name: "int32", size: 32, encoding: DW_ATE_signed)
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null)
!10 = !{!11, !12, !13, !14}
!11 = !DILocalVariable(name: "self_handle", arg: 1, scope: !5, file: !1, type: !9)
!12 = !DILocalVariable(name: "args", arg: 2, scope: !5, file: !1, type: !9)
!13 = !DILocalVariable(name: "num_args", arg: 3, scope: !5, file: !1, type: !8)
!14 = !DILocalVariable(name: "result", arg: 4, scope: !5, file: !1, type: !9)
!15 = !DILocation(line: 0, scope: !16, inlinedAt: !26)
!16 = distinct !DISubprogram(name: "gemm_compute_", scope: !1, file: !1, type: !17, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !21)
!17 = !DISubroutineType(types: !18)
!18 = !{!8, !19, !9, !19, !19}
!19 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !20)
!20 = !DIBasicType(name: "uint16", size: 16, encoding: DW_ATE_unsigned)
!21 = !{!22, !23, !24, !25}
!22 = !DILocalVariable(name: "A", arg: 1, scope: !16, file: !1, type: !19)
!23 = !DILocalVariable(name: "stack_ffi_any", arg: 2, scope: !16, file: !1, type: !9)
!24 = !DILocalVariable(name: "B", arg: 3, scope: !16, file: !1, type: !19)
!25 = !DILocalVariable(name: "C", arg: 4, scope: !16, file: !1, type: !19)
!26 = distinct !DILocation(line: 0, scope: !5)
!27 = !DILocation(line: 0, scope: !5)
!28 = !DILocalVariable(name: "stack_ffi_any", scope: !5, file: !1, type: !9)
!29 = !{!"branch_weights", i32 1048576, i32 1}
!30 = !{!31, !31, i64 0}
!31 = !{!"ctx_ptr", !32, i64 0}
!32 = !{!"tvm-tbaa"}
!33 = !{!"branch_weights", i32 1, i32 1048576}
!34 = !DILocalVariable(name: "A_handle.type_index", scope: !5, file: !1, type: !8)
!35 = !DILocalVariable(name: "B_handle.type_index", scope: !5, file: !1, type: !8)
!36 = !DILocalVariable(name: "C_handle.type_index", scope: !5, file: !1, type: !8)
!37 = !DILocalVariable(name: "A_handle", scope: !5, file: !1, type: !9)
!38 = !DILocalVariable(name: "B_handle", scope: !5, file: !1, type: !9)
!39 = !DILocalVariable(name: "C_handle", scope: !5, file: !1, type: !9)
!40 = !DILocalVariable(name: "gemm.A_is_null", scope: !5, file: !1, type: !41)
!41 = !DIBasicType(name: "bool", size: 8, encoding: DW_ATE_boolean)
!42 = !DILocalVariable(name: "gemm.B_is_null", scope: !5, file: !1, type: !41)
!43 = !DILocalVariable(name: "gemm.C_is_null", scope: !5, file: !1, type: !41)
!44 = !DILocalVariable(name: "gemm.A.shape", scope: !5, file: !1, type: !45)
!45 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !46)
!46 = !DIBasicType(name: "int64", size: 64, encoding: DW_ATE_signed)
!47 = !DILocalVariable(name: "gemm.B.shape", scope: !5, file: !1, type: !45)
!48 = !DILocalVariable(name: "gemm.C.shape", scope: !5, file: !1, type: !45)
!49 = !DILocalVariable(name: "gemm.A.strides", scope: !5, file: !1, type: !45)
!50 = !DILocalVariable(name: "dev_id", scope: !5, file: !1, type: !8)
!51 = !DILocalVariable(name: "A", scope: !5, file: !1, type: !19)
!52 = !DILocalVariable(name: "gemm.B.strides", scope: !5, file: !1, type: !45)
!53 = !DILocalVariable(name: "B", scope: !5, file: !1, type: !19)
!54 = !DILocalVariable(name: "gemm.C.strides", scope: !5, file: !1, type: !45)
!55 = !DILocalVariable(name: "C", scope: !5, file: !1, type: !19)
!56 = !{!57, !57, i64 0}
!57 = !{!"tvm-alias", !32}
