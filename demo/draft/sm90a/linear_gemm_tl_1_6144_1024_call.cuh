
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
@.str = private constant [57 x i8] c"Assert fail: num_args == 3, linear: num_args should be 3\00", align 1
@.str.1 = private constant [13 x i8] c"RuntimeError\00", align 1
@.str.2 = private constant [65 x i8] c"Assert fail: not T.isnullptr(args), linear: args pointer is NULL\00", align 1
@.str.3 = private constant [180 x i8] c"Assert fail: A_handle_type_index == 0 or A_handle_type_index == 4 or A_handle_type_index == 7 or 64 <= A_handle_type_index, kernel linear input A expected pointer or tensor handle\00", align 1
@.str.4 = private constant [180 x i8] c"Assert fail: B_handle_type_index == 0 or B_handle_type_index == 4 or B_handle_type_index == 7 or 64 <= B_handle_type_index, kernel linear input B expected pointer or tensor handle\00", align 1
@.str.5 = private constant [180 x i8] c"Assert fail: C_handle_type_index == 0 or C_handle_type_index == 4 or C_handle_type_index == 7 or 64 <= C_handle_type_index, kernel linear input C expected pointer or tensor handle\00", align 1
@.str.6 = private constant [81 x i8] c"Assert fail: not linear_A_is_null, linear.A is expected to have non-NULL pointer\00", align 1
@.str.7 = private constant [81 x i8] c"Assert fail: not linear_B_is_null, linear.B is expected to have non-NULL pointer\00", align 1
@.str.8 = private constant [81 x i8] c"Assert fail: not linear_C_is_null, linear.C is expected to have non-NULL pointer\00", align 1
@.str.9 = private constant [7 x i8] c"linear\00", align 1
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
@.tvm_func.__tvm_tensormap_create_tiled = internal unnamed_addr global ptr null, align 8
@.str.26 = private constant [29 x i8] c"__tvm_tensormap_create_tiled\00", align 1
@.tvm_func.linear_kernel = internal unnamed_addr global ptr null, align 8
@.str.27 = private constant [14 x i8] c"linear_kernel\00", align 1
@llvm.global_ctors = appending global [0 x { i32, ptr, ptr }] zeroinitializer

define dllexport i32 @linear(ptr nocapture readnone %self_handle, ptr readonly %args, i32 %num_args, ptr nocapture readnone %result) local_unnamed_addr #0 !dbg !5 {
entry:
  %0 = alloca ptr, align 8, !dbg !15
  %1 = alloca ptr, align 8, !dbg !15
  %A_desc56.i = alloca [16 x %0], align 8, !dbg !15
  %2 = alloca ptr, align 8, !dbg !15
  %B_desc57.i = alloca [16 x %0], align 8, !dbg !15
  %3 = alloca ptr, align 8, !dbg !15
  %C_desc58.i = alloca [16 x %0], align 8, !dbg !15
  call void @llvm.dbg.value(metadata ptr poison, metadata !11, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.value(metadata ptr %args, metadata !12, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.value(metadata i32 %num_args, metadata !13, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.value(metadata ptr poison, metadata !14, metadata !DIExpression()), !dbg !27
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
  %31 = alloca ptr, align 8, !dbg !27
  %32 = alloca ptr, align 8, !dbg !27
  %33 = alloca ptr, align 8, !dbg !27
  %stack_ffi_any375 = alloca [17 x %0], align 8, !dbg !27
  call void @llvm.dbg.declare(metadata ptr %stack_ffi_any375, metadata !28, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.declare(metadata ptr %stack_ffi_any375, metadata !28, metadata !DIExpression()), !dbg !27
  %34 = icmp eq i32 %num_args, 3, !dbg !27
  br i1 %34, label %assert_end, label %assert_fail, !dbg !27, !prof !29

common.ret:                                       ; preds = %linear_compute_.exit, %handle_init_end277, %handle_init276, %handle_init_end271, %handle_init270, %handle_init_end263, %handle_init262, %handle_init_end255, %handle_init254, %handle_init_end247, %handle_init246, %handle_init_end239, %handle_init238, %handle_init_end225, %handle_init224, %handle_init_end211, %handle_init210, %handle_init_end203, %handle_init202, %handle_init_end195, %handle_init194, %handle_init_end187, %handle_init186, %handle_init_end179, %handle_init178, %handle_init_end171, %handle_init170, %handle_init_end163, %handle_init162, %handle_init_end155, %handle_init154, %handle_init_end141, %handle_init140, %handle_init_end127, %handle_init126, %handle_init_end119, %handle_init118, %handle_init_end111, %handle_init110, %handle_init_end103, %handle_init102, %handle_init_end95, %handle_init94, %handle_init_end87, %handle_init86, %handle_init_end79, %handle_init78, %handle_init_end65, %handle_init64, %handle_init_end52, %handle_init51, %handle_init_end44, %handle_init43, %handle_init_end36, %handle_init35, %handle_init_end28, %handle_init27, %handle_init_end20, %handle_init19, %handle_init_end, %handle_init, %assert_fail13, %assert_fail11, %assert_fail9, %assert_fail7, %assert_fail5, %assert_fail3, %assert_fail1, %assert_fail
  %common.ret.op = phi i32 [ -1, %assert_fail ], [ -1, %assert_fail1 ], [ -1, %assert_fail3 ], [ -1, %assert_fail5 ], [ -1, %assert_fail7 ], [ -1, %assert_fail9 ], [ -1, %assert_fail11 ], [ -1, %assert_fail13 ], [ %80, %handle_init ], [ %83, %handle_init_end ], [ %103, %handle_init19 ], [ %106, %handle_init_end20 ], [ %135, %handle_init27 ], [ %138, %handle_init_end28 ], [ %169, %handle_init35 ], [ %172, %handle_init_end36 ], [ %194, %handle_init43 ], [ %197, %handle_init_end44 ], [ %217, %handle_init51 ], [ %220, %handle_init_end52 ], [ %242, %handle_init64 ], [ %245, %handle_init_end65 ], [ %269, %handle_init78 ], [ %272, %handle_init_end79 ], [ %290, %handle_init86 ], [ %293, %handle_init_end87 ], [ %311, %handle_init94 ], [ %314, %handle_init_end95 ], [ %338, %handle_init102 ], [ %341, %handle_init_end103 ], [ %372, %handle_init110 ], [ %375, %handle_init_end111 ], [ %397, %handle_init118 ], [ %400, %handle_init_end119 ], [ %420, %handle_init126 ], [ %423, %handle_init_end127 ], [ %445, %handle_init140 ], [ %448, %handle_init_end141 ], [ %472, %handle_init154 ], [ %475, %handle_init_end155 ], [ %494, %handle_init162 ], [ %497, %handle_init_end163 ], [ %520, %handle_init170 ], [ %523, %handle_init_end171 ], [ %541, %handle_init178 ], [ %544, %handle_init_end179 ], [ %568, %handle_init186 ], [ %571, %handle_init_end187 ], [ %602, %handle_init194 ], [ %605, %handle_init_end195 ], [ %627, %handle_init202 ], [ %630, %handle_init_end203 ], [ %650, %handle_init210 ], [ %653, %handle_init_end211 ], [ %675, %handle_init224 ], [ %678, %handle_init_end225 ], [ %702, %handle_init238 ], [ %705, %handle_init_end239 ], [ %724, %handle_init246 ], [ %727, %handle_init_end247 ], [ %750, %handle_init254 ], [ %753, %handle_init_end255 ], [ %771, %handle_init262 ], [ %774, %handle_init_end263 ], [ %795, %handle_init270 ], [ %798, %handle_init_end271 ], [ %803, %handle_init276 ], [ %806, %handle_init_end277 ], [ %common.ret.op.i, %linear_compute_.exit ]
  ret i32 %common.ret.op, !dbg !27

assert_fail:                                      ; preds = %entry
  %35 = load ptr, ptr @__TVMFFIErrorSetRaisedFromCStr, align 8, !dbg !27, !tbaa !30
  tail call void %35(ptr nonnull @.str.1, ptr nonnull @.str), !dbg !27
  br label %common.ret

assert_end:                                       ; preds = %entry
  %.not = icmp eq ptr %args, null, !dbg !27
  br i1 %.not, label %assert_fail1, label %assert_end2, !dbg !27, !prof !33

assert_fail1:                                     ; preds = %assert_end
  %36 = load ptr, ptr @__TVMFFIErrorSetRaisedFromCStr, align 8, !dbg !27, !tbaa !30
  tail call void %36(ptr nonnull @.str.1, ptr nonnull @.str.2), !dbg !27
  br label %common.ret

assert_end2:                                      ; preds = %assert_end
  %A_handle.type_index = load i32, ptr %args, align 4, !dbg !27
  %A_handle.type_index.fr = freeze i32 %A_handle.type_index, !dbg !27
  call void @llvm.dbg.declare(metadata i32 %A_handle.type_index, metadata !34, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.declare(metadata i32 %A_handle.type_index, metadata !34, metadata !DIExpression()), !dbg !27
  %37 = icmp sgt i32 %A_handle.type_index.fr, 63, !dbg !27
  br i1 %37, label %assert_end4, label %switch.early.test, !dbg !27

switch.early.test:                                ; preds = %assert_end2
  switch i32 %A_handle.type_index.fr, label %assert_fail3 [
    i32 7, label %assert_end4
    i32 4, label %assert_end4
    i32 0, label %assert_end4
  ], !dbg !27

assert_fail3:                                     ; preds = %switch.early.test
  %38 = load ptr, ptr @__TVMFFIErrorSetRaisedFromCStr, align 8, !dbg !27, !tbaa !30
  tail call void %38(ptr nonnull @.str.1, ptr nonnull @.str.3), !dbg !27
  br label %common.ret

assert_end4:                                      ; preds = %switch.early.test, %switch.early.test, %switch.early.test, %assert_end2
  %39 = getelementptr inbounds %0, ptr %args, i64 1, i32 0, !dbg !27
  %B_handle.type_index = load i32, ptr %39, align 4, !dbg !27
  %B_handle.type_index.fr = freeze i32 %B_handle.type_index, !dbg !27
  call void @llvm.dbg.declare(metadata i32 %B_handle.type_index, metadata !35, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.declare(metadata i32 %B_handle.type_index, metadata !35, metadata !DIExpression()), !dbg !27
  %40 = icmp sgt i32 %B_handle.type_index.fr, 63, !dbg !27
  br i1 %40, label %assert_end6, label %switch.early.test284, !dbg !27

switch.early.test284:                             ; preds = %assert_end4
  switch i32 %B_handle.type_index.fr, label %assert_fail5 [
    i32 7, label %assert_end6
    i32 4, label %assert_end6
    i32 0, label %assert_end6
  ], !dbg !27

assert_fail5:                                     ; preds = %switch.early.test284
  %41 = load ptr, ptr @__TVMFFIErrorSetRaisedFromCStr, align 8, !dbg !27, !tbaa !30
  tail call void %41(ptr nonnull @.str.1, ptr nonnull @.str.4), !dbg !27
  br label %common.ret

assert_end6:                                      ; preds = %switch.early.test284, %switch.early.test284, %switch.early.test284, %assert_end4
  %42 = getelementptr inbounds %0, ptr %args, i64 2, i32 0, !dbg !27
  %C_handle.type_index = load i32, ptr %42, align 4, !dbg !27
  %C_handle.type_index.fr = freeze i32 %C_handle.type_index, !dbg !27
  call void @llvm.dbg.declare(metadata i32 %C_handle.type_index, metadata !36, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.declare(metadata i32 %C_handle.type_index, metadata !36, metadata !DIExpression()), !dbg !27
  %43 = icmp sgt i32 %C_handle.type_index.fr, 63, !dbg !27
  br i1 %43, label %assert_end8, label %switch.early.test285, !dbg !27

switch.early.test285:                             ; preds = %assert_end6
  switch i32 %C_handle.type_index.fr, label %assert_fail7 [
    i32 7, label %assert_end8
    i32 4, label %assert_end8
    i32 0, label %assert_end8
  ], !dbg !27

assert_fail7:                                     ; preds = %switch.early.test285
  %44 = load ptr, ptr @__TVMFFIErrorSetRaisedFromCStr, align 8, !dbg !27, !tbaa !30
  tail call void %44(ptr nonnull @.str.1, ptr nonnull @.str.5), !dbg !27
  br label %common.ret

assert_end8:                                      ; preds = %switch.early.test285, %switch.early.test285, %switch.early.test285, %assert_end6
  %45 = getelementptr inbounds %0, ptr %args, i64 0, i32 2, !dbg !27
  %46 = load ptr, ptr %45, align 8, !dbg !27
  %47 = icmp eq i32 %A_handle.type_index.fr, 70, !dbg !27
  %A_handle.idx = select i1 %47, i64 24, i64 0, !dbg !27
  %A_handle = getelementptr i8, ptr %46, i64 %A_handle.idx, !dbg !27
  call void @llvm.dbg.declare(metadata ptr %A_handle, metadata !37, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.declare(metadata ptr %A_handle, metadata !37, metadata !DIExpression()), !dbg !27
  %48 = getelementptr inbounds %0, ptr %args, i64 1, i32 2, !dbg !27
  %49 = load ptr, ptr %48, align 8, !dbg !27
  %50 = icmp eq i32 %B_handle.type_index.fr, 70, !dbg !27
  %B_handle.idx = select i1 %50, i64 24, i64 0, !dbg !27
  %B_handle = getelementptr i8, ptr %49, i64 %B_handle.idx, !dbg !27
  call void @llvm.dbg.declare(metadata ptr %B_handle, metadata !38, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.declare(metadata ptr %B_handle, metadata !38, metadata !DIExpression()), !dbg !27
  %51 = getelementptr inbounds %0, ptr %args, i64 2, i32 2, !dbg !27
  %52 = load ptr, ptr %51, align 8, !dbg !27
  %53 = icmp eq i32 %C_handle.type_index.fr, 70, !dbg !27
  %C_handle.idx = select i1 %53, i64 24, i64 0, !dbg !27
  %C_handle = getelementptr i8, ptr %52, i64 %C_handle.idx, !dbg !27
  call void @llvm.dbg.declare(metadata ptr %C_handle, metadata !39, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.declare(metadata ptr %C_handle, metadata !39, metadata !DIExpression()), !dbg !27
  %linear.A_is_null.not = icmp eq ptr %A_handle, null, !dbg !27
  call void @llvm.dbg.declare(metadata i1 %linear.A_is_null.not, metadata !40, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.declare(metadata i1 %linear.A_is_null.not, metadata !40, metadata !DIExpression()), !dbg !27
  br i1 %linear.A_is_null.not, label %assert_fail9, label %assert_end10, !dbg !27, !prof !33

assert_fail9:                                     ; preds = %assert_end8
  %54 = load ptr, ptr @__TVMFFIErrorSetRaisedFromCStr, align 8, !dbg !27, !tbaa !30
  tail call void %54(ptr nonnull @.str.1, ptr nonnull @.str.6), !dbg !27
  br label %common.ret

assert_end10:                                     ; preds = %assert_end8
  %linear.B_is_null.not = icmp eq ptr %B_handle, null, !dbg !27
  call void @llvm.dbg.declare(metadata i1 %linear.B_is_null.not, metadata !42, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.declare(metadata i1 %linear.B_is_null.not, metadata !42, metadata !DIExpression()), !dbg !27
  br i1 %linear.B_is_null.not, label %assert_fail11, label %assert_end12, !dbg !27, !prof !33

assert_fail11:                                    ; preds = %assert_end10
  %55 = load ptr, ptr @__TVMFFIErrorSetRaisedFromCStr, align 8, !dbg !27, !tbaa !30
  tail call void %55(ptr nonnull @.str.1, ptr nonnull @.str.7), !dbg !27
  br label %common.ret

assert_end12:                                     ; preds = %assert_end10
  %linear.C_is_null.not = icmp eq ptr %C_handle, null, !dbg !27
  call void @llvm.dbg.declare(metadata i1 %linear.C_is_null.not, metadata !43, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.declare(metadata i1 %linear.C_is_null.not, metadata !43, metadata !DIExpression()), !dbg !27
  br i1 %linear.C_is_null.not, label %assert_fail13, label %assert_end14, !dbg !27, !prof !33

assert_fail13:                                    ; preds = %assert_end12
  %56 = load ptr, ptr @__TVMFFIErrorSetRaisedFromCStr, align 8, !dbg !27, !tbaa !30
  tail call void %56(ptr nonnull @.str.1, ptr nonnull @.str.8), !dbg !27
  br label %common.ret

assert_end14:                                     ; preds = %assert_end12
  %57 = getelementptr inbounds %1, ptr %A_handle, i64 0, i32 4, !dbg !27
  %linear.A.shape = load ptr, ptr %57, align 8, !dbg !27
  call void @llvm.dbg.declare(metadata ptr %linear.A.shape, metadata !44, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.declare(metadata ptr %linear.A.shape, metadata !44, metadata !DIExpression()), !dbg !27
  %58 = getelementptr inbounds %1, ptr %B_handle, i64 0, i32 4, !dbg !27
  %linear.B.shape = load ptr, ptr %58, align 8, !dbg !27
  call void @llvm.dbg.declare(metadata ptr %linear.B.shape, metadata !47, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.declare(metadata ptr %linear.B.shape, metadata !47, metadata !DIExpression()), !dbg !27
  %59 = getelementptr inbounds %1, ptr %C_handle, i64 0, i32 4, !dbg !27
  %linear.C.shape = load ptr, ptr %59, align 8, !dbg !27
  call void @llvm.dbg.declare(metadata ptr %linear.C.shape, metadata !48, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.declare(metadata ptr %linear.C.shape, metadata !48, metadata !DIExpression()), !dbg !27
  %60 = getelementptr inbounds %1, ptr %A_handle, i64 0, i32 2, !dbg !27
  %61 = load i32, ptr %60, align 4, !dbg !27
  %.not376 = icmp eq i32 %61, 2, !dbg !27
  br i1 %.not376, label %if_end, label %if_then, !dbg !27, !prof !33

if_then:                                          ; preds = %assert_end14
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any375, align 8, !dbg !27
  %62 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 0, i32 2, !dbg !27
  store ptr @.str.9, ptr %62, align 8, !dbg !27
  %63 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %63, align 8, !dbg !27
  %64 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 2, !dbg !27
  store ptr @.str.10, ptr %64, align 8, !dbg !27
  %65 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %65, align 8, !dbg !27
  %66 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 2, !dbg !27
  store i64 2, ptr %66, align 8, !dbg !27
  %67 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %67, align 8, !dbg !27
  %68 = load i32, ptr %60, align 4, !dbg !27
  %69 = sext i32 %68 to i64, !dbg !27
  %70 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 2, !dbg !27
  store i64 %69, ptr %70, align 8, !dbg !27
  %71 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 0, !dbg !27
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %71, i8 0, i64 16, i1 false), !dbg !27
  %72 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !27, !tbaa !30
  %73 = load ptr, ptr @.tvm_func.__tvm_error_ndim_mismatch, align 8, !dbg !27
  %.not439 = icmp eq ptr %73, null, !dbg !27
  br i1 %.not439, label %handle_init, label %handle_init_end, !dbg !27, !prof !33

if_end:                                           ; preds = %handle_init_end, %assert_end14
  %74 = getelementptr inbounds %1, ptr %A_handle, i64 0, i32 5, !dbg !27
  %linear.A.strides = load ptr, ptr %74, align 8, !dbg !27
  call void @llvm.dbg.declare(metadata ptr %linear.A.strides, metadata !49, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.declare(metadata ptr %linear.A.strides, metadata !49, metadata !DIExpression()), !dbg !27
  %75 = getelementptr inbounds %1, ptr %A_handle, i64 0, i32 1, i32 1, !dbg !27
  %dev_id = load i32, ptr %75, align 4, !dbg !27
  call void @llvm.dbg.declare(metadata i32 %dev_id, metadata !50, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.declare(metadata i32 %dev_id, metadata !50, metadata !DIExpression()), !dbg !27
  %A = load ptr, ptr %A_handle, align 8, !dbg !27
  call void @llvm.dbg.declare(metadata ptr %A, metadata !51, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.declare(metadata ptr %A, metadata !51, metadata !DIExpression()), !dbg !27
  call void @llvm.assume(i1 true) [ "align"(ptr %A, i64 64) ], !dbg !27
  %76 = getelementptr inbounds %1, ptr %B_handle, i64 0, i32 2, !dbg !27
  %77 = load i32, ptr %76, align 4, !dbg !27
  %.not377 = icmp eq i32 %77, 2, !dbg !27
  br i1 %.not377, label %if_end18, label %if_then17, !dbg !27, !prof !33

handle_init:                                      ; preds = %if_then
  %78 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !27, !tbaa !30
  %79 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !27, !tbaa !30
  %80 = call i32 %79(ptr %78, ptr nonnull @.str.11, ptr nonnull %33), !dbg !27
  %81 = icmp eq i32 %80, 0, !dbg !27
  br i1 %81, label %call_end, label %common.ret, !dbg !27, !prof !29

handle_init_end:                                  ; preds = %call_end, %if_then
  %82 = phi ptr [ %73, %if_then ], [ %85, %call_end ], !dbg !27
  %83 = call i32 %72(ptr %82, ptr nonnull %stack_ffi_any375, i32 4, ptr nonnull %71), !dbg !27
  %84 = icmp eq i32 %83, 0, !dbg !27
  br i1 %84, label %if_end, label %common.ret, !dbg !27, !prof !29

call_end:                                         ; preds = %handle_init
  %85 = load ptr, ptr %33, align 8, !dbg !27
  store ptr %85, ptr @.tvm_func.__tvm_error_ndim_mismatch, align 8, !dbg !27
  br label %handle_init_end, !dbg !27

if_then17:                                        ; preds = %if_end
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any375, align 8, !dbg !27
  %86 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 0, i32 2, !dbg !27
  store ptr @.str.9, ptr %86, align 8, !dbg !27
  %87 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %87, align 8, !dbg !27
  %88 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 2, !dbg !27
  store ptr @.str.12, ptr %88, align 8, !dbg !27
  %89 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %89, align 8, !dbg !27
  %90 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 2, !dbg !27
  store i64 2, ptr %90, align 8, !dbg !27
  %91 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %91, align 8, !dbg !27
  %92 = load i32, ptr %76, align 4, !dbg !27
  %93 = sext i32 %92 to i64, !dbg !27
  %94 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 2, !dbg !27
  store i64 %93, ptr %94, align 8, !dbg !27
  %95 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 0, !dbg !27
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %95, i8 0, i64 16, i1 false), !dbg !27
  %96 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !27, !tbaa !30
  %97 = load ptr, ptr @.tvm_func.__tvm_error_ndim_mismatch, align 8, !dbg !27
  %.not438 = icmp eq ptr %97, null, !dbg !27
  br i1 %.not438, label %handle_init19, label %handle_init_end20, !dbg !27, !prof !33

if_end18:                                         ; preds = %handle_init_end20, %if_end
  %98 = getelementptr inbounds %1, ptr %B_handle, i64 0, i32 5, !dbg !27
  %linear.B.strides = load ptr, ptr %98, align 8, !dbg !27
  call void @llvm.dbg.declare(metadata ptr %linear.B.strides, metadata !52, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.declare(metadata ptr %linear.B.strides, metadata !52, metadata !DIExpression()), !dbg !27
  %B = load ptr, ptr %B_handle, align 8, !dbg !27
  call void @llvm.dbg.declare(metadata ptr %B, metadata !53, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.declare(metadata ptr %B, metadata !53, metadata !DIExpression()), !dbg !27
  call void @llvm.assume(i1 true) [ "align"(ptr %B, i64 64) ], !dbg !27
  %99 = getelementptr inbounds %1, ptr %C_handle, i64 0, i32 2, !dbg !27
  %100 = load i32, ptr %99, align 4, !dbg !27
  %.not378 = icmp eq i32 %100, 2, !dbg !27
  br i1 %.not378, label %if_end26, label %if_then25, !dbg !27, !prof !33

handle_init19:                                    ; preds = %if_then17
  %101 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !27, !tbaa !30
  %102 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !27, !tbaa !30
  %103 = call i32 %102(ptr %101, ptr nonnull @.str.11, ptr nonnull %32), !dbg !27
  %104 = icmp eq i32 %103, 0, !dbg !27
  br i1 %104, label %call_end22, label %common.ret, !dbg !27, !prof !29

handle_init_end20:                                ; preds = %call_end22, %if_then17
  %105 = phi ptr [ %97, %if_then17 ], [ %108, %call_end22 ], !dbg !27
  %106 = call i32 %96(ptr %105, ptr nonnull %stack_ffi_any375, i32 4, ptr nonnull %95), !dbg !27
  %107 = icmp eq i32 %106, 0, !dbg !27
  br i1 %107, label %if_end18, label %common.ret, !dbg !27, !prof !29

call_end22:                                       ; preds = %handle_init19
  %108 = load ptr, ptr %32, align 8, !dbg !27
  store ptr %108, ptr @.tvm_func.__tvm_error_ndim_mismatch, align 8, !dbg !27
  br label %handle_init_end20, !dbg !27

if_then25:                                        ; preds = %if_end18
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any375, align 8, !dbg !27
  %109 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 0, i32 2, !dbg !27
  store ptr @.str.9, ptr %109, align 8, !dbg !27
  %110 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %110, align 8, !dbg !27
  %111 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 2, !dbg !27
  store ptr @.str.13, ptr %111, align 8, !dbg !27
  %112 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %112, align 8, !dbg !27
  %113 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 2, !dbg !27
  store i64 2, ptr %113, align 8, !dbg !27
  %114 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %114, align 8, !dbg !27
  %115 = load i32, ptr %99, align 4, !dbg !27
  %116 = sext i32 %115 to i64, !dbg !27
  %117 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 2, !dbg !27
  store i64 %116, ptr %117, align 8, !dbg !27
  %118 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 0, !dbg !27
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %118, i8 0, i64 16, i1 false), !dbg !27
  %119 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !27, !tbaa !30
  %120 = load ptr, ptr @.tvm_func.__tvm_error_ndim_mismatch, align 8, !dbg !27
  %.not437 = icmp eq ptr %120, null, !dbg !27
  br i1 %.not437, label %handle_init27, label %handle_init_end28, !dbg !27, !prof !33

if_end26:                                         ; preds = %handle_init_end28, %if_end18
  %121 = getelementptr inbounds %1, ptr %C_handle, i64 0, i32 5, !dbg !27
  %linear.C.strides = load ptr, ptr %121, align 8, !dbg !27
  call void @llvm.dbg.declare(metadata ptr %linear.C.strides, metadata !54, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.declare(metadata ptr %linear.C.strides, metadata !54, metadata !DIExpression()), !dbg !27
  %C = load ptr, ptr %C_handle, align 8, !dbg !27
  call void @llvm.dbg.declare(metadata ptr %C, metadata !55, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.declare(metadata ptr %C, metadata !55, metadata !DIExpression()), !dbg !27
  call void @llvm.assume(i1 true) [ "align"(ptr %C, i64 64) ], !dbg !27
  %122 = getelementptr inbounds %1, ptr %A_handle, i64 0, i32 3, i32 2, !dbg !27
  %123 = load i16, ptr %122, align 2, !dbg !27
  %124 = icmp ne i16 %123, 1, !dbg !27
  %125 = getelementptr inbounds %1, ptr %A_handle, i64 0, i32 3, i32 1, !dbg !27
  %126 = load i8, ptr %125, align 1, !dbg !27
  %127 = icmp ne i8 %126, 16, !dbg !27
  %128 = getelementptr inbounds %1, ptr %A_handle, i64 0, i32 3, i32 0, !dbg !27
  %129 = load i8, ptr %128, align 1, !dbg !27
  %130 = icmp ne i8 %129, 4, !dbg !27
  %131 = or i1 %127, %130, !dbg !27
  %132 = or i1 %124, %131, !dbg !27
  br i1 %132, label %if_then33, label %if_end34, !dbg !27, !prof !29

handle_init27:                                    ; preds = %if_then25
  %133 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !27, !tbaa !30
  %134 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !27, !tbaa !30
  %135 = call i32 %134(ptr %133, ptr nonnull @.str.11, ptr nonnull %31), !dbg !27
  %136 = icmp eq i32 %135, 0, !dbg !27
  br i1 %136, label %call_end30, label %common.ret, !dbg !27, !prof !29

handle_init_end28:                                ; preds = %call_end30, %if_then25
  %137 = phi ptr [ %120, %if_then25 ], [ %140, %call_end30 ], !dbg !27
  %138 = call i32 %119(ptr %137, ptr nonnull %stack_ffi_any375, i32 4, ptr nonnull %118), !dbg !27
  %139 = icmp eq i32 %138, 0, !dbg !27
  br i1 %139, label %if_end26, label %common.ret, !dbg !27, !prof !29

call_end30:                                       ; preds = %handle_init27
  %140 = load ptr, ptr %31, align 8, !dbg !27
  store ptr %140, ptr @.tvm_func.__tvm_error_ndim_mismatch, align 8, !dbg !27
  br label %handle_init_end28, !dbg !27

if_then33:                                        ; preds = %if_end26
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any375, align 8, !dbg !27
  %141 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 0, i32 2, !dbg !27
  store ptr @.str.9, ptr %141, align 8, !dbg !27
  %142 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %142, align 8, !dbg !27
  %143 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 2, !dbg !27
  store ptr @.str.10, ptr %143, align 8, !dbg !27
  %144 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %144, align 8, !dbg !27
  %145 = load i8, ptr %128, align 1, !dbg !27
  %146 = zext i8 %145 to i64, !dbg !27
  %147 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 2, !dbg !27
  store i64 %146, ptr %147, align 8, !dbg !27
  %148 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %148, align 8, !dbg !27
  %149 = load i8, ptr %125, align 1, !dbg !27
  %150 = zext i8 %149 to i64, !dbg !27
  %151 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 2, !dbg !27
  store i64 %150, ptr %151, align 8, !dbg !27
  %152 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %152, align 8, !dbg !27
  %153 = load i16, ptr %122, align 2, !dbg !27
  %154 = zext i16 %153 to i64, !dbg !27
  %155 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 2, !dbg !27
  store i64 %154, ptr %155, align 8, !dbg !27
  %156 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 5, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %156, align 8, !dbg !27
  %157 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 5, i32 2, !dbg !27
  store i64 4, ptr %157, align 8, !dbg !27
  %158 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 6, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %158, align 8, !dbg !27
  %159 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 6, i32 2, !dbg !27
  store i64 16, ptr %159, align 8, !dbg !27
  %160 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 7, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %160, align 8, !dbg !27
  %161 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 7, i32 2, !dbg !27
  store i64 1, ptr %161, align 8, !dbg !27
  %162 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 8, i32 0, !dbg !27
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %162, i8 0, i64 16, i1 false), !dbg !27
  %163 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !27, !tbaa !30
  %164 = load ptr, ptr @.tvm_func.__tvm_error_dtype_mismatch, align 8, !dbg !27
  %.not436 = icmp eq ptr %164, null, !dbg !27
  br i1 %.not436, label %handle_init35, label %handle_init_end36, !dbg !27, !prof !33

if_end34:                                         ; preds = %handle_init_end36, %if_end26
  %165 = load i64, ptr %linear.A.shape, align 8, !dbg !27, !tbaa !56
  %166 = and i64 %165, 4294967295, !dbg !27
  %.not379 = icmp eq i64 %166, 1, !dbg !27
  br i1 %.not379, label %if_end42, label %if_then41, !dbg !27, !prof !33

handle_init35:                                    ; preds = %if_then33
  %167 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !27, !tbaa !30
  %168 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !27, !tbaa !30
  %169 = call i32 %168(ptr %167, ptr nonnull @.str.14, ptr nonnull %30), !dbg !27
  %170 = icmp eq i32 %169, 0, !dbg !27
  br i1 %170, label %call_end38, label %common.ret, !dbg !27, !prof !29

handle_init_end36:                                ; preds = %call_end38, %if_then33
  %171 = phi ptr [ %164, %if_then33 ], [ %174, %call_end38 ], !dbg !27
  %172 = call i32 %163(ptr %171, ptr nonnull %stack_ffi_any375, i32 8, ptr nonnull %162), !dbg !27
  %173 = icmp eq i32 %172, 0, !dbg !27
  br i1 %173, label %if_end34, label %common.ret, !dbg !27, !prof !29

call_end38:                                       ; preds = %handle_init35
  %174 = load ptr, ptr %30, align 8, !dbg !27
  store ptr %174, ptr @.tvm_func.__tvm_error_dtype_mismatch, align 8, !dbg !27
  br label %handle_init_end36, !dbg !27

if_then41:                                        ; preds = %if_end34
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any375, align 8, !dbg !27
  %175 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 0, i32 2, !dbg !27
  store ptr @.str.9, ptr %175, align 8, !dbg !27
  %176 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %176, align 8, !dbg !27
  %177 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 2, !dbg !27
  store ptr @.str.10, ptr %177, align 8, !dbg !27
  %178 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %178, align 8, !dbg !27
  %179 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 2, !dbg !27
  store ptr @.str.15, ptr %179, align 8, !dbg !27
  %180 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %180, align 8, !dbg !27
  %181 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 2, !dbg !27
  store i64 1, ptr %181, align 8, !dbg !27
  %182 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %182, align 8, !dbg !27
  %183 = load i64, ptr %linear.A.shape, align 8, !dbg !27, !tbaa !56
  %sext434 = shl i64 %183, 32, !dbg !27
  %184 = ashr exact i64 %sext434, 32, !dbg !27
  %185 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 2, !dbg !27
  store i64 %184, ptr %185, align 8, !dbg !27
  %186 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 5, i32 0, !dbg !27
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %186, i8 0, i64 16, i1 false), !dbg !27
  %187 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !27, !tbaa !30
  %188 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !27
  %.not435 = icmp eq ptr %188, null, !dbg !27
  br i1 %.not435, label %handle_init43, label %handle_init_end44, !dbg !27, !prof !33

if_end42:                                         ; preds = %handle_init_end44, %if_end34
  %189 = getelementptr inbounds i64, ptr %linear.A.shape, i64 1, !dbg !27
  %190 = load i64, ptr %189, align 8, !dbg !27, !tbaa !56
  %191 = and i64 %190, 4294967295, !dbg !27
  %.not380 = icmp eq i64 %191, 1024, !dbg !27
  br i1 %.not380, label %if_end50, label %if_then49, !dbg !27, !prof !33

handle_init43:                                    ; preds = %if_then41
  %192 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !27, !tbaa !30
  %193 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !27, !tbaa !30
  %194 = call i32 %193(ptr %192, ptr nonnull @.str.16, ptr nonnull %29), !dbg !27
  %195 = icmp eq i32 %194, 0, !dbg !27
  br i1 %195, label %call_end46, label %common.ret, !dbg !27, !prof !29

handle_init_end44:                                ; preds = %call_end46, %if_then41
  %196 = phi ptr [ %188, %if_then41 ], [ %199, %call_end46 ], !dbg !27
  %197 = call i32 %187(ptr %196, ptr nonnull %stack_ffi_any375, i32 5, ptr nonnull %186), !dbg !27
  %198 = icmp eq i32 %197, 0, !dbg !27
  br i1 %198, label %if_end42, label %common.ret, !dbg !27, !prof !29

call_end46:                                       ; preds = %handle_init43
  %199 = load ptr, ptr %29, align 8, !dbg !27
  store ptr %199, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !27
  br label %handle_init_end44, !dbg !27

if_then49:                                        ; preds = %if_end42
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any375, align 8, !dbg !27
  %200 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 0, i32 2, !dbg !27
  store ptr @.str.9, ptr %200, align 8, !dbg !27
  %201 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %201, align 8, !dbg !27
  %202 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 2, !dbg !27
  store ptr @.str.10, ptr %202, align 8, !dbg !27
  %203 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %203, align 8, !dbg !27
  %204 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 2, !dbg !27
  store ptr @.str.17, ptr %204, align 8, !dbg !27
  %205 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %205, align 8, !dbg !27
  %206 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 2, !dbg !27
  store i64 1024, ptr %206, align 8, !dbg !27
  %207 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %207, align 8, !dbg !27
  %208 = load i64, ptr %189, align 8, !dbg !27, !tbaa !56
  %sext432 = shl i64 %208, 32, !dbg !27
  %209 = ashr exact i64 %sext432, 32, !dbg !27
  %210 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 2, !dbg !27
  store i64 %209, ptr %210, align 8, !dbg !27
  %211 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 5, i32 0, !dbg !27
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %211, i8 0, i64 16, i1 false), !dbg !27
  %212 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !27, !tbaa !30
  %213 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !27
  %.not433 = icmp eq ptr %213, null, !dbg !27
  br i1 %.not433, label %handle_init51, label %handle_init_end52, !dbg !27, !prof !33

if_end50:                                         ; preds = %handle_init_end52, %if_end42
  %214 = icmp eq ptr %linear.A.strides, null, !dbg !27
  br i1 %214, label %if_then73, label %if_end58, !dbg !27

handle_init51:                                    ; preds = %if_then49
  %215 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !27, !tbaa !30
  %216 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !27, !tbaa !30
  %217 = call i32 %216(ptr %215, ptr nonnull @.str.16, ptr nonnull %28), !dbg !27
  %218 = icmp eq i32 %217, 0, !dbg !27
  br i1 %218, label %call_end54, label %common.ret, !dbg !27, !prof !29

handle_init_end52:                                ; preds = %call_end54, %if_then49
  %219 = phi ptr [ %213, %if_then49 ], [ %222, %call_end54 ], !dbg !27
  %220 = call i32 %212(ptr %219, ptr nonnull %stack_ffi_any375, i32 5, ptr nonnull %211), !dbg !27
  %221 = icmp eq i32 %220, 0, !dbg !27
  br i1 %221, label %if_end50, label %common.ret, !dbg !27, !prof !29

call_end54:                                       ; preds = %handle_init51
  %222 = load ptr, ptr %28, align 8, !dbg !27
  store ptr %222, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !27
  br label %handle_init_end52, !dbg !27

if_end58:                                         ; preds = %if_end50
  %223 = getelementptr inbounds i64, ptr %linear.A.strides, i64 1, !dbg !27
  %224 = load i64, ptr %223, align 8, !dbg !27, !tbaa !56
  %225 = and i64 %224, 4294967295, !dbg !27
  %.not381 = icmp eq i64 %225, 1, !dbg !27
  br i1 %.not381, label %if_end72, label %if_end63, !dbg !27, !prof !33

if_end63:                                         ; preds = %if_end58
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any375, align 8, !dbg !27
  %226 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 0, i32 2, !dbg !27
  store ptr @.str.9, ptr %226, align 8, !dbg !27
  %227 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %227, align 8, !dbg !27
  %228 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 2, !dbg !27
  store ptr @.str.10, ptr %228, align 8, !dbg !27
  %229 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %229, align 8, !dbg !27
  %230 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 2, !dbg !27
  store ptr @.str.18, ptr %230, align 8, !dbg !27
  %231 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %231, align 8, !dbg !27
  %232 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 2, !dbg !27
  store i64 1, ptr %232, align 8, !dbg !27
  %233 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %233, align 8, !dbg !27
  %234 = load i64, ptr %223, align 8, !dbg !27, !tbaa !56
  %sext430 = shl i64 %234, 32, !dbg !27
  %235 = ashr exact i64 %sext430, 32, !dbg !27
  %236 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 2, !dbg !27
  store i64 %235, ptr %236, align 8, !dbg !27
  %237 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 5, i32 0, !dbg !27
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %237, i8 0, i64 16, i1 false), !dbg !27
  %238 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !27, !tbaa !30
  %239 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !27
  %.not431 = icmp eq ptr %239, null, !dbg !27
  br i1 %.not431, label %handle_init64, label %handle_init_end65, !dbg !27, !prof !33

handle_init64:                                    ; preds = %if_end63
  %240 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !27, !tbaa !30
  %241 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !27, !tbaa !30
  %242 = call i32 %241(ptr %240, ptr nonnull @.str.16, ptr nonnull %27), !dbg !27
  %243 = icmp eq i32 %242, 0, !dbg !27
  br i1 %243, label %call_end67, label %common.ret, !dbg !27, !prof !29

handle_init_end65:                                ; preds = %call_end67, %if_end63
  %244 = phi ptr [ %239, %if_end63 ], [ %247, %call_end67 ], !dbg !27
  %245 = call i32 %238(ptr %244, ptr nonnull %stack_ffi_any375, i32 5, ptr nonnull %237), !dbg !27
  %246 = icmp eq i32 %245, 0, !dbg !27
  br i1 %246, label %if_end72, label %common.ret, !dbg !27, !prof !29

call_end67:                                       ; preds = %handle_init64
  %247 = load ptr, ptr %27, align 8, !dbg !27
  store ptr %247, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !27
  br label %handle_init_end65, !dbg !27

if_end72:                                         ; preds = %if_end58, %handle_init_end65
  %248 = load i64, ptr %linear.A.strides, align 8, !dbg !27, !tbaa !56
  %249 = and i64 %248, 4294967295, !dbg !27
  %.not382 = icmp eq i64 %249, 1024, !dbg !27
  br i1 %.not382, label %if_end74, label %if_then73, !dbg !27, !prof !33

if_then73:                                        ; preds = %if_end50, %if_end72
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any375, align 8, !dbg !27
  %250 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 0, i32 2, !dbg !27
  store ptr @.str.9, ptr %250, align 8, !dbg !27
  %251 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %251, align 8, !dbg !27
  %252 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 2, !dbg !27
  store ptr @.str.10, ptr %252, align 8, !dbg !27
  %253 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %253, align 8, !dbg !27
  %254 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 2, !dbg !27
  store ptr @.str.19, ptr %254, align 8, !dbg !27
  %255 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %255, align 8, !dbg !27
  %256 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 2, !dbg !27
  store i64 1024, ptr %256, align 8, !dbg !27
  %257 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %257, align 8, !dbg !27
  br i1 %214, label %if_end77, label %if_else76, !dbg !27

if_end74:                                         ; preds = %handle_init_end79, %if_end72
  %258 = getelementptr inbounds %1, ptr %A_handle, i64 0, i32 6, !dbg !27
  %259 = load i64, ptr %258, align 8, !dbg !27
  %.not383 = icmp eq i64 %259, 0, !dbg !27
  br i1 %.not383, label %if_end85, label %if_then84, !dbg !27, !prof !33

if_else76:                                        ; preds = %if_then73
  %260 = load i64, ptr %linear.A.strides, align 8, !dbg !27, !tbaa !56
  br label %if_end77, !dbg !27

if_end77:                                         ; preds = %if_then73, %if_else76
  %261 = phi i64 [ %260, %if_else76 ], [ 1, %if_then73 ], !dbg !27
  %sext428 = shl i64 %261, 32, !dbg !27
  %262 = ashr exact i64 %sext428, 32, !dbg !27
  %263 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 2, !dbg !27
  store i64 %262, ptr %263, align 8, !dbg !27
  %264 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 5, i32 0, !dbg !27
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %264, i8 0, i64 16, i1 false), !dbg !27
  %265 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !27, !tbaa !30
  %266 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !27
  %.not429 = icmp eq ptr %266, null, !dbg !27
  br i1 %.not429, label %handle_init78, label %handle_init_end79, !dbg !27, !prof !33

handle_init78:                                    ; preds = %if_end77
  %267 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !27, !tbaa !30
  %268 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !27, !tbaa !30
  %269 = call i32 %268(ptr %267, ptr nonnull @.str.16, ptr nonnull %26), !dbg !27
  %270 = icmp eq i32 %269, 0, !dbg !27
  br i1 %270, label %call_end81, label %common.ret, !dbg !27, !prof !29

handle_init_end79:                                ; preds = %call_end81, %if_end77
  %271 = phi ptr [ %266, %if_end77 ], [ %274, %call_end81 ], !dbg !27
  %272 = call i32 %265(ptr %271, ptr nonnull %stack_ffi_any375, i32 5, ptr nonnull %264), !dbg !27
  %273 = icmp eq i32 %272, 0, !dbg !27
  br i1 %273, label %if_end74, label %common.ret, !dbg !27, !prof !29

call_end81:                                       ; preds = %handle_init78
  %274 = load ptr, ptr %26, align 8, !dbg !27
  store ptr %274, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !27
  br label %handle_init_end79, !dbg !27

if_then84:                                        ; preds = %if_end74
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any375, align 8, !dbg !27
  %275 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 0, i32 2, !dbg !27
  store ptr @.str.9, ptr %275, align 8, !dbg !27
  %276 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %276, align 8, !dbg !27
  %277 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 2, !dbg !27
  store ptr @.str.10, ptr %277, align 8, !dbg !27
  %278 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %278, align 8, !dbg !27
  %279 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 2, !dbg !27
  store i64 0, ptr %279, align 8, !dbg !27
  %280 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %280, align 8, !dbg !27
  %281 = load i64, ptr %258, align 8, !dbg !27
  %282 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 2, !dbg !27
  store i64 %281, ptr %282, align 8, !dbg !27
  %283 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 0, !dbg !27
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %283, i8 0, i64 16, i1 false), !dbg !27
  %284 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !27, !tbaa !30
  %285 = load ptr, ptr @.tvm_func.__tvm_error_byte_offset_mismatch, align 8, !dbg !27
  %.not427 = icmp eq ptr %285, null, !dbg !27
  br i1 %.not427, label %handle_init86, label %handle_init_end87, !dbg !27, !prof !33

if_end85:                                         ; preds = %handle_init_end87, %if_end74
  %286 = getelementptr inbounds %1, ptr %A_handle, i64 0, i32 1, i32 0, !dbg !27
  %287 = load i32, ptr %286, align 4, !dbg !27
  %.not384 = icmp eq i32 %287, 2, !dbg !27
  br i1 %.not384, label %if_end93, label %if_then92, !dbg !27, !prof !33

handle_init86:                                    ; preds = %if_then84
  %288 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !27, !tbaa !30
  %289 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !27, !tbaa !30
  %290 = call i32 %289(ptr %288, ptr nonnull @.str.20, ptr nonnull %25), !dbg !27
  %291 = icmp eq i32 %290, 0, !dbg !27
  br i1 %291, label %call_end89, label %common.ret, !dbg !27, !prof !29

handle_init_end87:                                ; preds = %call_end89, %if_then84
  %292 = phi ptr [ %285, %if_then84 ], [ %295, %call_end89 ], !dbg !27
  %293 = call i32 %284(ptr %292, ptr nonnull %stack_ffi_any375, i32 4, ptr nonnull %283), !dbg !27
  %294 = icmp eq i32 %293, 0, !dbg !27
  br i1 %294, label %if_end85, label %common.ret, !dbg !27, !prof !29

call_end89:                                       ; preds = %handle_init86
  %295 = load ptr, ptr %25, align 8, !dbg !27
  store ptr %295, ptr @.tvm_func.__tvm_error_byte_offset_mismatch, align 8, !dbg !27
  br label %handle_init_end87, !dbg !27

if_then92:                                        ; preds = %if_end85
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any375, align 8, !dbg !27
  %296 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 0, i32 2, !dbg !27
  store ptr @.str.9, ptr %296, align 8, !dbg !27
  %297 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %297, align 8, !dbg !27
  %298 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 2, !dbg !27
  store ptr @.str.10, ptr %298, align 8, !dbg !27
  %299 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %299, align 8, !dbg !27
  %300 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 2, !dbg !27
  store i64 2, ptr %300, align 8, !dbg !27
  %301 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %301, align 8, !dbg !27
  %302 = load i32, ptr %286, align 4, !dbg !27
  %303 = sext i32 %302 to i64, !dbg !27
  %304 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 2, !dbg !27
  store i64 %303, ptr %304, align 8, !dbg !27
  %305 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 0, !dbg !27
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %305, i8 0, i64 16, i1 false), !dbg !27
  %306 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !27, !tbaa !30
  %307 = load ptr, ptr @.tvm_func.__tvm_error_device_type_mismatch, align 8, !dbg !27
  %.not426 = icmp eq ptr %307, null, !dbg !27
  br i1 %.not426, label %handle_init94, label %handle_init_end95, !dbg !27, !prof !33

if_end93:                                         ; preds = %handle_init_end95, %if_end85
  %308 = icmp eq ptr %A, null, !dbg !27
  br i1 %308, label %if_then100, label %if_end101, !dbg !27, !prof !29

handle_init94:                                    ; preds = %if_then92
  %309 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !27, !tbaa !30
  %310 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !27, !tbaa !30
  %311 = call i32 %310(ptr %309, ptr nonnull @.str.21, ptr nonnull %24), !dbg !27
  %312 = icmp eq i32 %311, 0, !dbg !27
  br i1 %312, label %call_end97, label %common.ret, !dbg !27, !prof !29

handle_init_end95:                                ; preds = %call_end97, %if_then92
  %313 = phi ptr [ %307, %if_then92 ], [ %316, %call_end97 ], !dbg !27
  %314 = call i32 %306(ptr %313, ptr nonnull %stack_ffi_any375, i32 4, ptr nonnull %305), !dbg !27
  %315 = icmp eq i32 %314, 0, !dbg !27
  br i1 %315, label %if_end93, label %common.ret, !dbg !27, !prof !29

call_end97:                                       ; preds = %handle_init94
  %316 = load ptr, ptr %24, align 8, !dbg !27
  store ptr %316, ptr @.tvm_func.__tvm_error_device_type_mismatch, align 8, !dbg !27
  br label %handle_init_end95, !dbg !27

if_then100:                                       ; preds = %if_end93
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any375, align 8, !dbg !27
  %317 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 0, i32 2, !dbg !27
  store ptr @.str.9, ptr %317, align 8, !dbg !27
  %318 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %318, align 8, !dbg !27
  %319 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 2, !dbg !27
  store ptr @.str.10, ptr %319, align 8, !dbg !27
  %320 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %320, align 8, !dbg !27
  %321 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 2, !dbg !27
  store ptr @.str.22, ptr %321, align 8, !dbg !27
  %322 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 0, !dbg !27
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %322, i8 0, i64 16, i1 false), !dbg !27
  %323 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !27, !tbaa !30
  %324 = load ptr, ptr @.tvm_func.__tvm_error_null_ptr, align 8, !dbg !27
  %.not425 = icmp eq ptr %324, null, !dbg !27
  br i1 %.not425, label %handle_init102, label %handle_init_end103, !dbg !27, !prof !33

if_end101:                                        ; preds = %handle_init_end103, %if_end93
  %325 = getelementptr inbounds %1, ptr %B_handle, i64 0, i32 3, i32 2, !dbg !27
  %326 = load i16, ptr %325, align 2, !dbg !27
  %327 = icmp ne i16 %326, 1, !dbg !27
  %328 = getelementptr inbounds %1, ptr %B_handle, i64 0, i32 3, i32 1, !dbg !27
  %329 = load i8, ptr %328, align 1, !dbg !27
  %330 = icmp ne i8 %329, 16, !dbg !27
  %331 = getelementptr inbounds %1, ptr %B_handle, i64 0, i32 3, i32 0, !dbg !27
  %332 = load i8, ptr %331, align 1, !dbg !27
  %333 = icmp ne i8 %332, 4, !dbg !27
  %334 = or i1 %330, %333, !dbg !27
  %335 = or i1 %327, %334, !dbg !27
  br i1 %335, label %if_then108, label %if_end109, !dbg !27, !prof !29

handle_init102:                                   ; preds = %if_then100
  %336 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !27, !tbaa !30
  %337 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !27, !tbaa !30
  %338 = call i32 %337(ptr %336, ptr nonnull @.str.23, ptr nonnull %23), !dbg !27
  %339 = icmp eq i32 %338, 0, !dbg !27
  br i1 %339, label %call_end105, label %common.ret, !dbg !27, !prof !29

handle_init_end103:                               ; preds = %call_end105, %if_then100
  %340 = phi ptr [ %324, %if_then100 ], [ %343, %call_end105 ], !dbg !27
  %341 = call i32 %323(ptr %340, ptr nonnull %stack_ffi_any375, i32 3, ptr nonnull %322), !dbg !27
  %342 = icmp eq i32 %341, 0, !dbg !27
  br i1 %342, label %if_end101, label %common.ret, !dbg !27, !prof !29

call_end105:                                      ; preds = %handle_init102
  %343 = load ptr, ptr %23, align 8, !dbg !27
  store ptr %343, ptr @.tvm_func.__tvm_error_null_ptr, align 8, !dbg !27
  br label %handle_init_end103, !dbg !27

if_then108:                                       ; preds = %if_end101
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any375, align 8, !dbg !27
  %344 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 0, i32 2, !dbg !27
  store ptr @.str.9, ptr %344, align 8, !dbg !27
  %345 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %345, align 8, !dbg !27
  %346 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 2, !dbg !27
  store ptr @.str.12, ptr %346, align 8, !dbg !27
  %347 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %347, align 8, !dbg !27
  %348 = load i8, ptr %331, align 1, !dbg !27
  %349 = zext i8 %348 to i64, !dbg !27
  %350 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 2, !dbg !27
  store i64 %349, ptr %350, align 8, !dbg !27
  %351 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %351, align 8, !dbg !27
  %352 = load i8, ptr %328, align 1, !dbg !27
  %353 = zext i8 %352 to i64, !dbg !27
  %354 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 2, !dbg !27
  store i64 %353, ptr %354, align 8, !dbg !27
  %355 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %355, align 8, !dbg !27
  %356 = load i16, ptr %325, align 2, !dbg !27
  %357 = zext i16 %356 to i64, !dbg !27
  %358 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 2, !dbg !27
  store i64 %357, ptr %358, align 8, !dbg !27
  %359 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 5, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %359, align 8, !dbg !27
  %360 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 5, i32 2, !dbg !27
  store i64 4, ptr %360, align 8, !dbg !27
  %361 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 6, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %361, align 8, !dbg !27
  %362 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 6, i32 2, !dbg !27
  store i64 16, ptr %362, align 8, !dbg !27
  %363 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 7, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %363, align 8, !dbg !27
  %364 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 7, i32 2, !dbg !27
  store i64 1, ptr %364, align 8, !dbg !27
  %365 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 8, i32 0, !dbg !27
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %365, i8 0, i64 16, i1 false), !dbg !27
  %366 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !27, !tbaa !30
  %367 = load ptr, ptr @.tvm_func.__tvm_error_dtype_mismatch, align 8, !dbg !27
  %.not424 = icmp eq ptr %367, null, !dbg !27
  br i1 %.not424, label %handle_init110, label %handle_init_end111, !dbg !27, !prof !33

if_end109:                                        ; preds = %handle_init_end111, %if_end101
  %368 = load i64, ptr %linear.B.shape, align 8, !dbg !27, !tbaa !56
  %369 = and i64 %368, 4294967295, !dbg !27
  %.not385 = icmp eq i64 %369, 6144, !dbg !27
  br i1 %.not385, label %if_end117, label %if_then116, !dbg !27, !prof !33

handle_init110:                                   ; preds = %if_then108
  %370 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !27, !tbaa !30
  %371 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !27, !tbaa !30
  %372 = call i32 %371(ptr %370, ptr nonnull @.str.14, ptr nonnull %22), !dbg !27
  %373 = icmp eq i32 %372, 0, !dbg !27
  br i1 %373, label %call_end113, label %common.ret, !dbg !27, !prof !29

handle_init_end111:                               ; preds = %call_end113, %if_then108
  %374 = phi ptr [ %367, %if_then108 ], [ %377, %call_end113 ], !dbg !27
  %375 = call i32 %366(ptr %374, ptr nonnull %stack_ffi_any375, i32 8, ptr nonnull %365), !dbg !27
  %376 = icmp eq i32 %375, 0, !dbg !27
  br i1 %376, label %if_end109, label %common.ret, !dbg !27, !prof !29

call_end113:                                      ; preds = %handle_init110
  %377 = load ptr, ptr %22, align 8, !dbg !27
  store ptr %377, ptr @.tvm_func.__tvm_error_dtype_mismatch, align 8, !dbg !27
  br label %handle_init_end111, !dbg !27

if_then116:                                       ; preds = %if_end109
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any375, align 8, !dbg !27
  %378 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 0, i32 2, !dbg !27
  store ptr @.str.9, ptr %378, align 8, !dbg !27
  %379 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %379, align 8, !dbg !27
  %380 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 2, !dbg !27
  store ptr @.str.12, ptr %380, align 8, !dbg !27
  %381 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %381, align 8, !dbg !27
  %382 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 2, !dbg !27
  store ptr @.str.15, ptr %382, align 8, !dbg !27
  %383 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %383, align 8, !dbg !27
  %384 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 2, !dbg !27
  store i64 6144, ptr %384, align 8, !dbg !27
  %385 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %385, align 8, !dbg !27
  %386 = load i64, ptr %linear.B.shape, align 8, !dbg !27, !tbaa !56
  %sext422 = shl i64 %386, 32, !dbg !27
  %387 = ashr exact i64 %sext422, 32, !dbg !27
  %388 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 2, !dbg !27
  store i64 %387, ptr %388, align 8, !dbg !27
  %389 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 5, i32 0, !dbg !27
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %389, i8 0, i64 16, i1 false), !dbg !27
  %390 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !27, !tbaa !30
  %391 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !27
  %.not423 = icmp eq ptr %391, null, !dbg !27
  br i1 %.not423, label %handle_init118, label %handle_init_end119, !dbg !27, !prof !33

if_end117:                                        ; preds = %handle_init_end119, %if_end109
  %392 = getelementptr inbounds i64, ptr %linear.B.shape, i64 1, !dbg !27
  %393 = load i64, ptr %392, align 8, !dbg !27, !tbaa !56
  %394 = and i64 %393, 4294967295, !dbg !27
  %.not386 = icmp eq i64 %394, 1024, !dbg !27
  br i1 %.not386, label %if_end125, label %if_then124, !dbg !27, !prof !33

handle_init118:                                   ; preds = %if_then116
  %395 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !27, !tbaa !30
  %396 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !27, !tbaa !30
  %397 = call i32 %396(ptr %395, ptr nonnull @.str.16, ptr nonnull %21), !dbg !27
  %398 = icmp eq i32 %397, 0, !dbg !27
  br i1 %398, label %call_end121, label %common.ret, !dbg !27, !prof !29

handle_init_end119:                               ; preds = %call_end121, %if_then116
  %399 = phi ptr [ %391, %if_then116 ], [ %402, %call_end121 ], !dbg !27
  %400 = call i32 %390(ptr %399, ptr nonnull %stack_ffi_any375, i32 5, ptr nonnull %389), !dbg !27
  %401 = icmp eq i32 %400, 0, !dbg !27
  br i1 %401, label %if_end117, label %common.ret, !dbg !27, !prof !29

call_end121:                                      ; preds = %handle_init118
  %402 = load ptr, ptr %21, align 8, !dbg !27
  store ptr %402, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !27
  br label %handle_init_end119, !dbg !27

if_then124:                                       ; preds = %if_end117
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any375, align 8, !dbg !27
  %403 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 0, i32 2, !dbg !27
  store ptr @.str.9, ptr %403, align 8, !dbg !27
  %404 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %404, align 8, !dbg !27
  %405 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 2, !dbg !27
  store ptr @.str.12, ptr %405, align 8, !dbg !27
  %406 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %406, align 8, !dbg !27
  %407 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 2, !dbg !27
  store ptr @.str.17, ptr %407, align 8, !dbg !27
  %408 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %408, align 8, !dbg !27
  %409 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 2, !dbg !27
  store i64 1024, ptr %409, align 8, !dbg !27
  %410 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %410, align 8, !dbg !27
  %411 = load i64, ptr %392, align 8, !dbg !27, !tbaa !56
  %sext420 = shl i64 %411, 32, !dbg !27
  %412 = ashr exact i64 %sext420, 32, !dbg !27
  %413 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 2, !dbg !27
  store i64 %412, ptr %413, align 8, !dbg !27
  %414 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 5, i32 0, !dbg !27
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %414, i8 0, i64 16, i1 false), !dbg !27
  %415 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !27, !tbaa !30
  %416 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !27
  %.not421 = icmp eq ptr %416, null, !dbg !27
  br i1 %.not421, label %handle_init126, label %handle_init_end127, !dbg !27, !prof !33

if_end125:                                        ; preds = %handle_init_end127, %if_end117
  %417 = icmp eq ptr %linear.B.strides, null, !dbg !27
  br i1 %417, label %if_then149, label %if_end134, !dbg !27

handle_init126:                                   ; preds = %if_then124
  %418 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !27, !tbaa !30
  %419 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !27, !tbaa !30
  %420 = call i32 %419(ptr %418, ptr nonnull @.str.16, ptr nonnull %20), !dbg !27
  %421 = icmp eq i32 %420, 0, !dbg !27
  br i1 %421, label %call_end129, label %common.ret, !dbg !27, !prof !29

handle_init_end127:                               ; preds = %call_end129, %if_then124
  %422 = phi ptr [ %416, %if_then124 ], [ %425, %call_end129 ], !dbg !27
  %423 = call i32 %415(ptr %422, ptr nonnull %stack_ffi_any375, i32 5, ptr nonnull %414), !dbg !27
  %424 = icmp eq i32 %423, 0, !dbg !27
  br i1 %424, label %if_end125, label %common.ret, !dbg !27, !prof !29

call_end129:                                      ; preds = %handle_init126
  %425 = load ptr, ptr %20, align 8, !dbg !27
  store ptr %425, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !27
  br label %handle_init_end127, !dbg !27

if_end134:                                        ; preds = %if_end125
  %426 = getelementptr inbounds i64, ptr %linear.B.strides, i64 1, !dbg !27
  %427 = load i64, ptr %426, align 8, !dbg !27, !tbaa !56
  %428 = and i64 %427, 4294967295, !dbg !27
  %.not387 = icmp eq i64 %428, 1, !dbg !27
  br i1 %.not387, label %if_end148, label %if_end139, !dbg !27, !prof !33

if_end139:                                        ; preds = %if_end134
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any375, align 8, !dbg !27
  %429 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 0, i32 2, !dbg !27
  store ptr @.str.9, ptr %429, align 8, !dbg !27
  %430 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %430, align 8, !dbg !27
  %431 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 2, !dbg !27
  store ptr @.str.12, ptr %431, align 8, !dbg !27
  %432 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %432, align 8, !dbg !27
  %433 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 2, !dbg !27
  store ptr @.str.18, ptr %433, align 8, !dbg !27
  %434 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %434, align 8, !dbg !27
  %435 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 2, !dbg !27
  store i64 1, ptr %435, align 8, !dbg !27
  %436 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %436, align 8, !dbg !27
  %437 = load i64, ptr %426, align 8, !dbg !27, !tbaa !56
  %sext418 = shl i64 %437, 32, !dbg !27
  %438 = ashr exact i64 %sext418, 32, !dbg !27
  %439 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 2, !dbg !27
  store i64 %438, ptr %439, align 8, !dbg !27
  %440 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 5, i32 0, !dbg !27
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %440, i8 0, i64 16, i1 false), !dbg !27
  %441 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !27, !tbaa !30
  %442 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !27
  %.not419 = icmp eq ptr %442, null, !dbg !27
  br i1 %.not419, label %handle_init140, label %handle_init_end141, !dbg !27, !prof !33

handle_init140:                                   ; preds = %if_end139
  %443 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !27, !tbaa !30
  %444 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !27, !tbaa !30
  %445 = call i32 %444(ptr %443, ptr nonnull @.str.16, ptr nonnull %19), !dbg !27
  %446 = icmp eq i32 %445, 0, !dbg !27
  br i1 %446, label %call_end143, label %common.ret, !dbg !27, !prof !29

handle_init_end141:                               ; preds = %call_end143, %if_end139
  %447 = phi ptr [ %442, %if_end139 ], [ %450, %call_end143 ], !dbg !27
  %448 = call i32 %441(ptr %447, ptr nonnull %stack_ffi_any375, i32 5, ptr nonnull %440), !dbg !27
  %449 = icmp eq i32 %448, 0, !dbg !27
  br i1 %449, label %if_end148, label %common.ret, !dbg !27, !prof !29

call_end143:                                      ; preds = %handle_init140
  %450 = load ptr, ptr %19, align 8, !dbg !27
  store ptr %450, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !27
  br label %handle_init_end141, !dbg !27

if_end148:                                        ; preds = %if_end134, %handle_init_end141
  %451 = load i64, ptr %linear.B.strides, align 8, !dbg !27, !tbaa !56
  %452 = and i64 %451, 4294967295, !dbg !27
  %.not388 = icmp eq i64 %452, 1024, !dbg !27
  br i1 %.not388, label %if_end150, label %if_then149, !dbg !27, !prof !33

if_then149:                                       ; preds = %if_end125, %if_end148
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any375, align 8, !dbg !27
  %453 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 0, i32 2, !dbg !27
  store ptr @.str.9, ptr %453, align 8, !dbg !27
  %454 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %454, align 8, !dbg !27
  %455 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 2, !dbg !27
  store ptr @.str.12, ptr %455, align 8, !dbg !27
  %456 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %456, align 8, !dbg !27
  %457 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 2, !dbg !27
  store ptr @.str.19, ptr %457, align 8, !dbg !27
  %458 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %458, align 8, !dbg !27
  %459 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 2, !dbg !27
  store i64 1024, ptr %459, align 8, !dbg !27
  %460 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %460, align 8, !dbg !27
  br i1 %417, label %if_end153, label %if_else152, !dbg !27

if_end150:                                        ; preds = %handle_init_end155, %if_end148
  %461 = getelementptr inbounds %1, ptr %B_handle, i64 0, i32 6, !dbg !27
  %462 = load i64, ptr %461, align 8, !dbg !27
  %.not389 = icmp eq i64 %462, 0, !dbg !27
  br i1 %.not389, label %if_end161, label %if_then160, !dbg !27, !prof !33

if_else152:                                       ; preds = %if_then149
  %463 = load i64, ptr %linear.B.strides, align 8, !dbg !27, !tbaa !56
  br label %if_end153, !dbg !27

if_end153:                                        ; preds = %if_then149, %if_else152
  %464 = phi i64 [ %463, %if_else152 ], [ 1, %if_then149 ], !dbg !27
  %sext416 = shl i64 %464, 32, !dbg !27
  %465 = ashr exact i64 %sext416, 32, !dbg !27
  %466 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 2, !dbg !27
  store i64 %465, ptr %466, align 8, !dbg !27
  %467 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 5, i32 0, !dbg !27
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %467, i8 0, i64 16, i1 false), !dbg !27
  %468 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !27, !tbaa !30
  %469 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !27
  %.not417 = icmp eq ptr %469, null, !dbg !27
  br i1 %.not417, label %handle_init154, label %handle_init_end155, !dbg !27, !prof !33

handle_init154:                                   ; preds = %if_end153
  %470 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !27, !tbaa !30
  %471 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !27, !tbaa !30
  %472 = call i32 %471(ptr %470, ptr nonnull @.str.16, ptr nonnull %18), !dbg !27
  %473 = icmp eq i32 %472, 0, !dbg !27
  br i1 %473, label %call_end157, label %common.ret, !dbg !27, !prof !29

handle_init_end155:                               ; preds = %call_end157, %if_end153
  %474 = phi ptr [ %469, %if_end153 ], [ %477, %call_end157 ], !dbg !27
  %475 = call i32 %468(ptr %474, ptr nonnull %stack_ffi_any375, i32 5, ptr nonnull %467), !dbg !27
  %476 = icmp eq i32 %475, 0, !dbg !27
  br i1 %476, label %if_end150, label %common.ret, !dbg !27, !prof !29

call_end157:                                      ; preds = %handle_init154
  %477 = load ptr, ptr %18, align 8, !dbg !27
  store ptr %477, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !27
  br label %handle_init_end155, !dbg !27

if_then160:                                       ; preds = %if_end150
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any375, align 8, !dbg !27
  %478 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 0, i32 2, !dbg !27
  store ptr @.str.9, ptr %478, align 8, !dbg !27
  %479 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %479, align 8, !dbg !27
  %480 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 2, !dbg !27
  store ptr @.str.12, ptr %480, align 8, !dbg !27
  %481 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %481, align 8, !dbg !27
  %482 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 2, !dbg !27
  store i64 0, ptr %482, align 8, !dbg !27
  %483 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %483, align 8, !dbg !27
  %484 = load i64, ptr %461, align 8, !dbg !27
  %485 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 2, !dbg !27
  store i64 %484, ptr %485, align 8, !dbg !27
  %486 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 0, !dbg !27
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %486, i8 0, i64 16, i1 false), !dbg !27
  %487 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !27, !tbaa !30
  %488 = load ptr, ptr @.tvm_func.__tvm_error_byte_offset_mismatch, align 8, !dbg !27
  %.not415 = icmp eq ptr %488, null, !dbg !27
  br i1 %.not415, label %handle_init162, label %handle_init_end163, !dbg !27, !prof !33

if_end161:                                        ; preds = %handle_init_end163, %if_end150
  %489 = getelementptr inbounds %1, ptr %B_handle, i64 0, i32 1, i32 1, !dbg !27
  %490 = load i32, ptr %489, align 4, !dbg !27
  %491 = load i32, ptr %75, align 4, !dbg !27
  %.not390 = icmp eq i32 %490, %491, !dbg !27
  br i1 %.not390, label %if_end169, label %if_then168, !dbg !27, !prof !33

handle_init162:                                   ; preds = %if_then160
  %492 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !27, !tbaa !30
  %493 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !27, !tbaa !30
  %494 = call i32 %493(ptr %492, ptr nonnull @.str.20, ptr nonnull %17), !dbg !27
  %495 = icmp eq i32 %494, 0, !dbg !27
  br i1 %495, label %call_end165, label %common.ret, !dbg !27, !prof !29

handle_init_end163:                               ; preds = %call_end165, %if_then160
  %496 = phi ptr [ %488, %if_then160 ], [ %499, %call_end165 ], !dbg !27
  %497 = call i32 %487(ptr %496, ptr nonnull %stack_ffi_any375, i32 4, ptr nonnull %486), !dbg !27
  %498 = icmp eq i32 %497, 0, !dbg !27
  br i1 %498, label %if_end161, label %common.ret, !dbg !27, !prof !29

call_end165:                                      ; preds = %handle_init162
  %499 = load ptr, ptr %17, align 8, !dbg !27
  store ptr %499, ptr @.tvm_func.__tvm_error_byte_offset_mismatch, align 8, !dbg !27
  br label %handle_init_end163, !dbg !27

if_then168:                                       ; preds = %if_end161
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any375, align 8, !dbg !27
  %500 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 0, i32 2, !dbg !27
  store ptr @.str.9, ptr %500, align 8, !dbg !27
  %501 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %501, align 8, !dbg !27
  %502 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 2, !dbg !27
  store ptr @.str.12, ptr %502, align 8, !dbg !27
  %503 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %503, align 8, !dbg !27
  %504 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 2, !dbg !27
  store ptr @.str.24, ptr %504, align 8, !dbg !27
  %505 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %505, align 8, !dbg !27
  %506 = load i32, ptr %75, align 4, !dbg !27
  %507 = sext i32 %506 to i64, !dbg !27
  %508 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 2, !dbg !27
  store i64 %507, ptr %508, align 8, !dbg !27
  %509 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %509, align 8, !dbg !27
  %510 = load i32, ptr %489, align 4, !dbg !27
  %511 = sext i32 %510 to i64, !dbg !27
  %512 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 2, !dbg !27
  store i64 %511, ptr %512, align 8, !dbg !27
  %513 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 5, i32 0, !dbg !27
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %513, i8 0, i64 16, i1 false), !dbg !27
  %514 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !27, !tbaa !30
  %515 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !27
  %.not414 = icmp eq ptr %515, null, !dbg !27
  br i1 %.not414, label %handle_init170, label %handle_init_end171, !dbg !27, !prof !33

if_end169:                                        ; preds = %handle_init_end171, %if_end161
  %516 = getelementptr inbounds %1, ptr %B_handle, i64 0, i32 1, i32 0, !dbg !27
  %517 = load i32, ptr %516, align 4, !dbg !27
  %.not391 = icmp eq i32 %517, 2, !dbg !27
  br i1 %.not391, label %if_end177, label %if_then176, !dbg !27, !prof !33

handle_init170:                                   ; preds = %if_then168
  %518 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !27, !tbaa !30
  %519 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !27, !tbaa !30
  %520 = call i32 %519(ptr %518, ptr nonnull @.str.16, ptr nonnull %16), !dbg !27
  %521 = icmp eq i32 %520, 0, !dbg !27
  br i1 %521, label %call_end173, label %common.ret, !dbg !27, !prof !29

handle_init_end171:                               ; preds = %call_end173, %if_then168
  %522 = phi ptr [ %515, %if_then168 ], [ %525, %call_end173 ], !dbg !27
  %523 = call i32 %514(ptr %522, ptr nonnull %stack_ffi_any375, i32 5, ptr nonnull %513), !dbg !27
  %524 = icmp eq i32 %523, 0, !dbg !27
  br i1 %524, label %if_end169, label %common.ret, !dbg !27, !prof !29

call_end173:                                      ; preds = %handle_init170
  %525 = load ptr, ptr %16, align 8, !dbg !27
  store ptr %525, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !27
  br label %handle_init_end171, !dbg !27

if_then176:                                       ; preds = %if_end169
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any375, align 8, !dbg !27
  %526 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 0, i32 2, !dbg !27
  store ptr @.str.9, ptr %526, align 8, !dbg !27
  %527 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %527, align 8, !dbg !27
  %528 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 2, !dbg !27
  store ptr @.str.12, ptr %528, align 8, !dbg !27
  %529 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %529, align 8, !dbg !27
  %530 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 2, !dbg !27
  store i64 2, ptr %530, align 8, !dbg !27
  %531 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %531, align 8, !dbg !27
  %532 = load i32, ptr %516, align 4, !dbg !27
  %533 = sext i32 %532 to i64, !dbg !27
  %534 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 2, !dbg !27
  store i64 %533, ptr %534, align 8, !dbg !27
  %535 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 0, !dbg !27
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %535, i8 0, i64 16, i1 false), !dbg !27
  %536 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !27, !tbaa !30
  %537 = load ptr, ptr @.tvm_func.__tvm_error_device_type_mismatch, align 8, !dbg !27
  %.not413 = icmp eq ptr %537, null, !dbg !27
  br i1 %.not413, label %handle_init178, label %handle_init_end179, !dbg !27, !prof !33

if_end177:                                        ; preds = %handle_init_end179, %if_end169
  %538 = icmp eq ptr %B, null, !dbg !27
  br i1 %538, label %if_then184, label %if_end185, !dbg !27, !prof !29

handle_init178:                                   ; preds = %if_then176
  %539 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !27, !tbaa !30
  %540 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !27, !tbaa !30
  %541 = call i32 %540(ptr %539, ptr nonnull @.str.21, ptr nonnull %15), !dbg !27
  %542 = icmp eq i32 %541, 0, !dbg !27
  br i1 %542, label %call_end181, label %common.ret, !dbg !27, !prof !29

handle_init_end179:                               ; preds = %call_end181, %if_then176
  %543 = phi ptr [ %537, %if_then176 ], [ %546, %call_end181 ], !dbg !27
  %544 = call i32 %536(ptr %543, ptr nonnull %stack_ffi_any375, i32 4, ptr nonnull %535), !dbg !27
  %545 = icmp eq i32 %544, 0, !dbg !27
  br i1 %545, label %if_end177, label %common.ret, !dbg !27, !prof !29

call_end181:                                      ; preds = %handle_init178
  %546 = load ptr, ptr %15, align 8, !dbg !27
  store ptr %546, ptr @.tvm_func.__tvm_error_device_type_mismatch, align 8, !dbg !27
  br label %handle_init_end179, !dbg !27

if_then184:                                       ; preds = %if_end177
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any375, align 8, !dbg !27
  %547 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 0, i32 2, !dbg !27
  store ptr @.str.9, ptr %547, align 8, !dbg !27
  %548 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %548, align 8, !dbg !27
  %549 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 2, !dbg !27
  store ptr @.str.12, ptr %549, align 8, !dbg !27
  %550 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %550, align 8, !dbg !27
  %551 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 2, !dbg !27
  store ptr @.str.22, ptr %551, align 8, !dbg !27
  %552 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 0, !dbg !27
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %552, i8 0, i64 16, i1 false), !dbg !27
  %553 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !27, !tbaa !30
  %554 = load ptr, ptr @.tvm_func.__tvm_error_null_ptr, align 8, !dbg !27
  %.not412 = icmp eq ptr %554, null, !dbg !27
  br i1 %.not412, label %handle_init186, label %handle_init_end187, !dbg !27, !prof !33

if_end185:                                        ; preds = %handle_init_end187, %if_end177
  %555 = getelementptr inbounds %1, ptr %C_handle, i64 0, i32 3, i32 2, !dbg !27
  %556 = load i16, ptr %555, align 2, !dbg !27
  %557 = icmp ne i16 %556, 1, !dbg !27
  %558 = getelementptr inbounds %1, ptr %C_handle, i64 0, i32 3, i32 1, !dbg !27
  %559 = load i8, ptr %558, align 1, !dbg !27
  %560 = icmp ne i8 %559, 16, !dbg !27
  %561 = getelementptr inbounds %1, ptr %C_handle, i64 0, i32 3, i32 0, !dbg !27
  %562 = load i8, ptr %561, align 1, !dbg !27
  %563 = icmp ne i8 %562, 4, !dbg !27
  %564 = or i1 %560, %563, !dbg !27
  %565 = or i1 %557, %564, !dbg !27
  br i1 %565, label %if_then192, label %if_end193, !dbg !27, !prof !29

handle_init186:                                   ; preds = %if_then184
  %566 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !27, !tbaa !30
  %567 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !27, !tbaa !30
  %568 = call i32 %567(ptr %566, ptr nonnull @.str.23, ptr nonnull %14), !dbg !27
  %569 = icmp eq i32 %568, 0, !dbg !27
  br i1 %569, label %call_end189, label %common.ret, !dbg !27, !prof !29

handle_init_end187:                               ; preds = %call_end189, %if_then184
  %570 = phi ptr [ %554, %if_then184 ], [ %573, %call_end189 ], !dbg !27
  %571 = call i32 %553(ptr %570, ptr nonnull %stack_ffi_any375, i32 3, ptr nonnull %552), !dbg !27
  %572 = icmp eq i32 %571, 0, !dbg !27
  br i1 %572, label %if_end185, label %common.ret, !dbg !27, !prof !29

call_end189:                                      ; preds = %handle_init186
  %573 = load ptr, ptr %14, align 8, !dbg !27
  store ptr %573, ptr @.tvm_func.__tvm_error_null_ptr, align 8, !dbg !27
  br label %handle_init_end187, !dbg !27

if_then192:                                       ; preds = %if_end185
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any375, align 8, !dbg !27
  %574 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 0, i32 2, !dbg !27
  store ptr @.str.9, ptr %574, align 8, !dbg !27
  %575 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %575, align 8, !dbg !27
  %576 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 2, !dbg !27
  store ptr @.str.13, ptr %576, align 8, !dbg !27
  %577 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %577, align 8, !dbg !27
  %578 = load i8, ptr %561, align 1, !dbg !27
  %579 = zext i8 %578 to i64, !dbg !27
  %580 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 2, !dbg !27
  store i64 %579, ptr %580, align 8, !dbg !27
  %581 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %581, align 8, !dbg !27
  %582 = load i8, ptr %558, align 1, !dbg !27
  %583 = zext i8 %582 to i64, !dbg !27
  %584 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 2, !dbg !27
  store i64 %583, ptr %584, align 8, !dbg !27
  %585 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %585, align 8, !dbg !27
  %586 = load i16, ptr %555, align 2, !dbg !27
  %587 = zext i16 %586 to i64, !dbg !27
  %588 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 2, !dbg !27
  store i64 %587, ptr %588, align 8, !dbg !27
  %589 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 5, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %589, align 8, !dbg !27
  %590 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 5, i32 2, !dbg !27
  store i64 4, ptr %590, align 8, !dbg !27
  %591 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 6, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %591, align 8, !dbg !27
  %592 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 6, i32 2, !dbg !27
  store i64 16, ptr %592, align 8, !dbg !27
  %593 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 7, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %593, align 8, !dbg !27
  %594 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 7, i32 2, !dbg !27
  store i64 1, ptr %594, align 8, !dbg !27
  %595 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 8, i32 0, !dbg !27
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %595, i8 0, i64 16, i1 false), !dbg !27
  %596 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !27, !tbaa !30
  %597 = load ptr, ptr @.tvm_func.__tvm_error_dtype_mismatch, align 8, !dbg !27
  %.not411 = icmp eq ptr %597, null, !dbg !27
  br i1 %.not411, label %handle_init194, label %handle_init_end195, !dbg !27, !prof !33

if_end193:                                        ; preds = %handle_init_end195, %if_end185
  %598 = load i64, ptr %linear.C.shape, align 8, !dbg !27, !tbaa !56
  %599 = and i64 %598, 4294967295, !dbg !27
  %.not392 = icmp eq i64 %599, 1, !dbg !27
  br i1 %.not392, label %if_end201, label %if_then200, !dbg !27, !prof !33

handle_init194:                                   ; preds = %if_then192
  %600 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !27, !tbaa !30
  %601 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !27, !tbaa !30
  %602 = call i32 %601(ptr %600, ptr nonnull @.str.14, ptr nonnull %13), !dbg !27
  %603 = icmp eq i32 %602, 0, !dbg !27
  br i1 %603, label %call_end197, label %common.ret, !dbg !27, !prof !29

handle_init_end195:                               ; preds = %call_end197, %if_then192
  %604 = phi ptr [ %597, %if_then192 ], [ %607, %call_end197 ], !dbg !27
  %605 = call i32 %596(ptr %604, ptr nonnull %stack_ffi_any375, i32 8, ptr nonnull %595), !dbg !27
  %606 = icmp eq i32 %605, 0, !dbg !27
  br i1 %606, label %if_end193, label %common.ret, !dbg !27, !prof !29

call_end197:                                      ; preds = %handle_init194
  %607 = load ptr, ptr %13, align 8, !dbg !27
  store ptr %607, ptr @.tvm_func.__tvm_error_dtype_mismatch, align 8, !dbg !27
  br label %handle_init_end195, !dbg !27

if_then200:                                       ; preds = %if_end193
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any375, align 8, !dbg !27
  %608 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 0, i32 2, !dbg !27
  store ptr @.str.9, ptr %608, align 8, !dbg !27
  %609 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %609, align 8, !dbg !27
  %610 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 2, !dbg !27
  store ptr @.str.13, ptr %610, align 8, !dbg !27
  %611 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %611, align 8, !dbg !27
  %612 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 2, !dbg !27
  store ptr @.str.15, ptr %612, align 8, !dbg !27
  %613 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %613, align 8, !dbg !27
  %614 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 2, !dbg !27
  store i64 1, ptr %614, align 8, !dbg !27
  %615 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %615, align 8, !dbg !27
  %616 = load i64, ptr %linear.C.shape, align 8, !dbg !27, !tbaa !56
  %sext409 = shl i64 %616, 32, !dbg !27
  %617 = ashr exact i64 %sext409, 32, !dbg !27
  %618 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 2, !dbg !27
  store i64 %617, ptr %618, align 8, !dbg !27
  %619 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 5, i32 0, !dbg !27
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %619, i8 0, i64 16, i1 false), !dbg !27
  %620 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !27, !tbaa !30
  %621 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !27
  %.not410 = icmp eq ptr %621, null, !dbg !27
  br i1 %.not410, label %handle_init202, label %handle_init_end203, !dbg !27, !prof !33

if_end201:                                        ; preds = %handle_init_end203, %if_end193
  %622 = getelementptr inbounds i64, ptr %linear.C.shape, i64 1, !dbg !27
  %623 = load i64, ptr %622, align 8, !dbg !27, !tbaa !56
  %624 = and i64 %623, 4294967295, !dbg !27
  %.not393 = icmp eq i64 %624, 6144, !dbg !27
  br i1 %.not393, label %if_end209, label %if_then208, !dbg !27, !prof !33

handle_init202:                                   ; preds = %if_then200
  %625 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !27, !tbaa !30
  %626 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !27, !tbaa !30
  %627 = call i32 %626(ptr %625, ptr nonnull @.str.16, ptr nonnull %12), !dbg !27
  %628 = icmp eq i32 %627, 0, !dbg !27
  br i1 %628, label %call_end205, label %common.ret, !dbg !27, !prof !29

handle_init_end203:                               ; preds = %call_end205, %if_then200
  %629 = phi ptr [ %621, %if_then200 ], [ %632, %call_end205 ], !dbg !27
  %630 = call i32 %620(ptr %629, ptr nonnull %stack_ffi_any375, i32 5, ptr nonnull %619), !dbg !27
  %631 = icmp eq i32 %630, 0, !dbg !27
  br i1 %631, label %if_end201, label %common.ret, !dbg !27, !prof !29

call_end205:                                      ; preds = %handle_init202
  %632 = load ptr, ptr %12, align 8, !dbg !27
  store ptr %632, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !27
  br label %handle_init_end203, !dbg !27

if_then208:                                       ; preds = %if_end201
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any375, align 8, !dbg !27
  %633 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 0, i32 2, !dbg !27
  store ptr @.str.9, ptr %633, align 8, !dbg !27
  %634 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %634, align 8, !dbg !27
  %635 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 2, !dbg !27
  store ptr @.str.13, ptr %635, align 8, !dbg !27
  %636 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %636, align 8, !dbg !27
  %637 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 2, !dbg !27
  store ptr @.str.17, ptr %637, align 8, !dbg !27
  %638 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %638, align 8, !dbg !27
  %639 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 2, !dbg !27
  store i64 6144, ptr %639, align 8, !dbg !27
  %640 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %640, align 8, !dbg !27
  %641 = load i64, ptr %622, align 8, !dbg !27, !tbaa !56
  %sext407 = shl i64 %641, 32, !dbg !27
  %642 = ashr exact i64 %sext407, 32, !dbg !27
  %643 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 2, !dbg !27
  store i64 %642, ptr %643, align 8, !dbg !27
  %644 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 5, i32 0, !dbg !27
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %644, i8 0, i64 16, i1 false), !dbg !27
  %645 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !27, !tbaa !30
  %646 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !27
  %.not408 = icmp eq ptr %646, null, !dbg !27
  br i1 %.not408, label %handle_init210, label %handle_init_end211, !dbg !27, !prof !33

if_end209:                                        ; preds = %handle_init_end211, %if_end201
  %647 = icmp eq ptr %linear.C.strides, null, !dbg !27
  br i1 %647, label %if_then233, label %if_end218, !dbg !27

handle_init210:                                   ; preds = %if_then208
  %648 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !27, !tbaa !30
  %649 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !27, !tbaa !30
  %650 = call i32 %649(ptr %648, ptr nonnull @.str.16, ptr nonnull %11), !dbg !27
  %651 = icmp eq i32 %650, 0, !dbg !27
  br i1 %651, label %call_end213, label %common.ret, !dbg !27, !prof !29

handle_init_end211:                               ; preds = %call_end213, %if_then208
  %652 = phi ptr [ %646, %if_then208 ], [ %655, %call_end213 ], !dbg !27
  %653 = call i32 %645(ptr %652, ptr nonnull %stack_ffi_any375, i32 5, ptr nonnull %644), !dbg !27
  %654 = icmp eq i32 %653, 0, !dbg !27
  br i1 %654, label %if_end209, label %common.ret, !dbg !27, !prof !29

call_end213:                                      ; preds = %handle_init210
  %655 = load ptr, ptr %11, align 8, !dbg !27
  store ptr %655, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !27
  br label %handle_init_end211, !dbg !27

if_end218:                                        ; preds = %if_end209
  %656 = getelementptr inbounds i64, ptr %linear.C.strides, i64 1, !dbg !27
  %657 = load i64, ptr %656, align 8, !dbg !27, !tbaa !56
  %658 = and i64 %657, 4294967295, !dbg !27
  %.not394 = icmp eq i64 %658, 1, !dbg !27
  br i1 %.not394, label %if_end232, label %if_end223, !dbg !27, !prof !33

if_end223:                                        ; preds = %if_end218
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any375, align 8, !dbg !27
  %659 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 0, i32 2, !dbg !27
  store ptr @.str.9, ptr %659, align 8, !dbg !27
  %660 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %660, align 8, !dbg !27
  %661 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 2, !dbg !27
  store ptr @.str.13, ptr %661, align 8, !dbg !27
  %662 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %662, align 8, !dbg !27
  %663 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 2, !dbg !27
  store ptr @.str.18, ptr %663, align 8, !dbg !27
  %664 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %664, align 8, !dbg !27
  %665 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 2, !dbg !27
  store i64 1, ptr %665, align 8, !dbg !27
  %666 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %666, align 8, !dbg !27
  %667 = load i64, ptr %656, align 8, !dbg !27, !tbaa !56
  %sext405 = shl i64 %667, 32, !dbg !27
  %668 = ashr exact i64 %sext405, 32, !dbg !27
  %669 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 2, !dbg !27
  store i64 %668, ptr %669, align 8, !dbg !27
  %670 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 5, i32 0, !dbg !27
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %670, i8 0, i64 16, i1 false), !dbg !27
  %671 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !27, !tbaa !30
  %672 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !27
  %.not406 = icmp eq ptr %672, null, !dbg !27
  br i1 %.not406, label %handle_init224, label %handle_init_end225, !dbg !27, !prof !33

handle_init224:                                   ; preds = %if_end223
  %673 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !27, !tbaa !30
  %674 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !27, !tbaa !30
  %675 = call i32 %674(ptr %673, ptr nonnull @.str.16, ptr nonnull %10), !dbg !27
  %676 = icmp eq i32 %675, 0, !dbg !27
  br i1 %676, label %call_end227, label %common.ret, !dbg !27, !prof !29

handle_init_end225:                               ; preds = %call_end227, %if_end223
  %677 = phi ptr [ %672, %if_end223 ], [ %680, %call_end227 ], !dbg !27
  %678 = call i32 %671(ptr %677, ptr nonnull %stack_ffi_any375, i32 5, ptr nonnull %670), !dbg !27
  %679 = icmp eq i32 %678, 0, !dbg !27
  br i1 %679, label %if_end232, label %common.ret, !dbg !27, !prof !29

call_end227:                                      ; preds = %handle_init224
  %680 = load ptr, ptr %10, align 8, !dbg !27
  store ptr %680, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !27
  br label %handle_init_end225, !dbg !27

if_end232:                                        ; preds = %if_end218, %handle_init_end225
  %681 = load i64, ptr %linear.C.strides, align 8, !dbg !27, !tbaa !56
  %682 = and i64 %681, 4294967295, !dbg !27
  %.not395 = icmp eq i64 %682, 6144, !dbg !27
  br i1 %.not395, label %if_end234, label %if_then233, !dbg !27, !prof !33

if_then233:                                       ; preds = %if_end209, %if_end232
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any375, align 8, !dbg !27
  %683 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 0, i32 2, !dbg !27
  store ptr @.str.9, ptr %683, align 8, !dbg !27
  %684 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %684, align 8, !dbg !27
  %685 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 2, !dbg !27
  store ptr @.str.13, ptr %685, align 8, !dbg !27
  %686 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %686, align 8, !dbg !27
  %687 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 2, !dbg !27
  store ptr @.str.19, ptr %687, align 8, !dbg !27
  %688 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %688, align 8, !dbg !27
  %689 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 2, !dbg !27
  store i64 6144, ptr %689, align 8, !dbg !27
  %690 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %690, align 8, !dbg !27
  br i1 %647, label %if_end237, label %if_else236, !dbg !27

if_end234:                                        ; preds = %handle_init_end239, %if_end232
  %691 = getelementptr inbounds %1, ptr %C_handle, i64 0, i32 6, !dbg !27
  %692 = load i64, ptr %691, align 8, !dbg !27
  %.not396 = icmp eq i64 %692, 0, !dbg !27
  br i1 %.not396, label %if_end245, label %if_then244, !dbg !27, !prof !33

if_else236:                                       ; preds = %if_then233
  %693 = load i64, ptr %linear.C.strides, align 8, !dbg !27, !tbaa !56
  br label %if_end237, !dbg !27

if_end237:                                        ; preds = %if_then233, %if_else236
  %694 = phi i64 [ %693, %if_else236 ], [ 1, %if_then233 ], !dbg !27
  %sext = shl i64 %694, 32, !dbg !27
  %695 = ashr exact i64 %sext, 32, !dbg !27
  %696 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 2, !dbg !27
  store i64 %695, ptr %696, align 8, !dbg !27
  %697 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 5, i32 0, !dbg !27
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %697, i8 0, i64 16, i1 false), !dbg !27
  %698 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !27, !tbaa !30
  %699 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !27
  %.not404 = icmp eq ptr %699, null, !dbg !27
  br i1 %.not404, label %handle_init238, label %handle_init_end239, !dbg !27, !prof !33

handle_init238:                                   ; preds = %if_end237
  %700 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !27, !tbaa !30
  %701 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !27, !tbaa !30
  %702 = call i32 %701(ptr %700, ptr nonnull @.str.16, ptr nonnull %9), !dbg !27
  %703 = icmp eq i32 %702, 0, !dbg !27
  br i1 %703, label %call_end241, label %common.ret, !dbg !27, !prof !29

handle_init_end239:                               ; preds = %call_end241, %if_end237
  %704 = phi ptr [ %699, %if_end237 ], [ %707, %call_end241 ], !dbg !27
  %705 = call i32 %698(ptr %704, ptr nonnull %stack_ffi_any375, i32 5, ptr nonnull %697), !dbg !27
  %706 = icmp eq i32 %705, 0, !dbg !27
  br i1 %706, label %if_end234, label %common.ret, !dbg !27, !prof !29

call_end241:                                      ; preds = %handle_init238
  %707 = load ptr, ptr %9, align 8, !dbg !27
  store ptr %707, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !27
  br label %handle_init_end239, !dbg !27

if_then244:                                       ; preds = %if_end234
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any375, align 8, !dbg !27
  %708 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 0, i32 2, !dbg !27
  store ptr @.str.9, ptr %708, align 8, !dbg !27
  %709 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %709, align 8, !dbg !27
  %710 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 2, !dbg !27
  store ptr @.str.13, ptr %710, align 8, !dbg !27
  %711 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %711, align 8, !dbg !27
  %712 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 2, !dbg !27
  store i64 0, ptr %712, align 8, !dbg !27
  %713 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %713, align 8, !dbg !27
  %714 = load i64, ptr %691, align 8, !dbg !27
  %715 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 2, !dbg !27
  store i64 %714, ptr %715, align 8, !dbg !27
  %716 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 0, !dbg !27
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %716, i8 0, i64 16, i1 false), !dbg !27
  %717 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !27, !tbaa !30
  %718 = load ptr, ptr @.tvm_func.__tvm_error_byte_offset_mismatch, align 8, !dbg !27
  %.not403 = icmp eq ptr %718, null, !dbg !27
  br i1 %.not403, label %handle_init246, label %handle_init_end247, !dbg !27, !prof !33

if_end245:                                        ; preds = %handle_init_end247, %if_end234
  %719 = getelementptr inbounds %1, ptr %C_handle, i64 0, i32 1, i32 1, !dbg !27
  %720 = load i32, ptr %719, align 4, !dbg !27
  %721 = load i32, ptr %75, align 4, !dbg !27
  %.not397 = icmp eq i32 %720, %721, !dbg !27
  br i1 %.not397, label %if_end253, label %if_then252, !dbg !27, !prof !33

handle_init246:                                   ; preds = %if_then244
  %722 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !27, !tbaa !30
  %723 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !27, !tbaa !30
  %724 = call i32 %723(ptr %722, ptr nonnull @.str.20, ptr nonnull %8), !dbg !27
  %725 = icmp eq i32 %724, 0, !dbg !27
  br i1 %725, label %call_end249, label %common.ret, !dbg !27, !prof !29

handle_init_end247:                               ; preds = %call_end249, %if_then244
  %726 = phi ptr [ %718, %if_then244 ], [ %729, %call_end249 ], !dbg !27
  %727 = call i32 %717(ptr %726, ptr nonnull %stack_ffi_any375, i32 4, ptr nonnull %716), !dbg !27
  %728 = icmp eq i32 %727, 0, !dbg !27
  br i1 %728, label %if_end245, label %common.ret, !dbg !27, !prof !29

call_end249:                                      ; preds = %handle_init246
  %729 = load ptr, ptr %8, align 8, !dbg !27
  store ptr %729, ptr @.tvm_func.__tvm_error_byte_offset_mismatch, align 8, !dbg !27
  br label %handle_init_end247, !dbg !27

if_then252:                                       ; preds = %if_end245
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any375, align 8, !dbg !27
  %730 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 0, i32 2, !dbg !27
  store ptr @.str.9, ptr %730, align 8, !dbg !27
  %731 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %731, align 8, !dbg !27
  %732 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 2, !dbg !27
  store ptr @.str.13, ptr %732, align 8, !dbg !27
  %733 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %733, align 8, !dbg !27
  %734 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 2, !dbg !27
  store ptr @.str.24, ptr %734, align 8, !dbg !27
  %735 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %735, align 8, !dbg !27
  %736 = load i32, ptr %75, align 4, !dbg !27
  %737 = sext i32 %736 to i64, !dbg !27
  %738 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 2, !dbg !27
  store i64 %737, ptr %738, align 8, !dbg !27
  %739 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %739, align 8, !dbg !27
  %740 = load i32, ptr %719, align 4, !dbg !27
  %741 = sext i32 %740 to i64, !dbg !27
  %742 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 2, !dbg !27
  store i64 %741, ptr %742, align 8, !dbg !27
  %743 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 5, i32 0, !dbg !27
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %743, i8 0, i64 16, i1 false), !dbg !27
  %744 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !27, !tbaa !30
  %745 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !27
  %.not402 = icmp eq ptr %745, null, !dbg !27
  br i1 %.not402, label %handle_init254, label %handle_init_end255, !dbg !27, !prof !33

if_end253:                                        ; preds = %handle_init_end255, %if_end245
  %746 = getelementptr inbounds %1, ptr %C_handle, i64 0, i32 1, i32 0, !dbg !27
  %747 = load i32, ptr %746, align 4, !dbg !27
  %.not398 = icmp eq i32 %747, 2, !dbg !27
  br i1 %.not398, label %if_end261, label %if_then260, !dbg !27, !prof !33

handle_init254:                                   ; preds = %if_then252
  %748 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !27, !tbaa !30
  %749 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !27, !tbaa !30
  %750 = call i32 %749(ptr %748, ptr nonnull @.str.16, ptr nonnull %7), !dbg !27
  %751 = icmp eq i32 %750, 0, !dbg !27
  br i1 %751, label %call_end257, label %common.ret, !dbg !27, !prof !29

handle_init_end255:                               ; preds = %call_end257, %if_then252
  %752 = phi ptr [ %745, %if_then252 ], [ %755, %call_end257 ], !dbg !27
  %753 = call i32 %744(ptr %752, ptr nonnull %stack_ffi_any375, i32 5, ptr nonnull %743), !dbg !27
  %754 = icmp eq i32 %753, 0, !dbg !27
  br i1 %754, label %if_end253, label %common.ret, !dbg !27, !prof !29

call_end257:                                      ; preds = %handle_init254
  %755 = load ptr, ptr %7, align 8, !dbg !27
  store ptr %755, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !27
  br label %handle_init_end255, !dbg !27

if_then260:                                       ; preds = %if_end253
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any375, align 8, !dbg !27
  %756 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 0, i32 2, !dbg !27
  store ptr @.str.9, ptr %756, align 8, !dbg !27
  %757 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %757, align 8, !dbg !27
  %758 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 2, !dbg !27
  store ptr @.str.13, ptr %758, align 8, !dbg !27
  %759 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %759, align 8, !dbg !27
  %760 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 2, !dbg !27
  store i64 2, ptr %760, align 8, !dbg !27
  %761 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %761, align 8, !dbg !27
  %762 = load i32, ptr %746, align 4, !dbg !27
  %763 = sext i32 %762 to i64, !dbg !27
  %764 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 2, !dbg !27
  store i64 %763, ptr %764, align 8, !dbg !27
  %765 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 0, !dbg !27
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %765, i8 0, i64 16, i1 false), !dbg !27
  %766 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !27, !tbaa !30
  %767 = load ptr, ptr @.tvm_func.__tvm_error_device_type_mismatch, align 8, !dbg !27
  %.not401 = icmp eq ptr %767, null, !dbg !27
  br i1 %.not401, label %handle_init262, label %handle_init_end263, !dbg !27, !prof !33

if_end261:                                        ; preds = %handle_init_end263, %if_end253
  %768 = icmp eq ptr %C, null, !dbg !27
  br i1 %768, label %if_then268, label %if_end269, !dbg !27, !prof !29

handle_init262:                                   ; preds = %if_then260
  %769 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !27, !tbaa !30
  %770 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !27, !tbaa !30
  %771 = call i32 %770(ptr %769, ptr nonnull @.str.21, ptr nonnull %6), !dbg !27
  %772 = icmp eq i32 %771, 0, !dbg !27
  br i1 %772, label %call_end265, label %common.ret, !dbg !27, !prof !29

handle_init_end263:                               ; preds = %call_end265, %if_then260
  %773 = phi ptr [ %767, %if_then260 ], [ %776, %call_end265 ], !dbg !27
  %774 = call i32 %766(ptr %773, ptr nonnull %stack_ffi_any375, i32 4, ptr nonnull %765), !dbg !27
  %775 = icmp eq i32 %774, 0, !dbg !27
  br i1 %775, label %if_end261, label %common.ret, !dbg !27, !prof !29

call_end265:                                      ; preds = %handle_init262
  %776 = load ptr, ptr %6, align 8, !dbg !27
  store ptr %776, ptr @.tvm_func.__tvm_error_device_type_mismatch, align 8, !dbg !27
  br label %handle_init_end263, !dbg !27

if_then268:                                       ; preds = %if_end261
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any375, align 8, !dbg !27
  %777 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 0, i32 2, !dbg !27
  store ptr @.str.9, ptr %777, align 8, !dbg !27
  %778 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %778, align 8, !dbg !27
  %779 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 2, !dbg !27
  store ptr @.str.13, ptr %779, align 8, !dbg !27
  %780 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 0, !dbg !27
  store <2 x i32> <i32 8, i32 0>, ptr %780, align 8, !dbg !27
  %781 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 2, !dbg !27
  store ptr @.str.22, ptr %781, align 8, !dbg !27
  %782 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 0, !dbg !27
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %782, i8 0, i64 16, i1 false), !dbg !27
  %783 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !27, !tbaa !30
  %784 = load ptr, ptr @.tvm_func.__tvm_error_null_ptr, align 8, !dbg !27
  %.not400 = icmp eq ptr %784, null, !dbg !27
  br i1 %.not400, label %handle_init270, label %handle_init_end271, !dbg !27, !prof !33

if_end269:                                        ; preds = %handle_init_end271, %if_end261
  store <2 x i32> <i32 1, i32 0>, ptr %stack_ffi_any375, align 8, !dbg !27
  %785 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 0, i32 2, !dbg !27
  store i64 2, ptr %785, align 8, !dbg !27
  %786 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 0, !dbg !27
  store <2 x i32> <i32 1, i32 0>, ptr %786, align 8, !dbg !27
  %787 = sext i32 %dev_id to i64, !dbg !27
  %788 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 1, i32 2, !dbg !27
  store i64 %787, ptr %788, align 8, !dbg !27
  %789 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 0, !dbg !27
  %790 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 2, i32 2, !dbg !27
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %789, i8 0, i64 16, i1 false), !dbg !27
  %791 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !27, !tbaa !30
  %792 = load ptr, ptr @.tvm_func.__tvm_set_device, align 8, !dbg !27
  %.not399 = icmp eq ptr %792, null, !dbg !27
  br i1 %.not399, label %handle_init276, label %handle_init_end277, !dbg !27, !prof !33

handle_init270:                                   ; preds = %if_then268
  %793 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !27, !tbaa !30
  %794 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !27, !tbaa !30
  %795 = call i32 %794(ptr %793, ptr nonnull @.str.23, ptr nonnull %5), !dbg !27
  %796 = icmp eq i32 %795, 0, !dbg !27
  br i1 %796, label %call_end273, label %common.ret, !dbg !27, !prof !29

handle_init_end271:                               ; preds = %call_end273, %if_then268
  %797 = phi ptr [ %784, %if_then268 ], [ %800, %call_end273 ], !dbg !27
  %798 = call i32 %783(ptr %797, ptr nonnull %stack_ffi_any375, i32 3, ptr nonnull %782), !dbg !27
  %799 = icmp eq i32 %798, 0, !dbg !27
  br i1 %799, label %if_end269, label %common.ret, !dbg !27, !prof !29

call_end273:                                      ; preds = %handle_init270
  %800 = load ptr, ptr %5, align 8, !dbg !27
  store ptr %800, ptr @.tvm_func.__tvm_error_null_ptr, align 8, !dbg !27
  br label %handle_init_end271, !dbg !27

handle_init276:                                   ; preds = %if_end269
  %801 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !27, !tbaa !30
  %802 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !27, !tbaa !30
  %803 = call i32 %802(ptr %801, ptr nonnull @.str.25, ptr nonnull %4), !dbg !27
  %804 = icmp eq i32 %803, 0, !dbg !27
  br i1 %804, label %call_end279, label %common.ret, !dbg !27, !prof !29

handle_init_end277:                               ; preds = %call_end279, %if_end269
  %805 = phi ptr [ %792, %if_end269 ], [ %808, %call_end279 ], !dbg !27
  %806 = call i32 %791(ptr %805, ptr nonnull %stack_ffi_any375, i32 2, ptr nonnull %789), !dbg !27
  %807 = icmp eq i32 %806, 0, !dbg !27
  br i1 %807, label %call_end281, label %common.ret, !dbg !27, !prof !29

call_end279:                                      ; preds = %handle_init276
  %808 = load ptr, ptr %4, align 8, !dbg !27
  store ptr %808, ptr @.tvm_func.__tvm_set_device, align 8, !dbg !27
  br label %handle_init_end277, !dbg !27

call_end281:                                      ; preds = %handle_init_end277
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %0), !dbg !15
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %1), !dbg !15
  call void @llvm.lifetime.start.p0(i64 256, ptr nonnull %A_desc56.i), !dbg !15
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %2), !dbg !15
  call void @llvm.lifetime.start.p0(i64 256, ptr nonnull %B_desc57.i), !dbg !15
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %3), !dbg !15
  call void @llvm.lifetime.start.p0(i64 256, ptr nonnull %C_desc58.i), !dbg !15
  call void @llvm.dbg.value(metadata ptr %stack_ffi_any375, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.value(metadata ptr %C, metadata !23, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.value(metadata ptr %B, metadata !24, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.value(metadata ptr %A, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata ptr %C_desc58.i, metadata !58, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata ptr %C_desc58.i, metadata !58, metadata !DIExpression()), !dbg !15
  store <2 x i32> <i32 4, i32 0>, ptr %stack_ffi_any375, align 8, !dbg !15
  store ptr %C_desc58.i, ptr %785, align 8, !dbg !15
  store <2 x i32> <i32 1, i32 0>, ptr %786, align 8, !dbg !15
  store i64 9, ptr %788, align 8, !dbg !15
  store <2 x i32> <i32 1, i32 0>, ptr %789, align 8, !dbg !15
  store i64 2, ptr %790, align 8, !dbg !15
  %spec.select.i = select i1 %768, i32 0, i32 4, !dbg !15
  %809 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 0, !dbg !15
  store i32 %spec.select.i, ptr %809, align 8, !dbg !15
  %810 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 1, !dbg !15
  store i32 0, ptr %810, align 4, !dbg !15
  %811 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 3, i32 2, !dbg !15
  store ptr %C, ptr %811, align 8, !dbg !15
  %812 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 0, !dbg !15
  store <2 x i32> <i32 1, i32 0>, ptr %812, align 8, !dbg !15
  %813 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 4, i32 2, !dbg !15
  store i64 6144, ptr %813, align 8, !dbg !15
  %814 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 5, i32 0, !dbg !15
  store <2 x i32> <i32 1, i32 0>, ptr %814, align 8, !dbg !15
  %815 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 5, i32 2, !dbg !15
  store i64 1, ptr %815, align 8, !dbg !15
  %816 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 6, i32 0, !dbg !15
  store <2 x i32> <i32 1, i32 0>, ptr %816, align 8, !dbg !15
  %817 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 6, i32 2, !dbg !15
  store i64 2, ptr %817, align 8, !dbg !15
  %818 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 7, i32 0, !dbg !15
  store <2 x i32> <i32 1, i32 0>, ptr %818, align 8, !dbg !15
  %819 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 7, i32 2, !dbg !15
  store i64 12288, ptr %819, align 8, !dbg !15
  %820 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 8, i32 0, !dbg !15
  store <2 x i32> <i32 1, i32 0>, ptr %820, align 8, !dbg !15
  %821 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 8, i32 2, !dbg !15
  store i64 64, ptr %821, align 8, !dbg !15
  %822 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 9, i32 0, !dbg !15
  store <2 x i32> <i32 1, i32 0>, ptr %822, align 8, !dbg !15
  %823 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 9, i32 2, !dbg !15
  store i64 16, ptr %823, align 8, !dbg !15
  %824 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 10, i32 0, !dbg !15
  store <2 x i32> <i32 1, i32 0>, ptr %824, align 8, !dbg !15
  %825 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 10, i32 2, !dbg !15
  store i64 1, ptr %825, align 8, !dbg !15
  %826 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 11, i32 0, !dbg !15
  store <2 x i32> <i32 1, i32 0>, ptr %826, align 8, !dbg !15
  %827 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 11, i32 2, !dbg !15
  store i64 1, ptr %827, align 8, !dbg !15
  %828 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 12, i32 0, !dbg !15
  store <2 x i32> <i32 1, i32 0>, ptr %828, align 8, !dbg !15
  %829 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 12, i32 2, !dbg !15
  store i64 0, ptr %829, align 8, !dbg !15
  %830 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 13, i32 0, !dbg !15
  store <2 x i32> <i32 1, i32 0>, ptr %830, align 8, !dbg !15
  %831 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 13, i32 2, !dbg !15
  store i64 3, ptr %831, align 8, !dbg !15
  %832 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 14, i32 0, !dbg !15
  store <2 x i32> <i32 1, i32 0>, ptr %832, align 8, !dbg !15
  %833 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 14, i32 2, !dbg !15
  store i64 2, ptr %833, align 8, !dbg !15
  %834 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 15, i32 0, !dbg !15
  store i32 1, ptr %834, align 8, !dbg !15
  %835 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 15, i32 1, !dbg !15
  %836 = getelementptr inbounds %0, ptr %stack_ffi_any375, i64 16, i32 0, !dbg !15
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(28) %835, i8 0, i64 28, i1 false), !dbg !15
  %837 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !15, !tbaa !30
  %838 = load ptr, ptr @.tvm_func.__tvm_tensormap_create_tiled, align 8, !dbg !15
  %.not.i = icmp eq ptr %838, null, !dbg !15
  br i1 %.not.i, label %handle_init.i, label %handle_init_end.i, !dbg !15, !prof !33

handle_init.i:                                    ; preds = %call_end281
  %839 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !15, !tbaa !30
  %840 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !15, !tbaa !30
  %841 = call i32 %840(ptr %839, ptr nonnull @.str.26, ptr nonnull %3), !dbg !15
  %842 = icmp eq i32 %841, 0, !dbg !15
  br i1 %842, label %call_end.i, label %linear_compute_.exit, !dbg !15, !prof !29

handle_init_end.i:                                ; preds = %call_end.i, %call_end281
  %843 = phi ptr [ %838, %call_end281 ], [ %846, %call_end.i ], !dbg !15
  %844 = call i32 %837(ptr %843, ptr nonnull %stack_ffi_any375, i32 16, ptr nonnull %836), !dbg !15
  %845 = icmp eq i32 %844, 0, !dbg !15
  br i1 %845, label %if_else8.i, label %linear_compute_.exit, !dbg !15, !prof !29

call_end.i:                                       ; preds = %handle_init.i
  %846 = load ptr, ptr %3, align 8, !dbg !15
  store ptr %846, ptr @.tvm_func.__tvm_tensormap_create_tiled, align 8, !dbg !15
  br label %handle_init_end.i, !dbg !15

if_else8.i:                                       ; preds = %handle_init_end.i
  call void @llvm.dbg.declare(metadata ptr %B_desc57.i, metadata !61, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata ptr %B_desc57.i, metadata !61, metadata !DIExpression()), !dbg !15
  store <2 x i32> <i32 4, i32 0>, ptr %stack_ffi_any375, align 8, !dbg !15
  store ptr %B_desc57.i, ptr %785, align 8, !dbg !15
  store <2 x i32> <i32 1, i32 0>, ptr %786, align 8, !dbg !15
  store i64 9, ptr %788, align 8, !dbg !15
  store <2 x i32> <i32 1, i32 0>, ptr %789, align 8, !dbg !15
  store i64 2, ptr %790, align 8, !dbg !15
  %spec.select64.i = select i1 %538, i32 0, i32 4, !dbg !15
  store i32 %spec.select64.i, ptr %809, align 8, !dbg !15
  store i32 0, ptr %810, align 4, !dbg !15
  store ptr %B, ptr %811, align 8, !dbg !15
  store <2 x i32> <i32 1, i32 0>, ptr %812, align 8, !dbg !15
  store i64 1024, ptr %813, align 8, !dbg !15
  store <2 x i32> <i32 1, i32 0>, ptr %814, align 8, !dbg !15
  store i64 6144, ptr %815, align 8, !dbg !15
  store <2 x i32> <i32 1, i32 0>, ptr %816, align 8, !dbg !15
  store i64 2, ptr %817, align 8, !dbg !15
  store <2 x i32> <i32 1, i32 0>, ptr %818, align 8, !dbg !15
  store i64 2048, ptr %819, align 8, !dbg !15
  store <2 x i32> <i32 1, i32 0>, ptr %820, align 8, !dbg !15
  store i64 64, ptr %821, align 8, !dbg !15
  store <2 x i32> <i32 1, i32 0>, ptr %822, align 8, !dbg !15
  store i64 64, ptr %823, align 8, !dbg !15
  store <2 x i32> <i32 1, i32 0>, ptr %824, align 8, !dbg !15
  store i64 1, ptr %825, align 8, !dbg !15
  store <2 x i32> <i32 1, i32 0>, ptr %826, align 8, !dbg !15
  store i64 1, ptr %827, align 8, !dbg !15
  store <2 x i32> <i32 1, i32 0>, ptr %828, align 8, !dbg !15
  store i64 0, ptr %829, align 8, !dbg !15
  store <2 x i32> <i32 1, i32 0>, ptr %830, align 8, !dbg !15
  store i64 3, ptr %831, align 8, !dbg !15
  store <2 x i32> <i32 1, i32 0>, ptr %832, align 8, !dbg !15
  store i64 2, ptr %833, align 8, !dbg !15
  store i32 1, ptr %834, align 8, !dbg !15
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(28) %835, i8 0, i64 28, i1 false), !dbg !15
  %847 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !15, !tbaa !30
  %848 = load ptr, ptr @.tvm_func.__tvm_tensormap_create_tiled, align 8, !dbg !15
  %.not59.i = icmp eq ptr %848, null, !dbg !15
  br i1 %.not59.i, label %handle_init12.i, label %handle_init_end13.i, !dbg !15, !prof !33

handle_init12.i:                                  ; preds = %if_else8.i
  %849 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !15, !tbaa !30
  %850 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !15, !tbaa !30
  %851 = call i32 %850(ptr %849, ptr nonnull @.str.26, ptr nonnull %2), !dbg !15
  %852 = icmp eq i32 %851, 0, !dbg !15
  br i1 %852, label %call_end15.i, label %linear_compute_.exit, !dbg !15, !prof !29

handle_init_end13.i:                              ; preds = %call_end15.i, %if_else8.i
  %853 = phi ptr [ %848, %if_else8.i ], [ %856, %call_end15.i ], !dbg !15
  %854 = call i32 %847(ptr %853, ptr nonnull %stack_ffi_any375, i32 16, ptr nonnull %836), !dbg !15
  %855 = icmp eq i32 %854, 0, !dbg !15
  br i1 %855, label %if_else20.i, label %linear_compute_.exit, !dbg !15, !prof !29

call_end15.i:                                     ; preds = %handle_init12.i
  %856 = load ptr, ptr %2, align 8, !dbg !15
  store ptr %856, ptr @.tvm_func.__tvm_tensormap_create_tiled, align 8, !dbg !15
  br label %handle_init_end13.i, !dbg !15

if_else20.i:                                      ; preds = %handle_init_end13.i
  call void @llvm.dbg.declare(metadata ptr %A_desc56.i, metadata !62, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata ptr %A_desc56.i, metadata !62, metadata !DIExpression()), !dbg !15
  store <2 x i32> <i32 4, i32 0>, ptr %stack_ffi_any375, align 8, !dbg !15
  store ptr %A_desc56.i, ptr %785, align 8, !dbg !15
  store <2 x i32> <i32 1, i32 0>, ptr %786, align 8, !dbg !15
  store i64 9, ptr %788, align 8, !dbg !15
  store <2 x i32> <i32 1, i32 0>, ptr %789, align 8, !dbg !15
  store i64 2, ptr %790, align 8, !dbg !15
  %spec.select65.i = select i1 %308, i32 0, i32 4, !dbg !15
  store i32 %spec.select65.i, ptr %809, align 8, !dbg !15
  store i32 0, ptr %810, align 4, !dbg !15
  store ptr %A, ptr %811, align 8, !dbg !15
  store <2 x i32> <i32 1, i32 0>, ptr %812, align 8, !dbg !15
  store i64 1024, ptr %813, align 8, !dbg !15
  store <2 x i32> <i32 1, i32 0>, ptr %814, align 8, !dbg !15
  store i64 1, ptr %815, align 8, !dbg !15
  store <2 x i32> <i32 1, i32 0>, ptr %816, align 8, !dbg !15
  store i64 2, ptr %817, align 8, !dbg !15
  store <2 x i32> <i32 1, i32 0>, ptr %818, align 8, !dbg !15
  store i64 2048, ptr %819, align 8, !dbg !15
  store <2 x i32> <i32 1, i32 0>, ptr %820, align 8, !dbg !15
  store i64 64, ptr %821, align 8, !dbg !15
  store <2 x i32> <i32 1, i32 0>, ptr %822, align 8, !dbg !15
  store i64 16, ptr %823, align 8, !dbg !15
  store <2 x i32> <i32 1, i32 0>, ptr %824, align 8, !dbg !15
  store i64 1, ptr %825, align 8, !dbg !15
  store <2 x i32> <i32 1, i32 0>, ptr %826, align 8, !dbg !15
  store i64 1, ptr %827, align 8, !dbg !15
  store <2 x i32> <i32 1, i32 0>, ptr %828, align 8, !dbg !15
  store i64 0, ptr %829, align 8, !dbg !15
  store <2 x i32> <i32 1, i32 0>, ptr %830, align 8, !dbg !15
  store i64 3, ptr %831, align 8, !dbg !15
  store <2 x i32> <i32 1, i32 0>, ptr %832, align 8, !dbg !15
  store i64 2, ptr %833, align 8, !dbg !15
  store i32 1, ptr %834, align 8, !dbg !15
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(28) %835, i8 0, i64 28, i1 false), !dbg !15
  %857 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !15, !tbaa !30
  %858 = load ptr, ptr @.tvm_func.__tvm_tensormap_create_tiled, align 8, !dbg !15
  %.not60.i = icmp eq ptr %858, null, !dbg !15
  br i1 %.not60.i, label %handle_init24.i, label %handle_init_end25.i, !dbg !15, !prof !33

handle_init24.i:                                  ; preds = %if_else20.i
  %859 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !15, !tbaa !30
  %860 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !15, !tbaa !30
  %861 = call i32 %860(ptr %859, ptr nonnull @.str.26, ptr nonnull %1), !dbg !15
  %862 = icmp eq i32 %861, 0, !dbg !15
  br i1 %862, label %call_end27.i, label %linear_compute_.exit, !dbg !15, !prof !29

handle_init_end25.i:                              ; preds = %call_end27.i, %if_else20.i
  %863 = phi ptr [ %858, %if_else20.i ], [ %866, %call_end27.i ], !dbg !15
  %864 = call i32 %857(ptr %863, ptr nonnull %stack_ffi_any375, i32 16, ptr nonnull %836), !dbg !15
  %865 = icmp eq i32 %864, 0, !dbg !15
  br i1 %865, label %if_else32.i, label %linear_compute_.exit, !dbg !15, !prof !29

call_end27.i:                                     ; preds = %handle_init24.i
  %866 = load ptr, ptr %1, align 8, !dbg !15
  store ptr %866, ptr @.tvm_func.__tvm_tensormap_create_tiled, align 8, !dbg !15
  br label %handle_init_end25.i, !dbg !15

if_else32.i:                                      ; preds = %handle_init_end25.i
  store <2 x i32> <i32 4, i32 0>, ptr %stack_ffi_any375, align 8, !dbg !15
  store ptr %A_desc56.i, ptr %785, align 8, !dbg !15
  store <2 x i32> <i32 4, i32 0>, ptr %786, align 8, !dbg !15
  store ptr %B_desc57.i, ptr %788, align 8, !dbg !15
  store <2 x i32> <i32 4, i32 0>, ptr %789, align 8, !dbg !15
  store ptr %C_desc58.i, ptr %790, align 8, !dbg !15
  store <2 x i32> <i32 1, i32 0>, ptr %809, align 8, !dbg !15
  store i64 96, ptr %811, align 8, !dbg !15
  store <2 x i32> <i32 1, i32 0>, ptr %812, align 8, !dbg !15
  store i64 1, ptr %813, align 8, !dbg !15
  store <2 x i32> <i32 1, i32 0>, ptr %814, align 8, !dbg !15
  store i64 128, ptr %815, align 8, !dbg !15
  store <2 x i32> <i32 1, i32 0>, ptr %816, align 8, !dbg !15
  store i64 1, ptr %817, align 8, !dbg !15
  store <2 x i32> <i32 1, i32 0>, ptr %818, align 8, !dbg !15
  store i64 1, ptr %819, align 8, !dbg !15
  store <2 x i32> <i32 1, i32 0>, ptr %820, align 8, !dbg !15
  store i64 30720, ptr %821, align 8, !dbg !15
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %822, i8 0, i64 16, i1 false), !dbg !15
  %867 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !15, !tbaa !30
  %868 = load ptr, ptr @.tvm_func.linear_kernel, align 8, !dbg !15
  %.not61.i = icmp eq ptr %868, null, !dbg !15
  br i1 %.not61.i, label %handle_init39.i, label %handle_init_end40.i, !dbg !15, !prof !33

handle_init39.i:                                  ; preds = %if_else32.i
  %869 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !15, !tbaa !30
  %870 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !15, !tbaa !30
  %871 = call i32 %870(ptr %869, ptr nonnull @.str.27, ptr nonnull %0), !dbg !15
  %872 = icmp eq i32 %871, 0, !dbg !15
  br i1 %872, label %call_end42.i, label %linear_compute_.exit, !dbg !15, !prof !29

handle_init_end40.i:                              ; preds = %call_end42.i, %if_else32.i
  %873 = phi ptr [ %868, %if_else32.i ], [ %875, %call_end42.i ], !dbg !15
  %874 = call i32 %867(ptr %873, ptr nonnull %stack_ffi_any375, i32 9, ptr nonnull %822), !dbg !15
  br label %linear_compute_.exit, !dbg !15

call_end42.i:                                     ; preds = %handle_init39.i
  %875 = load ptr, ptr %0, align 8, !dbg !15
  store ptr %875, ptr @.tvm_func.linear_kernel, align 8, !dbg !15
  br label %handle_init_end40.i, !dbg !15

linear_compute_.exit:                             ; preds = %handle_init.i, %handle_init_end.i, %handle_init12.i, %handle_init_end13.i, %handle_init24.i, %handle_init_end25.i, %handle_init39.i, %handle_init_end40.i
  %common.ret.op.i = phi i32 [ %841, %handle_init.i ], [ %844, %handle_init_end.i ], [ %851, %handle_init12.i ], [ %854, %handle_init_end13.i ], [ %861, %handle_init24.i ], [ %864, %handle_init_end25.i ], [ %871, %handle_init39.i ], [ %874, %handle_init_end40.i ]
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %0), !dbg !15
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %1), !dbg !15
  call void @llvm.lifetime.end.p0(i64 256, ptr nonnull %A_desc56.i), !dbg !15
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %2), !dbg !15
  call void @llvm.lifetime.end.p0(i64 256, ptr nonnull %B_desc57.i), !dbg !15
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %3), !dbg !15
  call void @llvm.lifetime.end.p0(i64 256, ptr nonnull %C_desc58.i), !dbg !15
  br label %common.ret
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: inaccessiblememonly mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.assume(i1 noundef) #2

define weak dllexport i32 @__tvm_ffi_main(ptr %0, ptr %1, i32 %2, ptr %3) local_unnamed_addr {
entry:
  %4 = tail call i32 @linear(ptr poison, ptr %1, i32 %2, ptr poison), !dbg !27
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
!5 = distinct !DISubprogram(name: "linear", scope: !1, file: !1, type: !6, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !10)
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
!16 = distinct !DISubprogram(name: "linear_compute_", scope: !1, file: !1, type: !17, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !21)
!17 = !DISubroutineType(types: !18)
!18 = !{!8, !9, !19, !19, !19}
!19 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !20)
!20 = !DIBasicType(name: "uint16", size: 16, encoding: DW_ATE_unsigned)
!21 = !{!22, !23, !24, !25}
!22 = !DILocalVariable(name: "stack_ffi_any", arg: 1, scope: !16, file: !1, type: !9)
!23 = !DILocalVariable(name: "C", arg: 2, scope: !16, file: !1, type: !19)
!24 = !DILocalVariable(name: "B", arg: 3, scope: !16, file: !1, type: !19)
!25 = !DILocalVariable(name: "A", arg: 4, scope: !16, file: !1, type: !19)
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
!40 = !DILocalVariable(name: "linear.A_is_null", scope: !5, file: !1, type: !41)
!41 = !DIBasicType(name: "bool", size: 8, encoding: DW_ATE_boolean)
!42 = !DILocalVariable(name: "linear.B_is_null", scope: !5, file: !1, type: !41)
!43 = !DILocalVariable(name: "linear.C_is_null", scope: !5, file: !1, type: !41)
!44 = !DILocalVariable(name: "linear.A.shape", scope: !5, file: !1, type: !45)
!45 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !46)
!46 = !DIBasicType(name: "int64", size: 64, encoding: DW_ATE_signed)
!47 = !DILocalVariable(name: "linear.B.shape", scope: !5, file: !1, type: !45)
!48 = !DILocalVariable(name: "linear.C.shape", scope: !5, file: !1, type: !45)
!49 = !DILocalVariable(name: "linear.A.strides", scope: !5, file: !1, type: !45)
!50 = !DILocalVariable(name: "dev_id", scope: !5, file: !1, type: !8)
!51 = !DILocalVariable(name: "A", scope: !5, file: !1, type: !19)
!52 = !DILocalVariable(name: "linear.B.strides", scope: !5, file: !1, type: !45)
!53 = !DILocalVariable(name: "B", scope: !5, file: !1, type: !19)
!54 = !DILocalVariable(name: "linear.C.strides", scope: !5, file: !1, type: !45)
!55 = !DILocalVariable(name: "C", scope: !5, file: !1, type: !19)
!56 = !{!57, !57, i64 0}
!57 = !{!"tvm-alias", !32}
!58 = !DILocalVariable(name: "C_desc", scope: !16, file: !1, type: !59)
!59 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !60)
!60 = !DIBasicType(name: "uint8x128", size: 1024, encoding: DW_ATE_unsigned)
!61 = !DILocalVariable(name: "B_desc", scope: !16, file: !1, type: !59)
!62 = !DILocalVariable(name: "A_desc", scope: !16, file: !1, type: !59)
*/

// latency: 0 ms vs [ref-0 sim-0], idx: 28