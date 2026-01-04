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
@.str = private constant [80 x i8] c"Assert fail: num_args == 7, flashattn_gqa_decode_no_split: num_args should be 7\00", align 1
@.str.1 = private constant [13 x i8] c"RuntimeError\00", align 1
@.str.2 = private constant [88 x i8] c"Assert fail: not T.isnullptr(args), flashattn_gqa_decode_no_split: args pointer is NULL\00", align 1
@.str.3 = private constant [203 x i8] c"Assert fail: Q_handle_type_index == 0 or Q_handle_type_index == 4 or Q_handle_type_index == 7 or 64 <= Q_handle_type_index, kernel flashattn_gqa_decode_no_split input Q expected pointer or tensor handle\00", align 1
@.str.4 = private constant [203 x i8] c"Assert fail: K_handle_type_index == 0 or K_handle_type_index == 4 or K_handle_type_index == 7 or 64 <= K_handle_type_index, kernel flashattn_gqa_decode_no_split input K expected pointer or tensor handle\00", align 1
@.str.5 = private constant [203 x i8] c"Assert fail: V_handle_type_index == 0 or V_handle_type_index == 4 or V_handle_type_index == 7 or 64 <= V_handle_type_index, kernel flashattn_gqa_decode_no_split input V expected pointer or tensor handle\00", align 1
@.str.6 = private constant [218 x i8] c"Assert fail: mask_handle_type_index == 0 or mask_handle_type_index == 4 or mask_handle_type_index == 7 or 64 <= mask_handle_type_index, kernel flashattn_gqa_decode_no_split input mask expected pointer or tensor handle\00", align 1
@.str.7 = private constant [218 x i8] c"Assert fail: glse_handle_type_index == 0 or glse_handle_type_index == 4 or glse_handle_type_index == 7 or 64 <= glse_handle_type_index, kernel flashattn_gqa_decode_no_split input glse expected pointer or tensor handle\00", align 1
@.str.8 = private constant [268 x i8] c"Assert fail: Output_partial_handle_type_index == 0 or Output_partial_handle_type_index == 4 or Output_partial_handle_type_index == 7 or 64 <= Output_partial_handle_type_index, kernel flashattn_gqa_decode_no_split input Output_partial expected pointer or tensor handle\00", align 1
@.str.9 = private constant [228 x i8] c"Assert fail: Output_handle_type_index == 0 or Output_handle_type_index == 4 or Output_handle_type_index == 7 or 64 <= Output_handle_type_index, kernel flashattn_gqa_decode_no_split input Output expected pointer or tensor handle\00", align 1
@.str.10 = private constant [127 x i8] c"Assert fail: not flashattn_gqa_decode_no_split_Q_is_null, flashattn_gqa_decode_no_split.Q is expected to have non-NULL pointer\00", align 1
@.str.11 = private constant [127 x i8] c"Assert fail: not flashattn_gqa_decode_no_split_K_is_null, flashattn_gqa_decode_no_split.K is expected to have non-NULL pointer\00", align 1
@.str.12 = private constant [127 x i8] c"Assert fail: not flashattn_gqa_decode_no_split_V_is_null, flashattn_gqa_decode_no_split.V is expected to have non-NULL pointer\00", align 1
@.str.13 = private constant [133 x i8] c"Assert fail: not flashattn_gqa_decode_no_split_mask_is_null, flashattn_gqa_decode_no_split.mask is expected to have non-NULL pointer\00", align 1
@.str.14 = private constant [137 x i8] c"Assert fail: not flashattn_gqa_decode_no_split_Output_is_null, flashattn_gqa_decode_no_split.Output is expected to have non-NULL pointer\00", align 1
@.str.15 = private constant [30 x i8] c"flashattn_gqa_decode_no_split\00", align 1
@.str.16 = private constant [2 x i8] c"Q\00", align 1
@.tvm_func.__tvm_error_ndim_mismatch = internal unnamed_addr global ptr null, align 8
@.str.17 = private constant [26 x i8] c"__tvm_error_ndim_mismatch\00", align 1
@.str.18 = private constant [2 x i8] c"K\00", align 1
@.str.19 = private constant [2 x i8] c"V\00", align 1
@.str.20 = private constant [5 x i8] c"mask\00", align 1
@.str.21 = private constant [5 x i8] c"glse\00", align 1
@.str.22 = private constant [15 x i8] c"Output_partial\00", align 1
@.str.23 = private constant [7 x i8] c"Output\00", align 1
@.tvm_func.__tvm_error_dtype_mismatch = internal unnamed_addr global ptr null, align 8
@.str.24 = private constant [27 x i8] c"__tvm_error_dtype_mismatch\00", align 1
@.str.25 = private constant [9 x i8] c"shape[0]\00", align 1
@.tvm_func.__tvm_error_expect_eq = internal unnamed_addr global ptr null, align 8
@.str.26 = private constant [22 x i8] c"__tvm_error_expect_eq\00", align 1
@.str.27 = private constant [9 x i8] c"shape[1]\00", align 1
@.str.28 = private constant [9 x i8] c"shape[2]\00", align 1
@.str.29 = private constant [11 x i8] c"strides[2]\00", align 1
@.str.30 = private constant [11 x i8] c"strides[1]\00", align 1
@.str.31 = private constant [11 x i8] c"strides[0]\00", align 1
@.tvm_func.__tvm_error_byte_offset_mismatch = internal unnamed_addr global ptr null, align 8
@.str.32 = private constant [33 x i8] c"__tvm_error_byte_offset_mismatch\00", align 1
@.tvm_func.__tvm_error_device_type_mismatch = internal unnamed_addr global ptr null, align 8
@.str.33 = private constant [33 x i8] c"__tvm_error_device_type_mismatch\00", align 1
@.str.34 = private constant [13 x i8] c"data pointer\00", align 1
@.tvm_func.__tvm_error_null_ptr = internal unnamed_addr global ptr null, align 8
@.str.35 = private constant [21 x i8] c"__tvm_error_null_ptr\00", align 1
@.str.36 = private constant [9 x i8] c"shape[3]\00", align 1
@.str.37 = private constant [11 x i8] c"strides[3]\00", align 1
@.str.38 = private constant [10 x i8] c"device_id\00", align 1
@.tvm_func.__tvm_set_device = internal unnamed_addr global ptr null, align 8
@.str.39 = private constant [17 x i8] c"__tvm_set_device\00", align 1
@.tvm_func.flashattn_gqa_decode_no_split_kernel = internal unnamed_addr global ptr null, align 8
@.str.40 = private constant [37 x i8] c"flashattn_gqa_decode_no_split_kernel\00", align 1
@llvm.global_ctors = appending global [0 x { i32, ptr, ptr }] zeroinitializer

define dllexport i32 @flashattn_gqa_decode_no_split(ptr nocapture readnone %self_handle, ptr readonly %args, i32 %num_args, ptr nocapture readnone %result) local_unnamed_addr #0 !dbg !5 {
entry:
  %0 = alloca ptr, align 8, !dbg !15
  call void @llvm.dbg.value(metadata ptr poison, metadata !11, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.value(metadata ptr %args, metadata !12, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.value(metadata i32 %num_args, metadata !13, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.value(metadata ptr poison, metadata !14, metadata !DIExpression()), !dbg !31
  %1 = alloca ptr, align 8, !dbg !31
  %2 = alloca ptr, align 8, !dbg !31
  %3 = alloca ptr, align 8, !dbg !31
  %4 = alloca ptr, align 8, !dbg !31
  %5 = alloca ptr, align 8, !dbg !31
  %6 = alloca ptr, align 8, !dbg !31
  %7 = alloca ptr, align 8, !dbg !31
  %8 = alloca ptr, align 8, !dbg !31
  %9 = alloca ptr, align 8, !dbg !31
  %10 = alloca ptr, align 8, !dbg !31
  %11 = alloca ptr, align 8, !dbg !31
  %12 = alloca ptr, align 8, !dbg !31
  %13 = alloca ptr, align 8, !dbg !31
  %14 = alloca ptr, align 8, !dbg !31
  %15 = alloca ptr, align 8, !dbg !31
  %16 = alloca ptr, align 8, !dbg !31
  %17 = alloca ptr, align 8, !dbg !31
  %18 = alloca ptr, align 8, !dbg !31
  %19 = alloca ptr, align 8, !dbg !31
  %20 = alloca ptr, align 8, !dbg !31
  %21 = alloca ptr, align 8, !dbg !31
  %22 = alloca ptr, align 8, !dbg !31
  %23 = alloca ptr, align 8, !dbg !31
  %24 = alloca ptr, align 8, !dbg !31
  %25 = alloca ptr, align 8, !dbg !31
  %26 = alloca ptr, align 8, !dbg !31
  %27 = alloca ptr, align 8, !dbg !31
  %28 = alloca ptr, align 8, !dbg !31
  %29 = alloca ptr, align 8, !dbg !31
  %30 = alloca ptr, align 8, !dbg !31
  %31 = alloca ptr, align 8, !dbg !31
  %32 = alloca ptr, align 8, !dbg !31
  %33 = alloca ptr, align 8, !dbg !31
  %34 = alloca ptr, align 8, !dbg !31
  %35 = alloca ptr, align 8, !dbg !31
  %36 = alloca ptr, align 8, !dbg !31
  %37 = alloca ptr, align 8, !dbg !31
  %38 = alloca ptr, align 8, !dbg !31
  %39 = alloca ptr, align 8, !dbg !31
  %40 = alloca ptr, align 8, !dbg !31
  %41 = alloca ptr, align 8, !dbg !31
  %42 = alloca ptr, align 8, !dbg !31
  %43 = alloca ptr, align 8, !dbg !31
  %44 = alloca ptr, align 8, !dbg !31
  %45 = alloca ptr, align 8, !dbg !31
  %46 = alloca ptr, align 8, !dbg !31
  %47 = alloca ptr, align 8, !dbg !31
  %48 = alloca ptr, align 8, !dbg !31
  %49 = alloca ptr, align 8, !dbg !31
  %50 = alloca ptr, align 8, !dbg !31
  %51 = alloca ptr, align 8, !dbg !31
  %52 = alloca ptr, align 8, !dbg !31
  %53 = alloca ptr, align 8, !dbg !31
  %54 = alloca ptr, align 8, !dbg !31
  %55 = alloca ptr, align 8, !dbg !31
  %56 = alloca ptr, align 8, !dbg !31
  %57 = alloca ptr, align 8, !dbg !31
  %58 = alloca ptr, align 8, !dbg !31
  %59 = alloca ptr, align 8, !dbg !31
  %60 = alloca ptr, align 8, !dbg !31
  %61 = alloca ptr, align 8, !dbg !31
  %62 = alloca ptr, align 8, !dbg !31
  %63 = alloca ptr, align 8, !dbg !31
  %64 = alloca ptr, align 8, !dbg !31
  %65 = alloca ptr, align 8, !dbg !31
  %66 = alloca ptr, align 8, !dbg !31
  %67 = alloca ptr, align 8, !dbg !31
  %68 = alloca ptr, align 8, !dbg !31
  %69 = alloca ptr, align 8, !dbg !31
  %70 = alloca ptr, align 8, !dbg !31
  %71 = alloca ptr, align 8, !dbg !31
  %72 = alloca ptr, align 8, !dbg !31
  %73 = alloca ptr, align 8, !dbg !31
  %74 = alloca ptr, align 8, !dbg !31
  %75 = alloca ptr, align 8, !dbg !31
  %76 = alloca ptr, align 8, !dbg !31
  %77 = alloca ptr, align 8, !dbg !31
  %78 = alloca ptr, align 8, !dbg !31
  %79 = alloca ptr, align 8, !dbg !31
  %80 = alloca ptr, align 8, !dbg !31
  %81 = alloca ptr, align 8, !dbg !31
  %82 = alloca ptr, align 8, !dbg !31
  %83 = alloca ptr, align 8, !dbg !31
  %84 = alloca ptr, align 8, !dbg !31
  %85 = alloca ptr, align 8, !dbg !31
  %86 = alloca ptr, align 8, !dbg !31
  %87 = alloca ptr, align 8, !dbg !31
  %88 = alloca ptr, align 8, !dbg !31
  %89 = alloca ptr, align 8, !dbg !31
  %90 = alloca ptr, align 8, !dbg !31
  %stack_ffi_any1228 = alloca [13 x %0], align 8, !dbg !31
  call void @llvm.dbg.declare(metadata ptr %stack_ffi_any1228, metadata !32, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.declare(metadata ptr %stack_ffi_any1228, metadata !32, metadata !DIExpression()), !dbg !31
  %91 = icmp eq i32 %num_args, 7, !dbg !31
  br i1 %91, label %assert_end, label %assert_fail, !dbg !31, !prof !33

common.ret:                                       ; preds = %flashattn_gqa_decode_no_split_compute_.exit, %handle_init_end945, %handle_init944, %handle_init_end939, %handle_init938, %handle_init_end931, %handle_init930, %handle_init_end923, %handle_init922, %handle_init_end915, %handle_init914, %handle_init_end907, %handle_init906, %handle_init_end893, %handle_init892, %handle_init_end879, %handle_init878, %handle_init_end865, %handle_init864, %handle_init_end857, %handle_init856, %handle_init_end849, %handle_init848, %handle_init_end841, %handle_init840, %handle_init_end833, %handle_init832, %handle_init_end822, %handle_init821, %handle_init_end811, %handle_init810, %handle_init_end801, %handle_init800, %handle_init_end790, %handle_init789, %handle_init_end776, %handle_init775, %handle_init_end762, %handle_init761, %handle_init_end748, %handle_init747, %handle_init_end734, %handle_init733, %handle_init_end726, %handle_init725, %handle_init_end718, %handle_init717, %handle_init_end710, %handle_init709, %handle_init_end700, %handle_init699, %handle_init_end683, %handle_init682, %handle_init_end672, %handle_init671, %handle_init_end661, %handle_init660, %handle_init_end651, %handle_init650, %handle_init_end640, %handle_init639, %handle_init_end626, %handle_init625, %handle_init_end612, %handle_init611, %handle_init_end598, %handle_init597, %handle_init_end590, %handle_init589, %handle_init_end582, %handle_init581, %handle_init_end572, %handle_init571, %handle_init_end555, %handle_init554, %handle_init_end547, %handle_init546, %handle_init_end539, %handle_init538, %handle_init_end531, %handle_init530, %handle_init_end523, %handle_init522, %handle_init_end509, %handle_init508, %handle_init_end495, %handle_init494, %handle_init_end481, %handle_init480, %handle_init_end473, %handle_init472, %handle_init_end465, %handle_init464, %handle_init_end457, %handle_init456, %handle_init_end449, %handle_init448, %handle_init_end441, %handle_init440, %handle_init_end433, %handle_init432, %handle_init_end425, %handle_init424, %handle_init_end417, %handle_init416, %handle_init_end403, %handle_init402, %handle_init_end389, %handle_init388, %handle_init_end375, %handle_init374, %handle_init_end361, %handle_init360, %handle_init_end353, %handle_init352, %handle_init_end345, %handle_init344, %handle_init_end337, %handle_init336, %handle_init_end329, %handle_init328, %handle_init_end321, %handle_init320, %handle_init_end313, %handle_init312, %handle_init_end305, %handle_init304, %handle_init_end297, %handle_init296, %handle_init_end289, %handle_init288, %handle_init_end275, %handle_init274, %handle_init_end261, %handle_init260, %handle_init_end247, %handle_init246, %handle_init_end233, %handle_init232, %handle_init_end225, %handle_init224, %handle_init_end217, %handle_init216, %handle_init_end209, %handle_init208, %handle_init_end201, %handle_init200, %handle_init_end193, %handle_init192, %handle_init_end185, %handle_init184, %handle_init_end177, %handle_init176, %handle_init_end169, %handle_init168, %handle_init_end155, %handle_init154, %handle_init_end141, %handle_init140, %handle_init_end127, %handle_init126, %handle_init_end119, %handle_init118, %handle_init_end111, %handle_init110, %handle_init_end103, %handle_init102, %handle_init_end95, %handle_init94, %handle_init_end81, %handle_init80, %handle_init_end64, %handle_init63, %handle_init_end53, %handle_init52, %handle_init_end45, %handle_init44, %handle_init_end37, %handle_init36, %handle_init_end, %handle_init, %assert_fail25, %assert_fail23, %assert_fail21, %assert_fail19, %assert_fail17, %assert_fail15, %assert_fail13, %assert_fail11, %assert_fail9, %assert_fail7, %assert_fail5, %assert_fail3, %assert_fail1, %assert_fail
  %common.ret.op = phi i32 [ -1, %assert_fail ], [ -1, %assert_fail1 ], [ -1, %assert_fail3 ], [ -1, %assert_fail5 ], [ -1, %assert_fail7 ], [ -1, %assert_fail9 ], [ -1, %assert_fail11 ], [ -1, %assert_fail13 ], [ -1, %assert_fail15 ], [ -1, %assert_fail17 ], [ -1, %assert_fail19 ], [ -1, %assert_fail21 ], [ -1, %assert_fail23 ], [ -1, %assert_fail25 ], [ %169, %handle_init ], [ %172, %handle_init_end ], [ %192, %handle_init36 ], [ %195, %handle_init_end37 ], [ %215, %handle_init44 ], [ %218, %handle_init_end45 ], [ %236, %handle_init52 ], [ %239, %handle_init_end53 ], [ %258, %handle_init63 ], [ %261, %handle_init_end64 ], [ %283, %handle_init80 ], [ %286, %handle_init_end81 ], [ %320, %handle_init94 ], [ %323, %handle_init_end95 ], [ %354, %handle_init102 ], [ %357, %handle_init_end103 ], [ %379, %handle_init110 ], [ %382, %handle_init_end111 ], [ %404, %handle_init118 ], [ %407, %handle_init_end119 ], [ %427, %handle_init126 ], [ %430, %handle_init_end127 ], [ %452, %handle_init140 ], [ %455, %handle_init_end141 ], [ %479, %handle_init154 ], [ %482, %handle_init_end155 ], [ %506, %handle_init168 ], [ %509, %handle_init_end169 ], [ %527, %handle_init176 ], [ %530, %handle_init_end177 ], [ %548, %handle_init184 ], [ %551, %handle_init_end185 ], [ %575, %handle_init192 ], [ %578, %handle_init_end193 ], [ %609, %handle_init200 ], [ %612, %handle_init_end201 ], [ %634, %handle_init208 ], [ %637, %handle_init_end209 ], [ %659, %handle_init216 ], [ %662, %handle_init_end217 ], [ %684, %handle_init224 ], [ %687, %handle_init_end225 ], [ %707, %handle_init232 ], [ %710, %handle_init_end233 ], [ %732, %handle_init246 ], [ %735, %handle_init_end247 ], [ %759, %handle_init260 ], [ %762, %handle_init_end261 ], [ %786, %handle_init274 ], [ %789, %handle_init_end275 ], [ %813, %handle_init288 ], [ %816, %handle_init_end289 ], [ %835, %handle_init296 ], [ %838, %handle_init_end297 ], [ %861, %handle_init304 ], [ %864, %handle_init_end305 ], [ %882, %handle_init312 ], [ %885, %handle_init_end313 ], [ %909, %handle_init320 ], [ %912, %handle_init_end321 ], [ %943, %handle_init328 ], [ %946, %handle_init_end329 ], [ %968, %handle_init336 ], [ %971, %handle_init_end337 ], [ %993, %handle_init344 ], [ %996, %handle_init_end345 ], [ %1018, %handle_init352 ], [ %1021, %handle_init_end353 ], [ %1041, %handle_init360 ], [ %1044, %handle_init_end361 ], [ %1066, %handle_init374 ], [ %1069, %handle_init_end375 ], [ %1093, %handle_init388 ], [ %1096, %handle_init_end389 ], [ %1120, %handle_init402 ], [ %1123, %handle_init_end403 ], [ %1147, %handle_init416 ], [ %1150, %handle_init_end417 ], [ %1169, %handle_init424 ], [ %1172, %handle_init_end425 ], [ %1195, %handle_init432 ], [ %1198, %handle_init_end433 ], [ %1216, %handle_init440 ], [ %1219, %handle_init_end441 ], [ %1243, %handle_init448 ], [ %1246, %handle_init_end449 ], [ %1277, %handle_init456 ], [ %1280, %handle_init_end457 ], [ %1302, %handle_init464 ], [ %1305, %handle_init_end465 ], [ %1327, %handle_init472 ], [ %1330, %handle_init_end473 ], [ %1350, %handle_init480 ], [ %1353, %handle_init_end481 ], [ %1375, %handle_init494 ], [ %1378, %handle_init_end495 ], [ %1402, %handle_init508 ], [ %1405, %handle_init_end509 ], [ %1429, %handle_init522 ], [ %1432, %handle_init_end523 ], [ %1451, %handle_init530 ], [ %1454, %handle_init_end531 ], [ %1477, %handle_init538 ], [ %1480, %handle_init_end539 ], [ %1498, %handle_init546 ], [ %1501, %handle_init_end547 ], [ %1514, %handle_init554 ], [ %1517, %handle_init_end555 ], [ %1557, %handle_init571 ], [ %1560, %handle_init_end572 ], [ %1584, %handle_init581 ], [ %1587, %handle_init_end582 ], [ %1609, %handle_init589 ], [ %1612, %handle_init_end590 ], [ %1632, %handle_init597 ], [ %1635, %handle_init_end598 ], [ %1657, %handle_init611 ], [ %1660, %handle_init_end612 ], [ %1682, %handle_init625 ], [ %1685, %handle_init_end626 ], [ %1707, %handle_init639 ], [ %1710, %handle_init_end640 ], [ %1728, %handle_init650 ], [ %1731, %handle_init_end651 ], [ %1755, %handle_init660 ], [ %1758, %handle_init_end661 ], [ %1778, %handle_init671 ], [ %1781, %handle_init_end672 ], [ %1794, %handle_init682 ], [ %1797, %handle_init_end683 ], [ %1837, %handle_init699 ], [ %1840, %handle_init_end700 ], [ %1864, %handle_init709 ], [ %1867, %handle_init_end710 ], [ %1889, %handle_init717 ], [ %1892, %handle_init_end718 ], [ %1914, %handle_init725 ], [ %1917, %handle_init_end726 ], [ %1937, %handle_init733 ], [ %1940, %handle_init_end734 ], [ %1962, %handle_init747 ], [ %1965, %handle_init_end748 ], [ %1989, %handle_init761 ], [ %1992, %handle_init_end762 ], [ %2016, %handle_init775 ], [ %2019, %handle_init_end776 ], [ %2041, %handle_init789 ], [ %2044, %handle_init_end790 ], [ %2062, %handle_init800 ], [ %2065, %handle_init_end801 ], [ %2089, %handle_init810 ], [ %2092, %handle_init_end811 ], [ %2112, %handle_init821 ], [ %2115, %handle_init_end822 ], [ %2139, %handle_init832 ], [ %2142, %handle_init_end833 ], [ %2173, %handle_init840 ], [ %2176, %handle_init_end841 ], [ %2198, %handle_init848 ], [ %2201, %handle_init_end849 ], [ %2223, %handle_init856 ], [ %2226, %handle_init_end857 ], [ %2246, %handle_init864 ], [ %2249, %handle_init_end865 ], [ %2271, %handle_init878 ], [ %2274, %handle_init_end879 ], [ %2298, %handle_init892 ], [ %2301, %handle_init_end893 ], [ %2325, %handle_init906 ], [ %2328, %handle_init_end907 ], [ %2347, %handle_init914 ], [ %2350, %handle_init_end915 ], [ %2373, %handle_init922 ], [ %2376, %handle_init_end923 ], [ %2394, %handle_init930 ], [ %2397, %handle_init_end931 ], [ %2421, %handle_init938 ], [ %2424, %handle_init_end939 ], [ %2429, %handle_init944 ], [ %2432, %handle_init_end945 ], [ %common.ret.op.i, %flashattn_gqa_decode_no_split_compute_.exit ]
  ret i32 %common.ret.op, !dbg !31

assert_fail:                                      ; preds = %entry
  %92 = load ptr, ptr @__TVMFFIErrorSetRaisedFromCStr, align 8, !dbg !31, !tbaa !34
  tail call void %92(ptr nonnull @.str.1, ptr nonnull @.str), !dbg !31
  br label %common.ret

assert_end:                                       ; preds = %entry
  %.not = icmp eq ptr %args, null, !dbg !31
  br i1 %.not, label %assert_fail1, label %assert_end2, !dbg !31, !prof !37

assert_fail1:                                     ; preds = %assert_end
  %93 = load ptr, ptr @__TVMFFIErrorSetRaisedFromCStr, align 8, !dbg !31, !tbaa !34
  tail call void %93(ptr nonnull @.str.1, ptr nonnull @.str.2), !dbg !31
  br label %common.ret

assert_end2:                                      ; preds = %assert_end
  %Q_handle.type_index = load i32, ptr %args, align 4, !dbg !31
  %Q_handle.type_index.fr = freeze i32 %Q_handle.type_index, !dbg !31
  call void @llvm.dbg.declare(metadata i32 %Q_handle.type_index, metadata !38, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.declare(metadata i32 %Q_handle.type_index, metadata !38, metadata !DIExpression()), !dbg !31
  %94 = icmp sgt i32 %Q_handle.type_index.fr, 63, !dbg !31
  br i1 %94, label %assert_end4, label %switch.early.test, !dbg !31

switch.early.test:                                ; preds = %assert_end2
  switch i32 %Q_handle.type_index.fr, label %assert_fail3 [
    i32 7, label %assert_end4
    i32 4, label %assert_end4
    i32 0, label %assert_end4
  ], !dbg !31

assert_fail3:                                     ; preds = %switch.early.test
  %95 = load ptr, ptr @__TVMFFIErrorSetRaisedFromCStr, align 8, !dbg !31, !tbaa !34
  tail call void %95(ptr nonnull @.str.1, ptr nonnull @.str.3), !dbg !31
  br label %common.ret

assert_end4:                                      ; preds = %switch.early.test, %switch.early.test, %switch.early.test, %assert_end2
  %96 = getelementptr inbounds %0, ptr %args, i64 1, i32 0, !dbg !31
  %K_handle.type_index = load i32, ptr %96, align 4, !dbg !31
  %K_handle.type_index.fr = freeze i32 %K_handle.type_index, !dbg !31
  call void @llvm.dbg.declare(metadata i32 %K_handle.type_index, metadata !39, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.declare(metadata i32 %K_handle.type_index, metadata !39, metadata !DIExpression()), !dbg !31
  %97 = icmp sgt i32 %K_handle.type_index.fr, 63, !dbg !31
  br i1 %97, label %assert_end6, label %switch.early.test952, !dbg !31

switch.early.test952:                             ; preds = %assert_end4
  switch i32 %K_handle.type_index.fr, label %assert_fail5 [
    i32 7, label %assert_end6
    i32 4, label %assert_end6
    i32 0, label %assert_end6
  ], !dbg !31

assert_fail5:                                     ; preds = %switch.early.test952
  %98 = load ptr, ptr @__TVMFFIErrorSetRaisedFromCStr, align 8, !dbg !31, !tbaa !34
  tail call void %98(ptr nonnull @.str.1, ptr nonnull @.str.4), !dbg !31
  br label %common.ret

assert_end6:                                      ; preds = %switch.early.test952, %switch.early.test952, %switch.early.test952, %assert_end4
  %99 = getelementptr inbounds %0, ptr %args, i64 2, i32 0, !dbg !31
  %V_handle.type_index = load i32, ptr %99, align 4, !dbg !31
  %V_handle.type_index.fr = freeze i32 %V_handle.type_index, !dbg !31
  call void @llvm.dbg.declare(metadata i32 %V_handle.type_index, metadata !40, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.declare(metadata i32 %V_handle.type_index, metadata !40, metadata !DIExpression()), !dbg !31
  %100 = icmp sgt i32 %V_handle.type_index.fr, 63, !dbg !31
  br i1 %100, label %assert_end8, label %switch.early.test953, !dbg !31

switch.early.test953:                             ; preds = %assert_end6
  switch i32 %V_handle.type_index.fr, label %assert_fail7 [
    i32 7, label %assert_end8
    i32 4, label %assert_end8
    i32 0, label %assert_end8
  ], !dbg !31

assert_fail7:                                     ; preds = %switch.early.test953
  %101 = load ptr, ptr @__TVMFFIErrorSetRaisedFromCStr, align 8, !dbg !31, !tbaa !34
  tail call void %101(ptr nonnull @.str.1, ptr nonnull @.str.5), !dbg !31
  br label %common.ret

assert_end8:                                      ; preds = %switch.early.test953, %switch.early.test953, %switch.early.test953, %assert_end6
  %102 = getelementptr inbounds %0, ptr %args, i64 3, i32 0, !dbg !31
  %mask_handle.type_index = load i32, ptr %102, align 4, !dbg !31
  %mask_handle.type_index.fr = freeze i32 %mask_handle.type_index, !dbg !31
  call void @llvm.dbg.declare(metadata i32 %mask_handle.type_index, metadata !41, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.declare(metadata i32 %mask_handle.type_index, metadata !41, metadata !DIExpression()), !dbg !31
  %103 = icmp sgt i32 %mask_handle.type_index.fr, 63, !dbg !31
  br i1 %103, label %assert_end10, label %switch.early.test954, !dbg !31

switch.early.test954:                             ; preds = %assert_end8
  switch i32 %mask_handle.type_index.fr, label %assert_fail9 [
    i32 7, label %assert_end10
    i32 4, label %assert_end10
    i32 0, label %assert_end10
  ], !dbg !31

assert_fail9:                                     ; preds = %switch.early.test954
  %104 = load ptr, ptr @__TVMFFIErrorSetRaisedFromCStr, align 8, !dbg !31, !tbaa !34
  tail call void %104(ptr nonnull @.str.1, ptr nonnull @.str.6), !dbg !31
  br label %common.ret

assert_end10:                                     ; preds = %switch.early.test954, %switch.early.test954, %switch.early.test954, %assert_end8
  %105 = getelementptr inbounds %0, ptr %args, i64 4, i32 0, !dbg !31
  %glse_handle.type_index = load i32, ptr %105, align 4, !dbg !31
  %glse_handle.type_index.fr = freeze i32 %glse_handle.type_index, !dbg !31
  call void @llvm.dbg.declare(metadata i32 %glse_handle.type_index, metadata !42, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.declare(metadata i32 %glse_handle.type_index, metadata !42, metadata !DIExpression()), !dbg !31
  %106 = icmp sgt i32 %glse_handle.type_index.fr, 63, !dbg !31
  br i1 %106, label %assert_end12, label %switch.early.test955, !dbg !31

switch.early.test955:                             ; preds = %assert_end10
  switch i32 %glse_handle.type_index.fr, label %assert_fail11 [
    i32 7, label %assert_end12
    i32 4, label %assert_end12
    i32 0, label %assert_end12
  ], !dbg !31

assert_fail11:                                    ; preds = %switch.early.test955
  %107 = load ptr, ptr @__TVMFFIErrorSetRaisedFromCStr, align 8, !dbg !31, !tbaa !34
  tail call void %107(ptr nonnull @.str.1, ptr nonnull @.str.7), !dbg !31
  br label %common.ret

assert_end12:                                     ; preds = %switch.early.test955, %switch.early.test955, %switch.early.test955, %assert_end10
  %108 = getelementptr inbounds %0, ptr %args, i64 5, i32 0, !dbg !31
  %Output_partial_handle.type_index = load i32, ptr %108, align 4, !dbg !31
  %Output_partial_handle.type_index.fr = freeze i32 %Output_partial_handle.type_index, !dbg !31
  call void @llvm.dbg.declare(metadata i32 %Output_partial_handle.type_index, metadata !43, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.declare(metadata i32 %Output_partial_handle.type_index, metadata !43, metadata !DIExpression()), !dbg !31
  %109 = icmp sgt i32 %Output_partial_handle.type_index.fr, 63, !dbg !31
  br i1 %109, label %assert_end14, label %switch.early.test956, !dbg !31

switch.early.test956:                             ; preds = %assert_end12
  switch i32 %Output_partial_handle.type_index.fr, label %assert_fail13 [
    i32 7, label %assert_end14
    i32 4, label %assert_end14
    i32 0, label %assert_end14
  ], !dbg !31

assert_fail13:                                    ; preds = %switch.early.test956
  %110 = load ptr, ptr @__TVMFFIErrorSetRaisedFromCStr, align 8, !dbg !31, !tbaa !34
  tail call void %110(ptr nonnull @.str.1, ptr nonnull @.str.8), !dbg !31
  br label %common.ret

assert_end14:                                     ; preds = %switch.early.test956, %switch.early.test956, %switch.early.test956, %assert_end12
  %111 = getelementptr inbounds %0, ptr %args, i64 6, i32 0, !dbg !31
  %Output_handle.type_index = load i32, ptr %111, align 4, !dbg !31
  %Output_handle.type_index.fr = freeze i32 %Output_handle.type_index, !dbg !31
  call void @llvm.dbg.declare(metadata i32 %Output_handle.type_index, metadata !44, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.declare(metadata i32 %Output_handle.type_index, metadata !44, metadata !DIExpression()), !dbg !31
  %112 = icmp sgt i32 %Output_handle.type_index.fr, 63, !dbg !31
  br i1 %112, label %assert_end16, label %switch.early.test957, !dbg !31

switch.early.test957:                             ; preds = %assert_end14
  switch i32 %Output_handle.type_index.fr, label %assert_fail15 [
    i32 7, label %assert_end16
    i32 4, label %assert_end16
    i32 0, label %assert_end16
  ], !dbg !31

assert_fail15:                                    ; preds = %switch.early.test957
  %113 = load ptr, ptr @__TVMFFIErrorSetRaisedFromCStr, align 8, !dbg !31, !tbaa !34
  tail call void %113(ptr nonnull @.str.1, ptr nonnull @.str.9), !dbg !31
  br label %common.ret

assert_end16:                                     ; preds = %switch.early.test957, %switch.early.test957, %switch.early.test957, %assert_end14
  %114 = getelementptr inbounds %0, ptr %args, i64 0, i32 2, !dbg !31
  %115 = load ptr, ptr %114, align 8, !dbg !31
  %116 = icmp eq i32 %Q_handle.type_index.fr, 70, !dbg !31
  %Q_handle.idx = select i1 %116, i64 24, i64 0, !dbg !31
  %Q_handle = getelementptr i8, ptr %115, i64 %Q_handle.idx, !dbg !31
  call void @llvm.dbg.declare(metadata ptr %Q_handle, metadata !45, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.declare(metadata ptr %Q_handle, metadata !45, metadata !DIExpression()), !dbg !31
  %117 = getelementptr inbounds %0, ptr %args, i64 1, i32 2, !dbg !31
  %118 = load ptr, ptr %117, align 8, !dbg !31
  %119 = icmp eq i32 %K_handle.type_index.fr, 70, !dbg !31
  %K_handle.idx = select i1 %119, i64 24, i64 0, !dbg !31
  %K_handle = getelementptr i8, ptr %118, i64 %K_handle.idx, !dbg !31
  call void @llvm.dbg.declare(metadata ptr %K_handle, metadata !46, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.declare(metadata ptr %K_handle, metadata !46, metadata !DIExpression()), !dbg !31
  %120 = getelementptr inbounds %0, ptr %args, i64 2, i32 2, !dbg !31
  %121 = load ptr, ptr %120, align 8, !dbg !31
  %122 = icmp eq i32 %V_handle.type_index.fr, 70, !dbg !31
  %V_handle.idx = select i1 %122, i64 24, i64 0, !dbg !31
  %V_handle = getelementptr i8, ptr %121, i64 %V_handle.idx, !dbg !31
  call void @llvm.dbg.declare(metadata ptr %V_handle, metadata !47, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.declare(metadata ptr %V_handle, metadata !47, metadata !DIExpression()), !dbg !31
  %123 = getelementptr inbounds %0, ptr %args, i64 3, i32 2, !dbg !31
  %124 = load ptr, ptr %123, align 8, !dbg !31
  %125 = icmp eq i32 %mask_handle.type_index.fr, 70, !dbg !31
  %mask_handle.idx = select i1 %125, i64 24, i64 0, !dbg !31
  %mask_handle = getelementptr i8, ptr %124, i64 %mask_handle.idx, !dbg !31
  call void @llvm.dbg.declare(metadata ptr %mask_handle, metadata !48, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.declare(metadata ptr %mask_handle, metadata !48, metadata !DIExpression()), !dbg !31
  %126 = getelementptr inbounds %0, ptr %args, i64 4, i32 2, !dbg !31
  %127 = load ptr, ptr %126, align 8, !dbg !31
  %128 = icmp eq i32 %glse_handle.type_index.fr, 70, !dbg !31
  %glse_handle.idx = select i1 %128, i64 24, i64 0, !dbg !31
  %glse_handle = getelementptr i8, ptr %127, i64 %glse_handle.idx, !dbg !31
  call void @llvm.dbg.declare(metadata ptr %glse_handle, metadata !49, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.declare(metadata ptr %glse_handle, metadata !49, metadata !DIExpression()), !dbg !31
  %129 = getelementptr inbounds %0, ptr %args, i64 5, i32 2, !dbg !31
  %130 = load ptr, ptr %129, align 8, !dbg !31
  %131 = icmp eq i32 %Output_partial_handle.type_index.fr, 70, !dbg !31
  %Output_partial_handle.idx = select i1 %131, i64 24, i64 0, !dbg !31
  %Output_partial_handle = getelementptr i8, ptr %130, i64 %Output_partial_handle.idx, !dbg !31
  call void @llvm.dbg.declare(metadata ptr %Output_partial_handle, metadata !50, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.declare(metadata ptr %Output_partial_handle, metadata !50, metadata !DIExpression()), !dbg !31
  %132 = getelementptr inbounds %0, ptr %args, i64 6, i32 2, !dbg !31
  %133 = load ptr, ptr %132, align 8, !dbg !31
  %134 = icmp eq i32 %Output_handle.type_index.fr, 70, !dbg !31
  %Output_handle.idx = select i1 %134, i64 24, i64 0, !dbg !31
  %Output_handle = getelementptr i8, ptr %133, i64 %Output_handle.idx, !dbg !31
  call void @llvm.dbg.declare(metadata ptr %Output_handle, metadata !51, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.declare(metadata ptr %Output_handle, metadata !51, metadata !DIExpression()), !dbg !31
  %flashattn_gqa_decode_no_split.Q_is_null.not = icmp eq ptr %Q_handle, null, !dbg !31
  call void @llvm.dbg.declare(metadata i1 %flashattn_gqa_decode_no_split.Q_is_null.not, metadata !52, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.declare(metadata i1 %flashattn_gqa_decode_no_split.Q_is_null.not, metadata !52, metadata !DIExpression()), !dbg !31
  br i1 %flashattn_gqa_decode_no_split.Q_is_null.not, label %assert_fail17, label %assert_end18, !dbg !31, !prof !37

assert_fail17:                                    ; preds = %assert_end16
  %135 = load ptr, ptr @__TVMFFIErrorSetRaisedFromCStr, align 8, !dbg !31, !tbaa !34
  tail call void %135(ptr nonnull @.str.1, ptr nonnull @.str.10), !dbg !31
  br label %common.ret

assert_end18:                                     ; preds = %assert_end16
  %flashattn_gqa_decode_no_split.K_is_null.not = icmp eq ptr %K_handle, null, !dbg !31
  call void @llvm.dbg.declare(metadata i1 %flashattn_gqa_decode_no_split.K_is_null.not, metadata !54, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.declare(metadata i1 %flashattn_gqa_decode_no_split.K_is_null.not, metadata !54, metadata !DIExpression()), !dbg !31
  br i1 %flashattn_gqa_decode_no_split.K_is_null.not, label %assert_fail19, label %assert_end20, !dbg !31, !prof !37

assert_fail19:                                    ; preds = %assert_end18
  %136 = load ptr, ptr @__TVMFFIErrorSetRaisedFromCStr, align 8, !dbg !31, !tbaa !34
  tail call void %136(ptr nonnull @.str.1, ptr nonnull @.str.11), !dbg !31
  br label %common.ret

assert_end20:                                     ; preds = %assert_end18
  %flashattn_gqa_decode_no_split.V_is_null.not = icmp eq ptr %V_handle, null, !dbg !31
  call void @llvm.dbg.declare(metadata i1 %flashattn_gqa_decode_no_split.V_is_null.not, metadata !55, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.declare(metadata i1 %flashattn_gqa_decode_no_split.V_is_null.not, metadata !55, metadata !DIExpression()), !dbg !31
  br i1 %flashattn_gqa_decode_no_split.V_is_null.not, label %assert_fail21, label %assert_end22, !dbg !31, !prof !37

assert_fail21:                                    ; preds = %assert_end20
  %137 = load ptr, ptr @__TVMFFIErrorSetRaisedFromCStr, align 8, !dbg !31, !tbaa !34
  tail call void %137(ptr nonnull @.str.1, ptr nonnull @.str.12), !dbg !31
  br label %common.ret

assert_end22:                                     ; preds = %assert_end20
  %flashattn_gqa_decode_no_split.mask_is_null.not = icmp eq ptr %mask_handle, null, !dbg !31
  call void @llvm.dbg.declare(metadata i1 %flashattn_gqa_decode_no_split.mask_is_null.not, metadata !56, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.declare(metadata i1 %flashattn_gqa_decode_no_split.mask_is_null.not, metadata !56, metadata !DIExpression()), !dbg !31
  br i1 %flashattn_gqa_decode_no_split.mask_is_null.not, label %assert_fail23, label %assert_end24, !dbg !31, !prof !37

assert_fail23:                                    ; preds = %assert_end22
  %138 = load ptr, ptr @__TVMFFIErrorSetRaisedFromCStr, align 8, !dbg !31, !tbaa !34
  tail call void %138(ptr nonnull @.str.1, ptr nonnull @.str.13), !dbg !31
  br label %common.ret

assert_end24:                                     ; preds = %assert_end22
  call void @llvm.dbg.declare(metadata i1 %flashattn_gqa_decode_no_split.glse_is_null.not, metadata !57, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.declare(metadata i1 %flashattn_gqa_decode_no_split.glse_is_null.not, metadata !57, metadata !DIExpression()), !dbg !31
  %flashattn_gqa_decode_no_split.Output_partial_is_null.not = icmp eq ptr %Output_partial_handle, null, !dbg !31
  call void @llvm.dbg.declare(metadata i1 %flashattn_gqa_decode_no_split.Output_partial_is_null.not, metadata !58, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.declare(metadata i1 %flashattn_gqa_decode_no_split.Output_partial_is_null.not, metadata !58, metadata !DIExpression()), !dbg !31
  %flashattn_gqa_decode_no_split.Output_is_null.not = icmp eq ptr %Output_handle, null, !dbg !31
  call void @llvm.dbg.declare(metadata i1 %flashattn_gqa_decode_no_split.Output_is_null.not, metadata !59, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.declare(metadata i1 %flashattn_gqa_decode_no_split.Output_is_null.not, metadata !59, metadata !DIExpression()), !dbg !31
  br i1 %flashattn_gqa_decode_no_split.Output_is_null.not, label %assert_fail25, label %assert_end26, !dbg !31, !prof !37

assert_fail25:                                    ; preds = %assert_end24
  %139 = load ptr, ptr @__TVMFFIErrorSetRaisedFromCStr, align 8, !dbg !31, !tbaa !34
  tail call void %139(ptr nonnull @.str.1, ptr nonnull @.str.14), !dbg !31
  br label %common.ret

assert_end26:                                     ; preds = %assert_end24
  %flashattn_gqa_decode_no_split.glse_is_null.not = icmp eq ptr %glse_handle, null, !dbg !31
  %140 = getelementptr inbounds %1, ptr %Q_handle, i64 0, i32 4, !dbg !31
  %flashattn_gqa_decode_no_split.Q.shape = load ptr, ptr %140, align 8, !dbg !31
  call void @llvm.dbg.declare(metadata ptr %flashattn_gqa_decode_no_split.Q.shape, metadata !60, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.declare(metadata ptr %flashattn_gqa_decode_no_split.Q.shape, metadata !60, metadata !DIExpression()), !dbg !31
  %141 = getelementptr inbounds %1, ptr %K_handle, i64 0, i32 4, !dbg !31
  %flashattn_gqa_decode_no_split.K.shape = load ptr, ptr %141, align 8, !dbg !31
  call void @llvm.dbg.declare(metadata ptr %flashattn_gqa_decode_no_split.K.shape, metadata !63, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.declare(metadata ptr %flashattn_gqa_decode_no_split.K.shape, metadata !63, metadata !DIExpression()), !dbg !31
  %142 = getelementptr inbounds %1, ptr %V_handle, i64 0, i32 4, !dbg !31
  %flashattn_gqa_decode_no_split.V.shape = load ptr, ptr %142, align 8, !dbg !31
  call void @llvm.dbg.declare(metadata ptr %flashattn_gqa_decode_no_split.V.shape, metadata !64, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.declare(metadata ptr %flashattn_gqa_decode_no_split.V.shape, metadata !64, metadata !DIExpression()), !dbg !31
  %143 = getelementptr inbounds %1, ptr %mask_handle, i64 0, i32 4, !dbg !31
  %flashattn_gqa_decode_no_split.mask.shape = load ptr, ptr %143, align 8, !dbg !31
  call void @llvm.dbg.declare(metadata ptr %flashattn_gqa_decode_no_split.mask.shape, metadata !65, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.declare(metadata ptr %flashattn_gqa_decode_no_split.mask.shape, metadata !65, metadata !DIExpression()), !dbg !31
  br i1 %flashattn_gqa_decode_no_split.glse_is_null.not, label %if_end, label %if_then, !dbg !31

if_then:                                          ; preds = %assert_end26
  %144 = getelementptr inbounds %1, ptr %glse_handle, i64 0, i32 4, !dbg !31
  %145 = load ptr, ptr %144, align 8, !dbg !31
  br label %if_end, !dbg !31

if_end:                                           ; preds = %assert_end26, %if_then
  %flashattn_gqa_decode_no_split.glse.shape = phi ptr [ %145, %if_then ], [ null, %assert_end26 ], !dbg !31
  call void @llvm.dbg.declare(metadata ptr %flashattn_gqa_decode_no_split.glse.shape, metadata !66, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.declare(metadata ptr %flashattn_gqa_decode_no_split.glse.shape, metadata !66, metadata !DIExpression()), !dbg !31
  br i1 %flashattn_gqa_decode_no_split.Output_partial_is_null.not, label %if_end29, label %if_then27, !dbg !31

if_then27:                                        ; preds = %if_end
  %146 = getelementptr inbounds %1, ptr %Output_partial_handle, i64 0, i32 4, !dbg !31
  %147 = load ptr, ptr %146, align 8, !dbg !31
  br label %if_end29, !dbg !31

if_end29:                                         ; preds = %if_end, %if_then27
  %flashattn_gqa_decode_no_split.Output_partial.shape = phi ptr [ %147, %if_then27 ], [ null, %if_end ], !dbg !31
  call void @llvm.dbg.declare(metadata ptr %flashattn_gqa_decode_no_split.Output_partial.shape, metadata !67, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.declare(metadata ptr %flashattn_gqa_decode_no_split.Output_partial.shape, metadata !67, metadata !DIExpression()), !dbg !31
  %148 = getelementptr inbounds %1, ptr %Output_handle, i64 0, i32 4, !dbg !31
  %flashattn_gqa_decode_no_split.Output.shape = load ptr, ptr %148, align 8, !dbg !31
  call void @llvm.dbg.declare(metadata ptr %flashattn_gqa_decode_no_split.Output.shape, metadata !68, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.declare(metadata ptr %flashattn_gqa_decode_no_split.Output.shape, metadata !68, metadata !DIExpression()), !dbg !31
  %149 = getelementptr inbounds %1, ptr %Q_handle, i64 0, i32 2, !dbg !31
  %150 = load i32, ptr %149, align 4, !dbg !31
  %.not1229 = icmp eq i32 %150, 3, !dbg !31
  br i1 %.not1229, label %if_end31, label %if_then30, !dbg !31, !prof !37

if_then30:                                        ; preds = %if_end29
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %151 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %151, align 8, !dbg !31
  %152 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %152, align 8, !dbg !31
  %153 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.16, ptr %153, align 8, !dbg !31
  %154 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %154, align 8, !dbg !31
  %155 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store i64 3, ptr %155, align 8, !dbg !31
  %156 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %156, align 8, !dbg !31
  %157 = load i32, ptr %149, align 4, !dbg !31
  %158 = sext i32 %157 to i64, !dbg !31
  %159 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 %158, ptr %159, align 8, !dbg !31
  %160 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %160, i8 0, i64 16, i1 false), !dbg !31
  %161 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %162 = load ptr, ptr @.tvm_func.__tvm_error_ndim_mismatch, align 8, !dbg !31
  %.not1440 = icmp eq ptr %162, null, !dbg !31
  br i1 %.not1440, label %handle_init, label %handle_init_end, !dbg !31, !prof !37

if_end31:                                         ; preds = %handle_init_end, %if_end29
  %163 = getelementptr inbounds %1, ptr %Q_handle, i64 0, i32 5, !dbg !31
  %flashattn_gqa_decode_no_split.Q.strides = load ptr, ptr %163, align 8, !dbg !31
  call void @llvm.dbg.declare(metadata ptr %flashattn_gqa_decode_no_split.Q.strides, metadata !69, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.declare(metadata ptr %flashattn_gqa_decode_no_split.Q.strides, metadata !69, metadata !DIExpression()), !dbg !31
  %164 = getelementptr inbounds %1, ptr %Q_handle, i64 0, i32 1, i32 1, !dbg !31
  %dev_id = load i32, ptr %164, align 4, !dbg !31
  call void @llvm.dbg.declare(metadata i32 %dev_id, metadata !70, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.declare(metadata i32 %dev_id, metadata !70, metadata !DIExpression()), !dbg !31
  %Q = load ptr, ptr %Q_handle, align 8, !dbg !31
  call void @llvm.dbg.declare(metadata ptr %Q, metadata !71, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.declare(metadata ptr %Q, metadata !71, metadata !DIExpression()), !dbg !31
  call void @llvm.assume(i1 true) [ "align"(ptr %Q, i64 64) ], !dbg !31
  %165 = getelementptr inbounds %1, ptr %K_handle, i64 0, i32 2, !dbg !31
  %166 = load i32, ptr %165, align 4, !dbg !31
  %.not1230 = icmp eq i32 %166, 4, !dbg !31
  br i1 %.not1230, label %if_end35, label %if_then34, !dbg !31, !prof !37

handle_init:                                      ; preds = %if_then30
  %167 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %168 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %169 = call i32 %168(ptr %167, ptr nonnull @.str.17, ptr nonnull %90), !dbg !31
  %170 = icmp eq i32 %169, 0, !dbg !31
  br i1 %170, label %call_end, label %common.ret, !dbg !31, !prof !33

handle_init_end:                                  ; preds = %call_end, %if_then30
  %171 = phi ptr [ %162, %if_then30 ], [ %174, %call_end ], !dbg !31
  %172 = call i32 %161(ptr %171, ptr nonnull %stack_ffi_any1228, i32 4, ptr nonnull %160), !dbg !31
  %173 = icmp eq i32 %172, 0, !dbg !31
  br i1 %173, label %if_end31, label %common.ret, !dbg !31, !prof !33

call_end:                                         ; preds = %handle_init
  %174 = load ptr, ptr %90, align 8, !dbg !31
  store ptr %174, ptr @.tvm_func.__tvm_error_ndim_mismatch, align 8, !dbg !31
  br label %handle_init_end, !dbg !31

if_then34:                                        ; preds = %if_end31
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %175 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %175, align 8, !dbg !31
  %176 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %176, align 8, !dbg !31
  %177 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.18, ptr %177, align 8, !dbg !31
  %178 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %178, align 8, !dbg !31
  %179 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store i64 4, ptr %179, align 8, !dbg !31
  %180 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %180, align 8, !dbg !31
  %181 = load i32, ptr %165, align 4, !dbg !31
  %182 = sext i32 %181 to i64, !dbg !31
  %183 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 %182, ptr %183, align 8, !dbg !31
  %184 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %184, i8 0, i64 16, i1 false), !dbg !31
  %185 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %186 = load ptr, ptr @.tvm_func.__tvm_error_ndim_mismatch, align 8, !dbg !31
  %.not1439 = icmp eq ptr %186, null, !dbg !31
  br i1 %.not1439, label %handle_init36, label %handle_init_end37, !dbg !31, !prof !37

if_end35:                                         ; preds = %handle_init_end37, %if_end31
  %187 = getelementptr inbounds %1, ptr %K_handle, i64 0, i32 5, !dbg !31
  %flashattn_gqa_decode_no_split.K.strides = load ptr, ptr %187, align 8, !dbg !31
  call void @llvm.dbg.declare(metadata ptr %flashattn_gqa_decode_no_split.K.strides, metadata !72, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.declare(metadata ptr %flashattn_gqa_decode_no_split.K.strides, metadata !72, metadata !DIExpression()), !dbg !31
  %K = load ptr, ptr %K_handle, align 8, !dbg !31
  call void @llvm.dbg.declare(metadata ptr %K, metadata !73, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.declare(metadata ptr %K, metadata !73, metadata !DIExpression()), !dbg !31
  call void @llvm.assume(i1 true) [ "align"(ptr %K, i64 64) ], !dbg !31
  %188 = getelementptr inbounds %1, ptr %V_handle, i64 0, i32 2, !dbg !31
  %189 = load i32, ptr %188, align 4, !dbg !31
  %.not1231 = icmp eq i32 %189, 4, !dbg !31
  br i1 %.not1231, label %if_end43, label %if_then42, !dbg !31, !prof !37

handle_init36:                                    ; preds = %if_then34
  %190 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %191 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %192 = call i32 %191(ptr %190, ptr nonnull @.str.17, ptr nonnull %89), !dbg !31
  %193 = icmp eq i32 %192, 0, !dbg !31
  br i1 %193, label %call_end39, label %common.ret, !dbg !31, !prof !33

handle_init_end37:                                ; preds = %call_end39, %if_then34
  %194 = phi ptr [ %186, %if_then34 ], [ %197, %call_end39 ], !dbg !31
  %195 = call i32 %185(ptr %194, ptr nonnull %stack_ffi_any1228, i32 4, ptr nonnull %184), !dbg !31
  %196 = icmp eq i32 %195, 0, !dbg !31
  br i1 %196, label %if_end35, label %common.ret, !dbg !31, !prof !33

call_end39:                                       ; preds = %handle_init36
  %197 = load ptr, ptr %89, align 8, !dbg !31
  store ptr %197, ptr @.tvm_func.__tvm_error_ndim_mismatch, align 8, !dbg !31
  br label %handle_init_end37, !dbg !31

if_then42:                                        ; preds = %if_end35
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %198 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %198, align 8, !dbg !31
  %199 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %199, align 8, !dbg !31
  %200 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.19, ptr %200, align 8, !dbg !31
  %201 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %201, align 8, !dbg !31
  %202 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store i64 4, ptr %202, align 8, !dbg !31
  %203 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %203, align 8, !dbg !31
  %204 = load i32, ptr %188, align 4, !dbg !31
  %205 = sext i32 %204 to i64, !dbg !31
  %206 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 %205, ptr %206, align 8, !dbg !31
  %207 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %207, i8 0, i64 16, i1 false), !dbg !31
  %208 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %209 = load ptr, ptr @.tvm_func.__tvm_error_ndim_mismatch, align 8, !dbg !31
  %.not1438 = icmp eq ptr %209, null, !dbg !31
  br i1 %.not1438, label %handle_init44, label %handle_init_end45, !dbg !31, !prof !37

if_end43:                                         ; preds = %handle_init_end45, %if_end35
  %210 = getelementptr inbounds %1, ptr %V_handle, i64 0, i32 5, !dbg !31
  %flashattn_gqa_decode_no_split.V.strides = load ptr, ptr %210, align 8, !dbg !31
  call void @llvm.dbg.declare(metadata ptr %flashattn_gqa_decode_no_split.V.strides, metadata !74, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.declare(metadata ptr %flashattn_gqa_decode_no_split.V.strides, metadata !74, metadata !DIExpression()), !dbg !31
  %V = load ptr, ptr %V_handle, align 8, !dbg !31
  call void @llvm.dbg.declare(metadata ptr %V, metadata !75, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.declare(metadata ptr %V, metadata !75, metadata !DIExpression()), !dbg !31
  call void @llvm.assume(i1 true) [ "align"(ptr %V, i64 64) ], !dbg !31
  %211 = getelementptr inbounds %1, ptr %mask_handle, i64 0, i32 2, !dbg !31
  %212 = load i32, ptr %211, align 4, !dbg !31
  %.not1232 = icmp eq i32 %212, 3, !dbg !31
  br i1 %.not1232, label %if_end51, label %if_then50, !dbg !31, !prof !37

handle_init44:                                    ; preds = %if_then42
  %213 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %214 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %215 = call i32 %214(ptr %213, ptr nonnull @.str.17, ptr nonnull %88), !dbg !31
  %216 = icmp eq i32 %215, 0, !dbg !31
  br i1 %216, label %call_end47, label %common.ret, !dbg !31, !prof !33

handle_init_end45:                                ; preds = %call_end47, %if_then42
  %217 = phi ptr [ %209, %if_then42 ], [ %220, %call_end47 ], !dbg !31
  %218 = call i32 %208(ptr %217, ptr nonnull %stack_ffi_any1228, i32 4, ptr nonnull %207), !dbg !31
  %219 = icmp eq i32 %218, 0, !dbg !31
  br i1 %219, label %if_end43, label %common.ret, !dbg !31, !prof !33

call_end47:                                       ; preds = %handle_init44
  %220 = load ptr, ptr %88, align 8, !dbg !31
  store ptr %220, ptr @.tvm_func.__tvm_error_ndim_mismatch, align 8, !dbg !31
  br label %handle_init_end45, !dbg !31

if_then50:                                        ; preds = %if_end43
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %221 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %221, align 8, !dbg !31
  %222 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %222, align 8, !dbg !31
  %223 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.20, ptr %223, align 8, !dbg !31
  %224 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %224, align 8, !dbg !31
  %225 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store i64 3, ptr %225, align 8, !dbg !31
  %226 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %226, align 8, !dbg !31
  %227 = load i32, ptr %211, align 4, !dbg !31
  %228 = sext i32 %227 to i64, !dbg !31
  %229 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 %228, ptr %229, align 8, !dbg !31
  %230 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %230, i8 0, i64 16, i1 false), !dbg !31
  %231 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %232 = load ptr, ptr @.tvm_func.__tvm_error_ndim_mismatch, align 8, !dbg !31
  %.not1437 = icmp eq ptr %232, null, !dbg !31
  br i1 %.not1437, label %handle_init52, label %handle_init_end53, !dbg !31, !prof !37

if_end51:                                         ; preds = %handle_init_end53, %if_end43
  %233 = getelementptr inbounds %1, ptr %mask_handle, i64 0, i32 5, !dbg !31
  %flashattn_gqa_decode_no_split.mask.strides = load ptr, ptr %233, align 8, !dbg !31
  call void @llvm.dbg.declare(metadata ptr %flashattn_gqa_decode_no_split.mask.strides, metadata !76, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.declare(metadata ptr %flashattn_gqa_decode_no_split.mask.strides, metadata !76, metadata !DIExpression()), !dbg !31
  %mask = load ptr, ptr %mask_handle, align 8, !dbg !31
  call void @llvm.dbg.declare(metadata ptr %mask, metadata !77, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.declare(metadata ptr %mask, metadata !77, metadata !DIExpression()), !dbg !31
  call void @llvm.assume(i1 true) [ "align"(ptr %mask, i64 64) ], !dbg !31
  br i1 %flashattn_gqa_decode_no_split.glse_is_null.not, label %if_end74, label %if_then58, !dbg !31

handle_init52:                                    ; preds = %if_then50
  %234 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %235 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %236 = call i32 %235(ptr %234, ptr nonnull @.str.17, ptr nonnull %87), !dbg !31
  %237 = icmp eq i32 %236, 0, !dbg !31
  br i1 %237, label %call_end55, label %common.ret, !dbg !31, !prof !33

handle_init_end53:                                ; preds = %call_end55, %if_then50
  %238 = phi ptr [ %232, %if_then50 ], [ %241, %call_end55 ], !dbg !31
  %239 = call i32 %231(ptr %238, ptr nonnull %stack_ffi_any1228, i32 4, ptr nonnull %230), !dbg !31
  %240 = icmp eq i32 %239, 0, !dbg !31
  br i1 %240, label %if_end51, label %common.ret, !dbg !31, !prof !33

call_end55:                                       ; preds = %handle_init52
  %241 = load ptr, ptr %87, align 8, !dbg !31
  store ptr %241, ptr @.tvm_func.__tvm_error_ndim_mismatch, align 8, !dbg !31
  br label %handle_init_end53, !dbg !31

if_then58:                                        ; preds = %if_end51
  %242 = getelementptr inbounds %1, ptr %glse_handle, i64 0, i32 2, !dbg !31
  %243 = load i32, ptr %242, align 4, !dbg !31
  %.not1435 = icmp eq i32 %243, 3, !dbg !31
  br i1 %.not1435, label %if_then72, label %if_then61, !dbg !31, !prof !37

if_then61:                                        ; preds = %if_then58
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %244 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %244, align 8, !dbg !31
  %245 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %245, align 8, !dbg !31
  %246 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.21, ptr %246, align 8, !dbg !31
  %247 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %247, align 8, !dbg !31
  %248 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store i64 3, ptr %248, align 8, !dbg !31
  %249 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %249, align 8, !dbg !31
  %250 = load i32, ptr %242, align 4, !dbg !31
  %251 = sext i32 %250 to i64, !dbg !31
  %252 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 %251, ptr %252, align 8, !dbg !31
  %253 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %253, i8 0, i64 16, i1 false), !dbg !31
  %254 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %255 = load ptr, ptr @.tvm_func.__tvm_error_ndim_mismatch, align 8, !dbg !31
  %.not1436 = icmp eq ptr %255, null, !dbg !31
  br i1 %.not1436, label %handle_init63, label %handle_init_end64, !dbg !31, !prof !37

handle_init63:                                    ; preds = %if_then61
  %256 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %257 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %258 = call i32 %257(ptr %256, ptr nonnull @.str.17, ptr nonnull %86), !dbg !31
  %259 = icmp eq i32 %258, 0, !dbg !31
  br i1 %259, label %call_end66, label %common.ret, !dbg !31, !prof !33

handle_init_end64:                                ; preds = %call_end66, %if_then61
  %260 = phi ptr [ %255, %if_then61 ], [ %263, %call_end66 ], !dbg !31
  %261 = call i32 %254(ptr %260, ptr nonnull %stack_ffi_any1228, i32 4, ptr nonnull %253), !dbg !31
  %262 = icmp eq i32 %261, 0, !dbg !31
  br i1 %262, label %if_then72, label %common.ret, !dbg !31, !prof !33

call_end66:                                       ; preds = %handle_init63
  %263 = load ptr, ptr %86, align 8, !dbg !31
  store ptr %263, ptr @.tvm_func.__tvm_error_ndim_mismatch, align 8, !dbg !31
  br label %handle_init_end64, !dbg !31

if_then72:                                        ; preds = %handle_init_end64, %if_then58
  %264 = getelementptr inbounds %1, ptr %glse_handle, i64 0, i32 5, !dbg !31
  %265 = load ptr, ptr %264, align 8, !dbg !31
  call void @llvm.dbg.declare(metadata ptr undef, metadata !78, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.declare(metadata ptr undef, metadata !78, metadata !DIExpression()), !dbg !31
  %266 = load ptr, ptr %glse_handle, align 8, !dbg !31
  br label %if_end74, !dbg !31

if_end74:                                         ; preds = %if_end51, %if_then72
  %flashattn_gqa_decode_no_split.glse.strides1442 = phi ptr [ %265, %if_then72 ], [ null, %if_end51 ]
  %glse = phi ptr [ %266, %if_then72 ], [ null, %if_end51 ], !dbg !31
  call void @llvm.dbg.declare(metadata ptr %glse, metadata !79, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.declare(metadata ptr %glse, metadata !79, metadata !DIExpression()), !dbg !31
  call void @llvm.assume(i1 true) [ "align"(ptr %glse, i64 64) ], !dbg !31
  br i1 %flashattn_gqa_decode_no_split.Output_partial_is_null.not, label %if_end91, label %if_then75, !dbg !31

if_then75:                                        ; preds = %if_end74
  %267 = getelementptr inbounds %1, ptr %Output_partial_handle, i64 0, i32 2, !dbg !31
  %268 = load i32, ptr %267, align 4, !dbg !31
  %.not1433 = icmp eq i32 %268, 4, !dbg !31
  br i1 %.not1433, label %if_then89, label %if_then78, !dbg !31, !prof !37

if_then78:                                        ; preds = %if_then75
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %269 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %269, align 8, !dbg !31
  %270 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %270, align 8, !dbg !31
  %271 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.22, ptr %271, align 8, !dbg !31
  %272 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %272, align 8, !dbg !31
  %273 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store i64 4, ptr %273, align 8, !dbg !31
  %274 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %274, align 8, !dbg !31
  %275 = load i32, ptr %267, align 4, !dbg !31
  %276 = sext i32 %275 to i64, !dbg !31
  %277 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 %276, ptr %277, align 8, !dbg !31
  %278 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %278, i8 0, i64 16, i1 false), !dbg !31
  %279 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %280 = load ptr, ptr @.tvm_func.__tvm_error_ndim_mismatch, align 8, !dbg !31
  %.not1434 = icmp eq ptr %280, null, !dbg !31
  br i1 %.not1434, label %handle_init80, label %handle_init_end81, !dbg !31, !prof !37

handle_init80:                                    ; preds = %if_then78
  %281 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %282 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %283 = call i32 %282(ptr %281, ptr nonnull @.str.17, ptr nonnull %85), !dbg !31
  %284 = icmp eq i32 %283, 0, !dbg !31
  br i1 %284, label %call_end83, label %common.ret, !dbg !31, !prof !33

handle_init_end81:                                ; preds = %call_end83, %if_then78
  %285 = phi ptr [ %280, %if_then78 ], [ %288, %call_end83 ], !dbg !31
  %286 = call i32 %279(ptr %285, ptr nonnull %stack_ffi_any1228, i32 4, ptr nonnull %278), !dbg !31
  %287 = icmp eq i32 %286, 0, !dbg !31
  br i1 %287, label %if_then89, label %common.ret, !dbg !31, !prof !33

call_end83:                                       ; preds = %handle_init80
  %288 = load ptr, ptr %85, align 8, !dbg !31
  store ptr %288, ptr @.tvm_func.__tvm_error_ndim_mismatch, align 8, !dbg !31
  br label %handle_init_end81, !dbg !31

if_then89:                                        ; preds = %handle_init_end81, %if_then75
  %289 = getelementptr inbounds %1, ptr %Output_partial_handle, i64 0, i32 5, !dbg !31
  %290 = load ptr, ptr %289, align 8, !dbg !31
  call void @llvm.dbg.declare(metadata ptr undef, metadata !80, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.declare(metadata ptr undef, metadata !80, metadata !DIExpression()), !dbg !31
  %291 = load ptr, ptr %Output_partial_handle, align 8, !dbg !31
  br label %if_end91, !dbg !31

if_end91:                                         ; preds = %if_end74, %if_then89
  %flashattn_gqa_decode_no_split.Output_partial.strides1447 = phi ptr [ %290, %if_then89 ], [ null, %if_end74 ]
  %Output_partial = phi ptr [ %291, %if_then89 ], [ null, %if_end74 ], !dbg !31
  call void @llvm.dbg.declare(metadata ptr %Output_partial, metadata !81, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.declare(metadata ptr %Output_partial, metadata !81, metadata !DIExpression()), !dbg !31
  call void @llvm.assume(i1 true) [ "align"(ptr %Output_partial, i64 64) ], !dbg !31
  %292 = getelementptr inbounds %1, ptr %Output_handle, i64 0, i32 2, !dbg !31
  %293 = load i32, ptr %292, align 4, !dbg !31
  %.not1233 = icmp eq i32 %293, 3, !dbg !31
  br i1 %.not1233, label %if_end93, label %if_then92, !dbg !31, !prof !37

if_then92:                                        ; preds = %if_end91
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %294 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %294, align 8, !dbg !31
  %295 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %295, align 8, !dbg !31
  %296 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.23, ptr %296, align 8, !dbg !31
  %297 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %297, align 8, !dbg !31
  %298 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store i64 3, ptr %298, align 8, !dbg !31
  %299 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %299, align 8, !dbg !31
  %300 = load i32, ptr %292, align 4, !dbg !31
  %301 = sext i32 %300 to i64, !dbg !31
  %302 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 %301, ptr %302, align 8, !dbg !31
  %303 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %303, i8 0, i64 16, i1 false), !dbg !31
  %304 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %305 = load ptr, ptr @.tvm_func.__tvm_error_ndim_mismatch, align 8, !dbg !31
  %.not1432 = icmp eq ptr %305, null, !dbg !31
  br i1 %.not1432, label %handle_init94, label %handle_init_end95, !dbg !31, !prof !37

if_end93:                                         ; preds = %handle_init_end95, %if_end91
  %306 = getelementptr inbounds %1, ptr %Output_handle, i64 0, i32 5, !dbg !31
  %flashattn_gqa_decode_no_split.Output.strides = load ptr, ptr %306, align 8, !dbg !31
  call void @llvm.dbg.declare(metadata ptr %flashattn_gqa_decode_no_split.Output.strides, metadata !82, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.declare(metadata ptr %flashattn_gqa_decode_no_split.Output.strides, metadata !82, metadata !DIExpression()), !dbg !31
  %Output = load ptr, ptr %Output_handle, align 8, !dbg !31
  call void @llvm.dbg.declare(metadata ptr %Output, metadata !83, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.declare(metadata ptr %Output, metadata !83, metadata !DIExpression()), !dbg !31
  call void @llvm.assume(i1 true) [ "align"(ptr %Output, i64 64) ], !dbg !31
  %307 = getelementptr inbounds %1, ptr %Q_handle, i64 0, i32 3, i32 2, !dbg !31
  %308 = load i16, ptr %307, align 2, !dbg !31
  %309 = icmp ne i16 %308, 1, !dbg !31
  %310 = getelementptr inbounds %1, ptr %Q_handle, i64 0, i32 3, i32 1, !dbg !31
  %311 = load i8, ptr %310, align 1, !dbg !31
  %312 = icmp ne i8 %311, 16, !dbg !31
  %313 = getelementptr inbounds %1, ptr %Q_handle, i64 0, i32 3, i32 0, !dbg !31
  %314 = load i8, ptr %313, align 1, !dbg !31
  %315 = icmp ne i8 %314, 2, !dbg !31
  %316 = or i1 %312, %315, !dbg !31
  %317 = or i1 %309, %316, !dbg !31
  br i1 %317, label %if_then100, label %if_end101, !dbg !31, !prof !33

handle_init94:                                    ; preds = %if_then92
  %318 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %319 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %320 = call i32 %319(ptr %318, ptr nonnull @.str.17, ptr nonnull %84), !dbg !31
  %321 = icmp eq i32 %320, 0, !dbg !31
  br i1 %321, label %call_end97, label %common.ret, !dbg !31, !prof !33

handle_init_end95:                                ; preds = %call_end97, %if_then92
  %322 = phi ptr [ %305, %if_then92 ], [ %325, %call_end97 ], !dbg !31
  %323 = call i32 %304(ptr %322, ptr nonnull %stack_ffi_any1228, i32 4, ptr nonnull %303), !dbg !31
  %324 = icmp eq i32 %323, 0, !dbg !31
  br i1 %324, label %if_end93, label %common.ret, !dbg !31, !prof !33

call_end97:                                       ; preds = %handle_init94
  %325 = load ptr, ptr %84, align 8, !dbg !31
  store ptr %325, ptr @.tvm_func.__tvm_error_ndim_mismatch, align 8, !dbg !31
  br label %handle_init_end95, !dbg !31

if_then100:                                       ; preds = %if_end93
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %326 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %326, align 8, !dbg !31
  %327 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %327, align 8, !dbg !31
  %328 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.16, ptr %328, align 8, !dbg !31
  %329 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %329, align 8, !dbg !31
  %330 = load i8, ptr %313, align 1, !dbg !31
  %331 = zext i8 %330 to i64, !dbg !31
  %332 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store i64 %331, ptr %332, align 8, !dbg !31
  %333 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %333, align 8, !dbg !31
  %334 = load i8, ptr %310, align 1, !dbg !31
  %335 = zext i8 %334 to i64, !dbg !31
  %336 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 %335, ptr %336, align 8, !dbg !31
  %337 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %337, align 8, !dbg !31
  %338 = load i16, ptr %307, align 2, !dbg !31
  %339 = zext i16 %338 to i64, !dbg !31
  %340 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 2, !dbg !31
  store i64 %339, ptr %340, align 8, !dbg !31
  %341 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 5, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %341, align 8, !dbg !31
  %342 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 5, i32 2, !dbg !31
  store i64 2, ptr %342, align 8, !dbg !31
  %343 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 6, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %343, align 8, !dbg !31
  %344 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 6, i32 2, !dbg !31
  store i64 16, ptr %344, align 8, !dbg !31
  %345 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 7, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %345, align 8, !dbg !31
  %346 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 7, i32 2, !dbg !31
  store i64 1, ptr %346, align 8, !dbg !31
  %347 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 8, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %347, i8 0, i64 16, i1 false), !dbg !31
  %348 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %349 = load ptr, ptr @.tvm_func.__tvm_error_dtype_mismatch, align 8, !dbg !31
  %.not1431 = icmp eq ptr %349, null, !dbg !31
  br i1 %.not1431, label %handle_init102, label %handle_init_end103, !dbg !31, !prof !37

if_end101:                                        ; preds = %handle_init_end103, %if_end93
  %350 = load i64, ptr %flashattn_gqa_decode_no_split.Q.shape, align 8, !dbg !31, !tbaa !84
  %351 = and i64 %350, 4294967295, !dbg !31
  %.not1234 = icmp eq i64 %351, 1, !dbg !31
  br i1 %.not1234, label %if_end109, label %if_then108, !dbg !31, !prof !37

handle_init102:                                   ; preds = %if_then100
  %352 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %353 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %354 = call i32 %353(ptr %352, ptr nonnull @.str.24, ptr nonnull %83), !dbg !31
  %355 = icmp eq i32 %354, 0, !dbg !31
  br i1 %355, label %call_end105, label %common.ret, !dbg !31, !prof !33

handle_init_end103:                               ; preds = %call_end105, %if_then100
  %356 = phi ptr [ %349, %if_then100 ], [ %359, %call_end105 ], !dbg !31
  %357 = call i32 %348(ptr %356, ptr nonnull %stack_ffi_any1228, i32 8, ptr nonnull %347), !dbg !31
  %358 = icmp eq i32 %357, 0, !dbg !31
  br i1 %358, label %if_end101, label %common.ret, !dbg !31, !prof !33

call_end105:                                      ; preds = %handle_init102
  %359 = load ptr, ptr %83, align 8, !dbg !31
  store ptr %359, ptr @.tvm_func.__tvm_error_dtype_mismatch, align 8, !dbg !31
  br label %handle_init_end103, !dbg !31

if_then108:                                       ; preds = %if_end101
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %360 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %360, align 8, !dbg !31
  %361 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %361, align 8, !dbg !31
  %362 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.16, ptr %362, align 8, !dbg !31
  %363 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %363, align 8, !dbg !31
  %364 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store ptr @.str.25, ptr %364, align 8, !dbg !31
  %365 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %365, align 8, !dbg !31
  %366 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 1, ptr %366, align 8, !dbg !31
  %367 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %367, align 8, !dbg !31
  %368 = load i64, ptr %flashattn_gqa_decode_no_split.Q.shape, align 8, !dbg !31, !tbaa !84
  %sext1429 = shl i64 %368, 32, !dbg !31
  %369 = ashr exact i64 %sext1429, 32, !dbg !31
  %370 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 2, !dbg !31
  store i64 %369, ptr %370, align 8, !dbg !31
  %371 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 5, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %371, i8 0, i64 16, i1 false), !dbg !31
  %372 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %373 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  %.not1430 = icmp eq ptr %373, null, !dbg !31
  br i1 %.not1430, label %handle_init110, label %handle_init_end111, !dbg !31, !prof !37

if_end109:                                        ; preds = %handle_init_end111, %if_end101
  %374 = getelementptr inbounds i64, ptr %flashattn_gqa_decode_no_split.Q.shape, i64 1, !dbg !31
  %375 = load i64, ptr %374, align 8, !dbg !31, !tbaa !84
  %376 = and i64 %375, 4294967295, !dbg !31
  %.not1235 = icmp eq i64 %376, 32, !dbg !31
  br i1 %.not1235, label %if_end117, label %if_then116, !dbg !31, !prof !37

handle_init110:                                   ; preds = %if_then108
  %377 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %378 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %379 = call i32 %378(ptr %377, ptr nonnull @.str.26, ptr nonnull %82), !dbg !31
  %380 = icmp eq i32 %379, 0, !dbg !31
  br i1 %380, label %call_end113, label %common.ret, !dbg !31, !prof !33

handle_init_end111:                               ; preds = %call_end113, %if_then108
  %381 = phi ptr [ %373, %if_then108 ], [ %384, %call_end113 ], !dbg !31
  %382 = call i32 %372(ptr %381, ptr nonnull %stack_ffi_any1228, i32 5, ptr nonnull %371), !dbg !31
  %383 = icmp eq i32 %382, 0, !dbg !31
  br i1 %383, label %if_end109, label %common.ret, !dbg !31, !prof !33

call_end113:                                      ; preds = %handle_init110
  %384 = load ptr, ptr %82, align 8, !dbg !31
  store ptr %384, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  br label %handle_init_end111, !dbg !31

if_then116:                                       ; preds = %if_end109
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %385 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %385, align 8, !dbg !31
  %386 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %386, align 8, !dbg !31
  %387 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.16, ptr %387, align 8, !dbg !31
  %388 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %388, align 8, !dbg !31
  %389 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store ptr @.str.27, ptr %389, align 8, !dbg !31
  %390 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %390, align 8, !dbg !31
  %391 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 32, ptr %391, align 8, !dbg !31
  %392 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %392, align 8, !dbg !31
  %393 = load i64, ptr %374, align 8, !dbg !31, !tbaa !84
  %sext1427 = shl i64 %393, 32, !dbg !31
  %394 = ashr exact i64 %sext1427, 32, !dbg !31
  %395 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 2, !dbg !31
  store i64 %394, ptr %395, align 8, !dbg !31
  %396 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 5, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %396, i8 0, i64 16, i1 false), !dbg !31
  %397 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %398 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  %.not1428 = icmp eq ptr %398, null, !dbg !31
  br i1 %.not1428, label %handle_init118, label %handle_init_end119, !dbg !31, !prof !37

if_end117:                                        ; preds = %handle_init_end119, %if_end109
  %399 = getelementptr inbounds i64, ptr %flashattn_gqa_decode_no_split.Q.shape, i64 2, !dbg !31
  %400 = load i64, ptr %399, align 8, !dbg !31, !tbaa !84
  %401 = and i64 %400, 4294967295, !dbg !31
  %.not1236 = icmp eq i64 %401, 128, !dbg !31
  br i1 %.not1236, label %if_end125, label %if_then124, !dbg !31, !prof !37

handle_init118:                                   ; preds = %if_then116
  %402 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %403 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %404 = call i32 %403(ptr %402, ptr nonnull @.str.26, ptr nonnull %81), !dbg !31
  %405 = icmp eq i32 %404, 0, !dbg !31
  br i1 %405, label %call_end121, label %common.ret, !dbg !31, !prof !33

handle_init_end119:                               ; preds = %call_end121, %if_then116
  %406 = phi ptr [ %398, %if_then116 ], [ %409, %call_end121 ], !dbg !31
  %407 = call i32 %397(ptr %406, ptr nonnull %stack_ffi_any1228, i32 5, ptr nonnull %396), !dbg !31
  %408 = icmp eq i32 %407, 0, !dbg !31
  br i1 %408, label %if_end117, label %common.ret, !dbg !31, !prof !33

call_end121:                                      ; preds = %handle_init118
  %409 = load ptr, ptr %81, align 8, !dbg !31
  store ptr %409, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  br label %handle_init_end119, !dbg !31

if_then124:                                       ; preds = %if_end117
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %410 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %410, align 8, !dbg !31
  %411 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %411, align 8, !dbg !31
  %412 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.16, ptr %412, align 8, !dbg !31
  %413 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %413, align 8, !dbg !31
  %414 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store ptr @.str.28, ptr %414, align 8, !dbg !31
  %415 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %415, align 8, !dbg !31
  %416 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 128, ptr %416, align 8, !dbg !31
  %417 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %417, align 8, !dbg !31
  %418 = load i64, ptr %399, align 8, !dbg !31, !tbaa !84
  %sext1425 = shl i64 %418, 32, !dbg !31
  %419 = ashr exact i64 %sext1425, 32, !dbg !31
  %420 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 2, !dbg !31
  store i64 %419, ptr %420, align 8, !dbg !31
  %421 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 5, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %421, i8 0, i64 16, i1 false), !dbg !31
  %422 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %423 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  %.not1426 = icmp eq ptr %423, null, !dbg !31
  br i1 %.not1426, label %handle_init126, label %handle_init_end127, !dbg !31, !prof !37

if_end125:                                        ; preds = %handle_init_end127, %if_end117
  %424 = icmp eq ptr %flashattn_gqa_decode_no_split.Q.strides, null, !dbg !31
  br i1 %424, label %if_then149, label %if_end134, !dbg !31

handle_init126:                                   ; preds = %if_then124
  %425 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %426 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %427 = call i32 %426(ptr %425, ptr nonnull @.str.26, ptr nonnull %80), !dbg !31
  %428 = icmp eq i32 %427, 0, !dbg !31
  br i1 %428, label %call_end129, label %common.ret, !dbg !31, !prof !33

handle_init_end127:                               ; preds = %call_end129, %if_then124
  %429 = phi ptr [ %423, %if_then124 ], [ %432, %call_end129 ], !dbg !31
  %430 = call i32 %422(ptr %429, ptr nonnull %stack_ffi_any1228, i32 5, ptr nonnull %421), !dbg !31
  %431 = icmp eq i32 %430, 0, !dbg !31
  br i1 %431, label %if_end125, label %common.ret, !dbg !31, !prof !33

call_end129:                                      ; preds = %handle_init126
  %432 = load ptr, ptr %80, align 8, !dbg !31
  store ptr %432, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  br label %handle_init_end127, !dbg !31

if_end134:                                        ; preds = %if_end125
  %433 = getelementptr inbounds i64, ptr %flashattn_gqa_decode_no_split.Q.strides, i64 2, !dbg !31
  %434 = load i64, ptr %433, align 8, !dbg !31, !tbaa !84
  %435 = and i64 %434, 4294967295, !dbg !31
  %.not1237 = icmp eq i64 %435, 1, !dbg !31
  br i1 %.not1237, label %if_end148, label %if_end139, !dbg !31, !prof !37

if_end139:                                        ; preds = %if_end134
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %436 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %436, align 8, !dbg !31
  %437 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %437, align 8, !dbg !31
  %438 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.16, ptr %438, align 8, !dbg !31
  %439 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %439, align 8, !dbg !31
  %440 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store ptr @.str.29, ptr %440, align 8, !dbg !31
  %441 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %441, align 8, !dbg !31
  %442 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 1, ptr %442, align 8, !dbg !31
  %443 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %443, align 8, !dbg !31
  %444 = load i64, ptr %433, align 8, !dbg !31, !tbaa !84
  %sext1423 = shl i64 %444, 32, !dbg !31
  %445 = ashr exact i64 %sext1423, 32, !dbg !31
  %446 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 2, !dbg !31
  store i64 %445, ptr %446, align 8, !dbg !31
  %447 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 5, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %447, i8 0, i64 16, i1 false), !dbg !31
  %448 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %449 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  %.not1424 = icmp eq ptr %449, null, !dbg !31
  br i1 %.not1424, label %handle_init140, label %handle_init_end141, !dbg !31, !prof !37

handle_init140:                                   ; preds = %if_end139
  %450 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %451 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %452 = call i32 %451(ptr %450, ptr nonnull @.str.26, ptr nonnull %79), !dbg !31
  %453 = icmp eq i32 %452, 0, !dbg !31
  br i1 %453, label %call_end143, label %common.ret, !dbg !31, !prof !33

handle_init_end141:                               ; preds = %call_end143, %if_end139
  %454 = phi ptr [ %449, %if_end139 ], [ %457, %call_end143 ], !dbg !31
  %455 = call i32 %448(ptr %454, ptr nonnull %stack_ffi_any1228, i32 5, ptr nonnull %447), !dbg !31
  %456 = icmp eq i32 %455, 0, !dbg !31
  br i1 %456, label %if_end148, label %common.ret, !dbg !31, !prof !33

call_end143:                                      ; preds = %handle_init140
  %457 = load ptr, ptr %79, align 8, !dbg !31
  store ptr %457, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  br label %handle_init_end141, !dbg !31

if_end148:                                        ; preds = %if_end134, %handle_init_end141
  %458 = getelementptr inbounds i64, ptr %flashattn_gqa_decode_no_split.Q.strides, i64 1, !dbg !31
  %459 = load i64, ptr %458, align 8, !dbg !31, !tbaa !84
  %460 = and i64 %459, 4294967295, !dbg !31
  %.not1238 = icmp eq i64 %460, 128, !dbg !31
  br i1 %.not1238, label %if_end162, label %if_then149, !dbg !31, !prof !37

if_then149:                                       ; preds = %if_end125, %if_end148
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %461 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %461, align 8, !dbg !31
  %462 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %462, align 8, !dbg !31
  %463 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.16, ptr %463, align 8, !dbg !31
  %464 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %464, align 8, !dbg !31
  %465 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store ptr @.str.30, ptr %465, align 8, !dbg !31
  %466 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %466, align 8, !dbg !31
  %467 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 128, ptr %467, align 8, !dbg !31
  %468 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %468, align 8, !dbg !31
  br i1 %424, label %if_end153, label %if_else152, !dbg !31

if_end150:                                        ; preds = %handle_init_end155
  br i1 %424, label %if_then163, label %if_end162, !dbg !31

if_else152:                                       ; preds = %if_then149
  %469 = getelementptr inbounds i64, ptr %flashattn_gqa_decode_no_split.Q.strides, i64 1, !dbg !31
  %470 = load i64, ptr %469, align 8, !dbg !31, !tbaa !84
  br label %if_end153, !dbg !31

if_end153:                                        ; preds = %if_then149, %if_else152
  %471 = phi i64 [ %470, %if_else152 ], [ 1, %if_then149 ], !dbg !31
  %sext1421 = shl i64 %471, 32, !dbg !31
  %472 = ashr exact i64 %sext1421, 32, !dbg !31
  %473 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 2, !dbg !31
  store i64 %472, ptr %473, align 8, !dbg !31
  %474 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 5, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %474, i8 0, i64 16, i1 false), !dbg !31
  %475 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %476 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  %.not1422 = icmp eq ptr %476, null, !dbg !31
  br i1 %.not1422, label %handle_init154, label %handle_init_end155, !dbg !31, !prof !37

handle_init154:                                   ; preds = %if_end153
  %477 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %478 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %479 = call i32 %478(ptr %477, ptr nonnull @.str.26, ptr nonnull %78), !dbg !31
  %480 = icmp eq i32 %479, 0, !dbg !31
  br i1 %480, label %call_end157, label %common.ret, !dbg !31, !prof !33

handle_init_end155:                               ; preds = %call_end157, %if_end153
  %481 = phi ptr [ %476, %if_end153 ], [ %484, %call_end157 ], !dbg !31
  %482 = call i32 %475(ptr %481, ptr nonnull %stack_ffi_any1228, i32 5, ptr nonnull %474), !dbg !31
  %483 = icmp eq i32 %482, 0, !dbg !31
  br i1 %483, label %if_end150, label %common.ret, !dbg !31, !prof !33

call_end157:                                      ; preds = %handle_init154
  %484 = load ptr, ptr %78, align 8, !dbg !31
  store ptr %484, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  br label %handle_init_end155, !dbg !31

if_end162:                                        ; preds = %if_end148, %if_end150
  %485 = load i64, ptr %flashattn_gqa_decode_no_split.Q.strides, align 8, !dbg !31, !tbaa !84
  %486 = and i64 %485, 4294967295, !dbg !31
  %.not1239 = icmp eq i64 %486, 4096, !dbg !31
  br i1 %.not1239, label %if_end164, label %if_then163, !dbg !31, !prof !37

if_then163:                                       ; preds = %if_end150, %if_end162
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %487 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %487, align 8, !dbg !31
  %488 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %488, align 8, !dbg !31
  %489 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.16, ptr %489, align 8, !dbg !31
  %490 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %490, align 8, !dbg !31
  %491 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store ptr @.str.31, ptr %491, align 8, !dbg !31
  %492 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %492, align 8, !dbg !31
  %493 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 4096, ptr %493, align 8, !dbg !31
  %494 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %494, align 8, !dbg !31
  br i1 %424, label %if_end167, label %if_else166, !dbg !31

if_end164:                                        ; preds = %handle_init_end169, %if_end162
  %495 = getelementptr inbounds %1, ptr %Q_handle, i64 0, i32 6, !dbg !31
  %496 = load i64, ptr %495, align 8, !dbg !31
  %.not1240 = icmp eq i64 %496, 0, !dbg !31
  br i1 %.not1240, label %if_end175, label %if_then174, !dbg !31, !prof !37

if_else166:                                       ; preds = %if_then163
  %497 = load i64, ptr %flashattn_gqa_decode_no_split.Q.strides, align 8, !dbg !31, !tbaa !84
  br label %if_end167, !dbg !31

if_end167:                                        ; preds = %if_then163, %if_else166
  %498 = phi i64 [ %497, %if_else166 ], [ 1, %if_then163 ], !dbg !31
  %sext1419 = shl i64 %498, 32, !dbg !31
  %499 = ashr exact i64 %sext1419, 32, !dbg !31
  %500 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 2, !dbg !31
  store i64 %499, ptr %500, align 8, !dbg !31
  %501 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 5, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %501, i8 0, i64 16, i1 false), !dbg !31
  %502 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %503 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  %.not1420 = icmp eq ptr %503, null, !dbg !31
  br i1 %.not1420, label %handle_init168, label %handle_init_end169, !dbg !31, !prof !37

handle_init168:                                   ; preds = %if_end167
  %504 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %505 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %506 = call i32 %505(ptr %504, ptr nonnull @.str.26, ptr nonnull %77), !dbg !31
  %507 = icmp eq i32 %506, 0, !dbg !31
  br i1 %507, label %call_end171, label %common.ret, !dbg !31, !prof !33

handle_init_end169:                               ; preds = %call_end171, %if_end167
  %508 = phi ptr [ %503, %if_end167 ], [ %511, %call_end171 ], !dbg !31
  %509 = call i32 %502(ptr %508, ptr nonnull %stack_ffi_any1228, i32 5, ptr nonnull %501), !dbg !31
  %510 = icmp eq i32 %509, 0, !dbg !31
  br i1 %510, label %if_end164, label %common.ret, !dbg !31, !prof !33

call_end171:                                      ; preds = %handle_init168
  %511 = load ptr, ptr %77, align 8, !dbg !31
  store ptr %511, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  br label %handle_init_end169, !dbg !31

if_then174:                                       ; preds = %if_end164
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %512 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %512, align 8, !dbg !31
  %513 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %513, align 8, !dbg !31
  %514 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.16, ptr %514, align 8, !dbg !31
  %515 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %515, align 8, !dbg !31
  %516 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store i64 0, ptr %516, align 8, !dbg !31
  %517 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %517, align 8, !dbg !31
  %518 = load i64, ptr %495, align 8, !dbg !31
  %519 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 %518, ptr %519, align 8, !dbg !31
  %520 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %520, i8 0, i64 16, i1 false), !dbg !31
  %521 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %522 = load ptr, ptr @.tvm_func.__tvm_error_byte_offset_mismatch, align 8, !dbg !31
  %.not1418 = icmp eq ptr %522, null, !dbg !31
  br i1 %.not1418, label %handle_init176, label %handle_init_end177, !dbg !31, !prof !37

if_end175:                                        ; preds = %handle_init_end177, %if_end164
  %523 = getelementptr inbounds %1, ptr %Q_handle, i64 0, i32 1, i32 0, !dbg !31
  %524 = load i32, ptr %523, align 4, !dbg !31
  %.not1241 = icmp eq i32 %524, 2, !dbg !31
  br i1 %.not1241, label %if_end183, label %if_then182, !dbg !31, !prof !37

handle_init176:                                   ; preds = %if_then174
  %525 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %526 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %527 = call i32 %526(ptr %525, ptr nonnull @.str.32, ptr nonnull %76), !dbg !31
  %528 = icmp eq i32 %527, 0, !dbg !31
  br i1 %528, label %call_end179, label %common.ret, !dbg !31, !prof !33

handle_init_end177:                               ; preds = %call_end179, %if_then174
  %529 = phi ptr [ %522, %if_then174 ], [ %532, %call_end179 ], !dbg !31
  %530 = call i32 %521(ptr %529, ptr nonnull %stack_ffi_any1228, i32 4, ptr nonnull %520), !dbg !31
  %531 = icmp eq i32 %530, 0, !dbg !31
  br i1 %531, label %if_end175, label %common.ret, !dbg !31, !prof !33

call_end179:                                      ; preds = %handle_init176
  %532 = load ptr, ptr %76, align 8, !dbg !31
  store ptr %532, ptr @.tvm_func.__tvm_error_byte_offset_mismatch, align 8, !dbg !31
  br label %handle_init_end177, !dbg !31

if_then182:                                       ; preds = %if_end175
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %533 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %533, align 8, !dbg !31
  %534 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %534, align 8, !dbg !31
  %535 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.16, ptr %535, align 8, !dbg !31
  %536 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %536, align 8, !dbg !31
  %537 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store i64 2, ptr %537, align 8, !dbg !31
  %538 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %538, align 8, !dbg !31
  %539 = load i32, ptr %523, align 4, !dbg !31
  %540 = sext i32 %539 to i64, !dbg !31
  %541 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 %540, ptr %541, align 8, !dbg !31
  %542 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %542, i8 0, i64 16, i1 false), !dbg !31
  %543 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %544 = load ptr, ptr @.tvm_func.__tvm_error_device_type_mismatch, align 8, !dbg !31
  %.not1417 = icmp eq ptr %544, null, !dbg !31
  br i1 %.not1417, label %handle_init184, label %handle_init_end185, !dbg !31, !prof !37

if_end183:                                        ; preds = %handle_init_end185, %if_end175
  %545 = icmp eq ptr %Q, null, !dbg !31
  br i1 %545, label %if_then190, label %if_end191, !dbg !31, !prof !33

handle_init184:                                   ; preds = %if_then182
  %546 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %547 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %548 = call i32 %547(ptr %546, ptr nonnull @.str.33, ptr nonnull %75), !dbg !31
  %549 = icmp eq i32 %548, 0, !dbg !31
  br i1 %549, label %call_end187, label %common.ret, !dbg !31, !prof !33

handle_init_end185:                               ; preds = %call_end187, %if_then182
  %550 = phi ptr [ %544, %if_then182 ], [ %553, %call_end187 ], !dbg !31
  %551 = call i32 %543(ptr %550, ptr nonnull %stack_ffi_any1228, i32 4, ptr nonnull %542), !dbg !31
  %552 = icmp eq i32 %551, 0, !dbg !31
  br i1 %552, label %if_end183, label %common.ret, !dbg !31, !prof !33

call_end187:                                      ; preds = %handle_init184
  %553 = load ptr, ptr %75, align 8, !dbg !31
  store ptr %553, ptr @.tvm_func.__tvm_error_device_type_mismatch, align 8, !dbg !31
  br label %handle_init_end185, !dbg !31

if_then190:                                       ; preds = %if_end183
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %554 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %554, align 8, !dbg !31
  %555 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %555, align 8, !dbg !31
  %556 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.16, ptr %556, align 8, !dbg !31
  %557 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %557, align 8, !dbg !31
  %558 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store ptr @.str.34, ptr %558, align 8, !dbg !31
  %559 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %559, i8 0, i64 16, i1 false), !dbg !31
  %560 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %561 = load ptr, ptr @.tvm_func.__tvm_error_null_ptr, align 8, !dbg !31
  %.not1416 = icmp eq ptr %561, null, !dbg !31
  br i1 %.not1416, label %handle_init192, label %handle_init_end193, !dbg !31, !prof !37

if_end191:                                        ; preds = %handle_init_end193, %if_end183
  %562 = getelementptr inbounds %1, ptr %K_handle, i64 0, i32 3, i32 2, !dbg !31
  %563 = load i16, ptr %562, align 2, !dbg !31
  %564 = icmp ne i16 %563, 1, !dbg !31
  %565 = getelementptr inbounds %1, ptr %K_handle, i64 0, i32 3, i32 1, !dbg !31
  %566 = load i8, ptr %565, align 1, !dbg !31
  %567 = icmp ne i8 %566, 16, !dbg !31
  %568 = getelementptr inbounds %1, ptr %K_handle, i64 0, i32 3, i32 0, !dbg !31
  %569 = load i8, ptr %568, align 1, !dbg !31
  %570 = icmp ne i8 %569, 2, !dbg !31
  %571 = or i1 %567, %570, !dbg !31
  %572 = or i1 %564, %571, !dbg !31
  br i1 %572, label %if_then198, label %if_end199, !dbg !31, !prof !33

handle_init192:                                   ; preds = %if_then190
  %573 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %574 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %575 = call i32 %574(ptr %573, ptr nonnull @.str.35, ptr nonnull %74), !dbg !31
  %576 = icmp eq i32 %575, 0, !dbg !31
  br i1 %576, label %call_end195, label %common.ret, !dbg !31, !prof !33

handle_init_end193:                               ; preds = %call_end195, %if_then190
  %577 = phi ptr [ %561, %if_then190 ], [ %580, %call_end195 ], !dbg !31
  %578 = call i32 %560(ptr %577, ptr nonnull %stack_ffi_any1228, i32 3, ptr nonnull %559), !dbg !31
  %579 = icmp eq i32 %578, 0, !dbg !31
  br i1 %579, label %if_end191, label %common.ret, !dbg !31, !prof !33

call_end195:                                      ; preds = %handle_init192
  %580 = load ptr, ptr %74, align 8, !dbg !31
  store ptr %580, ptr @.tvm_func.__tvm_error_null_ptr, align 8, !dbg !31
  br label %handle_init_end193, !dbg !31

if_then198:                                       ; preds = %if_end191
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %581 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %581, align 8, !dbg !31
  %582 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %582, align 8, !dbg !31
  %583 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.18, ptr %583, align 8, !dbg !31
  %584 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %584, align 8, !dbg !31
  %585 = load i8, ptr %568, align 1, !dbg !31
  %586 = zext i8 %585 to i64, !dbg !31
  %587 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store i64 %586, ptr %587, align 8, !dbg !31
  %588 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %588, align 8, !dbg !31
  %589 = load i8, ptr %565, align 1, !dbg !31
  %590 = zext i8 %589 to i64, !dbg !31
  %591 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 %590, ptr %591, align 8, !dbg !31
  %592 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %592, align 8, !dbg !31
  %593 = load i16, ptr %562, align 2, !dbg !31
  %594 = zext i16 %593 to i64, !dbg !31
  %595 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 2, !dbg !31
  store i64 %594, ptr %595, align 8, !dbg !31
  %596 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 5, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %596, align 8, !dbg !31
  %597 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 5, i32 2, !dbg !31
  store i64 2, ptr %597, align 8, !dbg !31
  %598 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 6, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %598, align 8, !dbg !31
  %599 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 6, i32 2, !dbg !31
  store i64 16, ptr %599, align 8, !dbg !31
  %600 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 7, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %600, align 8, !dbg !31
  %601 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 7, i32 2, !dbg !31
  store i64 1, ptr %601, align 8, !dbg !31
  %602 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 8, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %602, i8 0, i64 16, i1 false), !dbg !31
  %603 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %604 = load ptr, ptr @.tvm_func.__tvm_error_dtype_mismatch, align 8, !dbg !31
  %.not1415 = icmp eq ptr %604, null, !dbg !31
  br i1 %.not1415, label %handle_init200, label %handle_init_end201, !dbg !31, !prof !37

if_end199:                                        ; preds = %handle_init_end201, %if_end191
  %605 = load i64, ptr %flashattn_gqa_decode_no_split.K.shape, align 8, !dbg !31, !tbaa !84
  %606 = and i64 %605, 4294967295, !dbg !31
  %.not1242 = icmp eq i64 %606, 1, !dbg !31
  br i1 %.not1242, label %if_end207, label %if_then206, !dbg !31, !prof !37

handle_init200:                                   ; preds = %if_then198
  %607 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %608 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %609 = call i32 %608(ptr %607, ptr nonnull @.str.24, ptr nonnull %73), !dbg !31
  %610 = icmp eq i32 %609, 0, !dbg !31
  br i1 %610, label %call_end203, label %common.ret, !dbg !31, !prof !33

handle_init_end201:                               ; preds = %call_end203, %if_then198
  %611 = phi ptr [ %604, %if_then198 ], [ %614, %call_end203 ], !dbg !31
  %612 = call i32 %603(ptr %611, ptr nonnull %stack_ffi_any1228, i32 8, ptr nonnull %602), !dbg !31
  %613 = icmp eq i32 %612, 0, !dbg !31
  br i1 %613, label %if_end199, label %common.ret, !dbg !31, !prof !33

call_end203:                                      ; preds = %handle_init200
  %614 = load ptr, ptr %73, align 8, !dbg !31
  store ptr %614, ptr @.tvm_func.__tvm_error_dtype_mismatch, align 8, !dbg !31
  br label %handle_init_end201, !dbg !31

if_then206:                                       ; preds = %if_end199
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %615 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %615, align 8, !dbg !31
  %616 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %616, align 8, !dbg !31
  %617 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.18, ptr %617, align 8, !dbg !31
  %618 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %618, align 8, !dbg !31
  %619 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store ptr @.str.25, ptr %619, align 8, !dbg !31
  %620 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %620, align 8, !dbg !31
  %621 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 1, ptr %621, align 8, !dbg !31
  %622 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %622, align 8, !dbg !31
  %623 = load i64, ptr %flashattn_gqa_decode_no_split.K.shape, align 8, !dbg !31, !tbaa !84
  %sext1413 = shl i64 %623, 32, !dbg !31
  %624 = ashr exact i64 %sext1413, 32, !dbg !31
  %625 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 2, !dbg !31
  store i64 %624, ptr %625, align 8, !dbg !31
  %626 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 5, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %626, i8 0, i64 16, i1 false), !dbg !31
  %627 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %628 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  %.not1414 = icmp eq ptr %628, null, !dbg !31
  br i1 %.not1414, label %handle_init208, label %handle_init_end209, !dbg !31, !prof !37

if_end207:                                        ; preds = %handle_init_end209, %if_end199
  %629 = getelementptr inbounds i64, ptr %flashattn_gqa_decode_no_split.K.shape, i64 1, !dbg !31
  %630 = load i64, ptr %629, align 8, !dbg !31, !tbaa !84
  %631 = and i64 %630, 4294967295, !dbg !31
  %.not1243 = icmp eq i64 %631, 8192, !dbg !31
  br i1 %.not1243, label %if_end215, label %if_then214, !dbg !31, !prof !37

handle_init208:                                   ; preds = %if_then206
  %632 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %633 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %634 = call i32 %633(ptr %632, ptr nonnull @.str.26, ptr nonnull %72), !dbg !31
  %635 = icmp eq i32 %634, 0, !dbg !31
  br i1 %635, label %call_end211, label %common.ret, !dbg !31, !prof !33

handle_init_end209:                               ; preds = %call_end211, %if_then206
  %636 = phi ptr [ %628, %if_then206 ], [ %639, %call_end211 ], !dbg !31
  %637 = call i32 %627(ptr %636, ptr nonnull %stack_ffi_any1228, i32 5, ptr nonnull %626), !dbg !31
  %638 = icmp eq i32 %637, 0, !dbg !31
  br i1 %638, label %if_end207, label %common.ret, !dbg !31, !prof !33

call_end211:                                      ; preds = %handle_init208
  %639 = load ptr, ptr %72, align 8, !dbg !31
  store ptr %639, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  br label %handle_init_end209, !dbg !31

if_then214:                                       ; preds = %if_end207
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %640 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %640, align 8, !dbg !31
  %641 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %641, align 8, !dbg !31
  %642 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.18, ptr %642, align 8, !dbg !31
  %643 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %643, align 8, !dbg !31
  %644 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store ptr @.str.27, ptr %644, align 8, !dbg !31
  %645 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %645, align 8, !dbg !31
  %646 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 8192, ptr %646, align 8, !dbg !31
  %647 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %647, align 8, !dbg !31
  %648 = load i64, ptr %629, align 8, !dbg !31, !tbaa !84
  %sext1411 = shl i64 %648, 32, !dbg !31
  %649 = ashr exact i64 %sext1411, 32, !dbg !31
  %650 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 2, !dbg !31
  store i64 %649, ptr %650, align 8, !dbg !31
  %651 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 5, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %651, i8 0, i64 16, i1 false), !dbg !31
  %652 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %653 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  %.not1412 = icmp eq ptr %653, null, !dbg !31
  br i1 %.not1412, label %handle_init216, label %handle_init_end217, !dbg !31, !prof !37

if_end215:                                        ; preds = %handle_init_end217, %if_end207
  %654 = getelementptr inbounds i64, ptr %flashattn_gqa_decode_no_split.K.shape, i64 2, !dbg !31
  %655 = load i64, ptr %654, align 8, !dbg !31, !tbaa !84
  %656 = and i64 %655, 4294967295, !dbg !31
  %.not1244 = icmp eq i64 %656, 8, !dbg !31
  br i1 %.not1244, label %if_end223, label %if_then222, !dbg !31, !prof !37

handle_init216:                                   ; preds = %if_then214
  %657 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %658 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %659 = call i32 %658(ptr %657, ptr nonnull @.str.26, ptr nonnull %71), !dbg !31
  %660 = icmp eq i32 %659, 0, !dbg !31
  br i1 %660, label %call_end219, label %common.ret, !dbg !31, !prof !33

handle_init_end217:                               ; preds = %call_end219, %if_then214
  %661 = phi ptr [ %653, %if_then214 ], [ %664, %call_end219 ], !dbg !31
  %662 = call i32 %652(ptr %661, ptr nonnull %stack_ffi_any1228, i32 5, ptr nonnull %651), !dbg !31
  %663 = icmp eq i32 %662, 0, !dbg !31
  br i1 %663, label %if_end215, label %common.ret, !dbg !31, !prof !33

call_end219:                                      ; preds = %handle_init216
  %664 = load ptr, ptr %71, align 8, !dbg !31
  store ptr %664, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  br label %handle_init_end217, !dbg !31

if_then222:                                       ; preds = %if_end215
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %665 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %665, align 8, !dbg !31
  %666 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %666, align 8, !dbg !31
  %667 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.18, ptr %667, align 8, !dbg !31
  %668 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %668, align 8, !dbg !31
  %669 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store ptr @.str.28, ptr %669, align 8, !dbg !31
  %670 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %670, align 8, !dbg !31
  %671 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 8, ptr %671, align 8, !dbg !31
  %672 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %672, align 8, !dbg !31
  %673 = load i64, ptr %654, align 8, !dbg !31, !tbaa !84
  %sext1409 = shl i64 %673, 32, !dbg !31
  %674 = ashr exact i64 %sext1409, 32, !dbg !31
  %675 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 2, !dbg !31
  store i64 %674, ptr %675, align 8, !dbg !31
  %676 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 5, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %676, i8 0, i64 16, i1 false), !dbg !31
  %677 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %678 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  %.not1410 = icmp eq ptr %678, null, !dbg !31
  br i1 %.not1410, label %handle_init224, label %handle_init_end225, !dbg !31, !prof !37

if_end223:                                        ; preds = %handle_init_end225, %if_end215
  %679 = getelementptr inbounds i64, ptr %flashattn_gqa_decode_no_split.K.shape, i64 3, !dbg !31
  %680 = load i64, ptr %679, align 8, !dbg !31, !tbaa !84
  %681 = and i64 %680, 4294967295, !dbg !31
  %.not1245 = icmp eq i64 %681, 128, !dbg !31
  br i1 %.not1245, label %if_end231, label %if_then230, !dbg !31, !prof !37

handle_init224:                                   ; preds = %if_then222
  %682 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %683 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %684 = call i32 %683(ptr %682, ptr nonnull @.str.26, ptr nonnull %70), !dbg !31
  %685 = icmp eq i32 %684, 0, !dbg !31
  br i1 %685, label %call_end227, label %common.ret, !dbg !31, !prof !33

handle_init_end225:                               ; preds = %call_end227, %if_then222
  %686 = phi ptr [ %678, %if_then222 ], [ %689, %call_end227 ], !dbg !31
  %687 = call i32 %677(ptr %686, ptr nonnull %stack_ffi_any1228, i32 5, ptr nonnull %676), !dbg !31
  %688 = icmp eq i32 %687, 0, !dbg !31
  br i1 %688, label %if_end223, label %common.ret, !dbg !31, !prof !33

call_end227:                                      ; preds = %handle_init224
  %689 = load ptr, ptr %70, align 8, !dbg !31
  store ptr %689, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  br label %handle_init_end225, !dbg !31

if_then230:                                       ; preds = %if_end223
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %690 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %690, align 8, !dbg !31
  %691 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %691, align 8, !dbg !31
  %692 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.18, ptr %692, align 8, !dbg !31
  %693 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %693, align 8, !dbg !31
  %694 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store ptr @.str.36, ptr %694, align 8, !dbg !31
  %695 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %695, align 8, !dbg !31
  %696 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 128, ptr %696, align 8, !dbg !31
  %697 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %697, align 8, !dbg !31
  %698 = load i64, ptr %679, align 8, !dbg !31, !tbaa !84
  %sext1407 = shl i64 %698, 32, !dbg !31
  %699 = ashr exact i64 %sext1407, 32, !dbg !31
  %700 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 2, !dbg !31
  store i64 %699, ptr %700, align 8, !dbg !31
  %701 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 5, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %701, i8 0, i64 16, i1 false), !dbg !31
  %702 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %703 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  %.not1408 = icmp eq ptr %703, null, !dbg !31
  br i1 %.not1408, label %handle_init232, label %handle_init_end233, !dbg !31, !prof !37

if_end231:                                        ; preds = %handle_init_end233, %if_end223
  %704 = icmp eq ptr %flashattn_gqa_decode_no_split.K.strides, null, !dbg !31
  br i1 %704, label %if_then255, label %if_end240, !dbg !31

handle_init232:                                   ; preds = %if_then230
  %705 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %706 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %707 = call i32 %706(ptr %705, ptr nonnull @.str.26, ptr nonnull %69), !dbg !31
  %708 = icmp eq i32 %707, 0, !dbg !31
  br i1 %708, label %call_end235, label %common.ret, !dbg !31, !prof !33

handle_init_end233:                               ; preds = %call_end235, %if_then230
  %709 = phi ptr [ %703, %if_then230 ], [ %712, %call_end235 ], !dbg !31
  %710 = call i32 %702(ptr %709, ptr nonnull %stack_ffi_any1228, i32 5, ptr nonnull %701), !dbg !31
  %711 = icmp eq i32 %710, 0, !dbg !31
  br i1 %711, label %if_end231, label %common.ret, !dbg !31, !prof !33

call_end235:                                      ; preds = %handle_init232
  %712 = load ptr, ptr %69, align 8, !dbg !31
  store ptr %712, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  br label %handle_init_end233, !dbg !31

if_end240:                                        ; preds = %if_end231
  %713 = getelementptr inbounds i64, ptr %flashattn_gqa_decode_no_split.K.strides, i64 3, !dbg !31
  %714 = load i64, ptr %713, align 8, !dbg !31, !tbaa !84
  %715 = and i64 %714, 4294967295, !dbg !31
  %.not1246 = icmp eq i64 %715, 1, !dbg !31
  br i1 %.not1246, label %if_end254, label %if_end245, !dbg !31, !prof !37

if_end245:                                        ; preds = %if_end240
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %716 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %716, align 8, !dbg !31
  %717 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %717, align 8, !dbg !31
  %718 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.18, ptr %718, align 8, !dbg !31
  %719 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %719, align 8, !dbg !31
  %720 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store ptr @.str.37, ptr %720, align 8, !dbg !31
  %721 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %721, align 8, !dbg !31
  %722 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 1, ptr %722, align 8, !dbg !31
  %723 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %723, align 8, !dbg !31
  %724 = load i64, ptr %713, align 8, !dbg !31, !tbaa !84
  %sext1405 = shl i64 %724, 32, !dbg !31
  %725 = ashr exact i64 %sext1405, 32, !dbg !31
  %726 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 2, !dbg !31
  store i64 %725, ptr %726, align 8, !dbg !31
  %727 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 5, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %727, i8 0, i64 16, i1 false), !dbg !31
  %728 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %729 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  %.not1406 = icmp eq ptr %729, null, !dbg !31
  br i1 %.not1406, label %handle_init246, label %handle_init_end247, !dbg !31, !prof !37

handle_init246:                                   ; preds = %if_end245
  %730 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %731 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %732 = call i32 %731(ptr %730, ptr nonnull @.str.26, ptr nonnull %68), !dbg !31
  %733 = icmp eq i32 %732, 0, !dbg !31
  br i1 %733, label %call_end249, label %common.ret, !dbg !31, !prof !33

handle_init_end247:                               ; preds = %call_end249, %if_end245
  %734 = phi ptr [ %729, %if_end245 ], [ %737, %call_end249 ], !dbg !31
  %735 = call i32 %728(ptr %734, ptr nonnull %stack_ffi_any1228, i32 5, ptr nonnull %727), !dbg !31
  %736 = icmp eq i32 %735, 0, !dbg !31
  br i1 %736, label %if_end254, label %common.ret, !dbg !31, !prof !33

call_end249:                                      ; preds = %handle_init246
  %737 = load ptr, ptr %68, align 8, !dbg !31
  store ptr %737, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  br label %handle_init_end247, !dbg !31

if_end254:                                        ; preds = %if_end240, %handle_init_end247
  %738 = getelementptr inbounds i64, ptr %flashattn_gqa_decode_no_split.K.strides, i64 2, !dbg !31
  %739 = load i64, ptr %738, align 8, !dbg !31, !tbaa !84
  %740 = and i64 %739, 4294967295, !dbg !31
  %.not1247 = icmp eq i64 %740, 128, !dbg !31
  br i1 %.not1247, label %if_end268, label %if_then255, !dbg !31, !prof !37

if_then255:                                       ; preds = %if_end231, %if_end254
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %741 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %741, align 8, !dbg !31
  %742 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %742, align 8, !dbg !31
  %743 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.18, ptr %743, align 8, !dbg !31
  %744 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %744, align 8, !dbg !31
  %745 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store ptr @.str.29, ptr %745, align 8, !dbg !31
  %746 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %746, align 8, !dbg !31
  %747 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 128, ptr %747, align 8, !dbg !31
  %748 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %748, align 8, !dbg !31
  br i1 %704, label %if_end259, label %if_else258, !dbg !31

if_end256:                                        ; preds = %handle_init_end261
  br i1 %704, label %if_then269, label %if_end268, !dbg !31

if_else258:                                       ; preds = %if_then255
  %749 = getelementptr inbounds i64, ptr %flashattn_gqa_decode_no_split.K.strides, i64 2, !dbg !31
  %750 = load i64, ptr %749, align 8, !dbg !31, !tbaa !84
  br label %if_end259, !dbg !31

if_end259:                                        ; preds = %if_then255, %if_else258
  %751 = phi i64 [ %750, %if_else258 ], [ 1, %if_then255 ], !dbg !31
  %sext1403 = shl i64 %751, 32, !dbg !31
  %752 = ashr exact i64 %sext1403, 32, !dbg !31
  %753 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 2, !dbg !31
  store i64 %752, ptr %753, align 8, !dbg !31
  %754 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 5, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %754, i8 0, i64 16, i1 false), !dbg !31
  %755 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %756 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  %.not1404 = icmp eq ptr %756, null, !dbg !31
  br i1 %.not1404, label %handle_init260, label %handle_init_end261, !dbg !31, !prof !37

handle_init260:                                   ; preds = %if_end259
  %757 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %758 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %759 = call i32 %758(ptr %757, ptr nonnull @.str.26, ptr nonnull %67), !dbg !31
  %760 = icmp eq i32 %759, 0, !dbg !31
  br i1 %760, label %call_end263, label %common.ret, !dbg !31, !prof !33

handle_init_end261:                               ; preds = %call_end263, %if_end259
  %761 = phi ptr [ %756, %if_end259 ], [ %764, %call_end263 ], !dbg !31
  %762 = call i32 %755(ptr %761, ptr nonnull %stack_ffi_any1228, i32 5, ptr nonnull %754), !dbg !31
  %763 = icmp eq i32 %762, 0, !dbg !31
  br i1 %763, label %if_end256, label %common.ret, !dbg !31, !prof !33

call_end263:                                      ; preds = %handle_init260
  %764 = load ptr, ptr %67, align 8, !dbg !31
  store ptr %764, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  br label %handle_init_end261, !dbg !31

if_end268:                                        ; preds = %if_end254, %if_end256
  %765 = getelementptr inbounds i64, ptr %flashattn_gqa_decode_no_split.K.strides, i64 1, !dbg !31
  %766 = load i64, ptr %765, align 8, !dbg !31, !tbaa !84
  %767 = and i64 %766, 4294967295, !dbg !31
  %.not1248 = icmp eq i64 %767, 1024, !dbg !31
  br i1 %.not1248, label %if_end282, label %if_then269, !dbg !31, !prof !37

if_then269:                                       ; preds = %if_end256, %if_end268
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %768 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %768, align 8, !dbg !31
  %769 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %769, align 8, !dbg !31
  %770 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.18, ptr %770, align 8, !dbg !31
  %771 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %771, align 8, !dbg !31
  %772 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store ptr @.str.30, ptr %772, align 8, !dbg !31
  %773 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %773, align 8, !dbg !31
  %774 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 1024, ptr %774, align 8, !dbg !31
  %775 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %775, align 8, !dbg !31
  br i1 %704, label %if_end273, label %if_else272, !dbg !31

if_end270:                                        ; preds = %handle_init_end275
  br i1 %704, label %if_then283, label %if_end282, !dbg !31

if_else272:                                       ; preds = %if_then269
  %776 = getelementptr inbounds i64, ptr %flashattn_gqa_decode_no_split.K.strides, i64 1, !dbg !31
  %777 = load i64, ptr %776, align 8, !dbg !31, !tbaa !84
  br label %if_end273, !dbg !31

if_end273:                                        ; preds = %if_then269, %if_else272
  %778 = phi i64 [ %777, %if_else272 ], [ 1, %if_then269 ], !dbg !31
  %sext1401 = shl i64 %778, 32, !dbg !31
  %779 = ashr exact i64 %sext1401, 32, !dbg !31
  %780 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 2, !dbg !31
  store i64 %779, ptr %780, align 8, !dbg !31
  %781 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 5, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %781, i8 0, i64 16, i1 false), !dbg !31
  %782 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %783 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  %.not1402 = icmp eq ptr %783, null, !dbg !31
  br i1 %.not1402, label %handle_init274, label %handle_init_end275, !dbg !31, !prof !37

handle_init274:                                   ; preds = %if_end273
  %784 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %785 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %786 = call i32 %785(ptr %784, ptr nonnull @.str.26, ptr nonnull %66), !dbg !31
  %787 = icmp eq i32 %786, 0, !dbg !31
  br i1 %787, label %call_end277, label %common.ret, !dbg !31, !prof !33

handle_init_end275:                               ; preds = %call_end277, %if_end273
  %788 = phi ptr [ %783, %if_end273 ], [ %791, %call_end277 ], !dbg !31
  %789 = call i32 %782(ptr %788, ptr nonnull %stack_ffi_any1228, i32 5, ptr nonnull %781), !dbg !31
  %790 = icmp eq i32 %789, 0, !dbg !31
  br i1 %790, label %if_end270, label %common.ret, !dbg !31, !prof !33

call_end277:                                      ; preds = %handle_init274
  %791 = load ptr, ptr %66, align 8, !dbg !31
  store ptr %791, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  br label %handle_init_end275, !dbg !31

if_end282:                                        ; preds = %if_end268, %if_end270
  %792 = load i64, ptr %flashattn_gqa_decode_no_split.K.strides, align 8, !dbg !31, !tbaa !84
  %793 = and i64 %792, 4294967295, !dbg !31
  %.not1249 = icmp eq i64 %793, 8388608, !dbg !31
  br i1 %.not1249, label %if_end284, label %if_then283, !dbg !31, !prof !37

if_then283:                                       ; preds = %if_end270, %if_end282
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %794 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %794, align 8, !dbg !31
  %795 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %795, align 8, !dbg !31
  %796 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.18, ptr %796, align 8, !dbg !31
  %797 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %797, align 8, !dbg !31
  %798 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store ptr @.str.31, ptr %798, align 8, !dbg !31
  %799 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %799, align 8, !dbg !31
  %800 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 8388608, ptr %800, align 8, !dbg !31
  %801 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %801, align 8, !dbg !31
  br i1 %704, label %if_end287, label %if_else286, !dbg !31

if_end284:                                        ; preds = %handle_init_end289, %if_end282
  %802 = getelementptr inbounds %1, ptr %K_handle, i64 0, i32 6, !dbg !31
  %803 = load i64, ptr %802, align 8, !dbg !31
  %.not1250 = icmp eq i64 %803, 0, !dbg !31
  br i1 %.not1250, label %if_end295, label %if_then294, !dbg !31, !prof !37

if_else286:                                       ; preds = %if_then283
  %804 = load i64, ptr %flashattn_gqa_decode_no_split.K.strides, align 8, !dbg !31, !tbaa !84
  br label %if_end287, !dbg !31

if_end287:                                        ; preds = %if_then283, %if_else286
  %805 = phi i64 [ %804, %if_else286 ], [ 1, %if_then283 ], !dbg !31
  %sext1399 = shl i64 %805, 32, !dbg !31
  %806 = ashr exact i64 %sext1399, 32, !dbg !31
  %807 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 2, !dbg !31
  store i64 %806, ptr %807, align 8, !dbg !31
  %808 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 5, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %808, i8 0, i64 16, i1 false), !dbg !31
  %809 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %810 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  %.not1400 = icmp eq ptr %810, null, !dbg !31
  br i1 %.not1400, label %handle_init288, label %handle_init_end289, !dbg !31, !prof !37

handle_init288:                                   ; preds = %if_end287
  %811 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %812 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %813 = call i32 %812(ptr %811, ptr nonnull @.str.26, ptr nonnull %65), !dbg !31
  %814 = icmp eq i32 %813, 0, !dbg !31
  br i1 %814, label %call_end291, label %common.ret, !dbg !31, !prof !33

handle_init_end289:                               ; preds = %call_end291, %if_end287
  %815 = phi ptr [ %810, %if_end287 ], [ %818, %call_end291 ], !dbg !31
  %816 = call i32 %809(ptr %815, ptr nonnull %stack_ffi_any1228, i32 5, ptr nonnull %808), !dbg !31
  %817 = icmp eq i32 %816, 0, !dbg !31
  br i1 %817, label %if_end284, label %common.ret, !dbg !31, !prof !33

call_end291:                                      ; preds = %handle_init288
  %818 = load ptr, ptr %65, align 8, !dbg !31
  store ptr %818, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  br label %handle_init_end289, !dbg !31

if_then294:                                       ; preds = %if_end284
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %819 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %819, align 8, !dbg !31
  %820 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %820, align 8, !dbg !31
  %821 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.18, ptr %821, align 8, !dbg !31
  %822 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %822, align 8, !dbg !31
  %823 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store i64 0, ptr %823, align 8, !dbg !31
  %824 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %824, align 8, !dbg !31
  %825 = load i64, ptr %802, align 8, !dbg !31
  %826 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 %825, ptr %826, align 8, !dbg !31
  %827 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %827, i8 0, i64 16, i1 false), !dbg !31
  %828 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %829 = load ptr, ptr @.tvm_func.__tvm_error_byte_offset_mismatch, align 8, !dbg !31
  %.not1398 = icmp eq ptr %829, null, !dbg !31
  br i1 %.not1398, label %handle_init296, label %handle_init_end297, !dbg !31, !prof !37

if_end295:                                        ; preds = %handle_init_end297, %if_end284
  %830 = getelementptr inbounds %1, ptr %K_handle, i64 0, i32 1, i32 1, !dbg !31
  %831 = load i32, ptr %830, align 4, !dbg !31
  %832 = load i32, ptr %164, align 4, !dbg !31
  %.not1251 = icmp eq i32 %831, %832, !dbg !31
  br i1 %.not1251, label %if_end303, label %if_then302, !dbg !31, !prof !37

handle_init296:                                   ; preds = %if_then294
  %833 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %834 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %835 = call i32 %834(ptr %833, ptr nonnull @.str.32, ptr nonnull %64), !dbg !31
  %836 = icmp eq i32 %835, 0, !dbg !31
  br i1 %836, label %call_end299, label %common.ret, !dbg !31, !prof !33

handle_init_end297:                               ; preds = %call_end299, %if_then294
  %837 = phi ptr [ %829, %if_then294 ], [ %840, %call_end299 ], !dbg !31
  %838 = call i32 %828(ptr %837, ptr nonnull %stack_ffi_any1228, i32 4, ptr nonnull %827), !dbg !31
  %839 = icmp eq i32 %838, 0, !dbg !31
  br i1 %839, label %if_end295, label %common.ret, !dbg !31, !prof !33

call_end299:                                      ; preds = %handle_init296
  %840 = load ptr, ptr %64, align 8, !dbg !31
  store ptr %840, ptr @.tvm_func.__tvm_error_byte_offset_mismatch, align 8, !dbg !31
  br label %handle_init_end297, !dbg !31

if_then302:                                       ; preds = %if_end295
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %841 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %841, align 8, !dbg !31
  %842 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %842, align 8, !dbg !31
  %843 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.18, ptr %843, align 8, !dbg !31
  %844 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %844, align 8, !dbg !31
  %845 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store ptr @.str.38, ptr %845, align 8, !dbg !31
  %846 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %846, align 8, !dbg !31
  %847 = load i32, ptr %164, align 4, !dbg !31
  %848 = sext i32 %847 to i64, !dbg !31
  %849 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 %848, ptr %849, align 8, !dbg !31
  %850 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %850, align 8, !dbg !31
  %851 = load i32, ptr %830, align 4, !dbg !31
  %852 = sext i32 %851 to i64, !dbg !31
  %853 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 2, !dbg !31
  store i64 %852, ptr %853, align 8, !dbg !31
  %854 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 5, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %854, i8 0, i64 16, i1 false), !dbg !31
  %855 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %856 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  %.not1397 = icmp eq ptr %856, null, !dbg !31
  br i1 %.not1397, label %handle_init304, label %handle_init_end305, !dbg !31, !prof !37

if_end303:                                        ; preds = %handle_init_end305, %if_end295
  %857 = getelementptr inbounds %1, ptr %K_handle, i64 0, i32 1, i32 0, !dbg !31
  %858 = load i32, ptr %857, align 4, !dbg !31
  %.not1252 = icmp eq i32 %858, 2, !dbg !31
  br i1 %.not1252, label %if_end311, label %if_then310, !dbg !31, !prof !37

handle_init304:                                   ; preds = %if_then302
  %859 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %860 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %861 = call i32 %860(ptr %859, ptr nonnull @.str.26, ptr nonnull %63), !dbg !31
  %862 = icmp eq i32 %861, 0, !dbg !31
  br i1 %862, label %call_end307, label %common.ret, !dbg !31, !prof !33

handle_init_end305:                               ; preds = %call_end307, %if_then302
  %863 = phi ptr [ %856, %if_then302 ], [ %866, %call_end307 ], !dbg !31
  %864 = call i32 %855(ptr %863, ptr nonnull %stack_ffi_any1228, i32 5, ptr nonnull %854), !dbg !31
  %865 = icmp eq i32 %864, 0, !dbg !31
  br i1 %865, label %if_end303, label %common.ret, !dbg !31, !prof !33

call_end307:                                      ; preds = %handle_init304
  %866 = load ptr, ptr %63, align 8, !dbg !31
  store ptr %866, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  br label %handle_init_end305, !dbg !31

if_then310:                                       ; preds = %if_end303
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %867 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %867, align 8, !dbg !31
  %868 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %868, align 8, !dbg !31
  %869 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.18, ptr %869, align 8, !dbg !31
  %870 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %870, align 8, !dbg !31
  %871 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store i64 2, ptr %871, align 8, !dbg !31
  %872 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %872, align 8, !dbg !31
  %873 = load i32, ptr %857, align 4, !dbg !31
  %874 = sext i32 %873 to i64, !dbg !31
  %875 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 %874, ptr %875, align 8, !dbg !31
  %876 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %876, i8 0, i64 16, i1 false), !dbg !31
  %877 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %878 = load ptr, ptr @.tvm_func.__tvm_error_device_type_mismatch, align 8, !dbg !31
  %.not1396 = icmp eq ptr %878, null, !dbg !31
  br i1 %.not1396, label %handle_init312, label %handle_init_end313, !dbg !31, !prof !37

if_end311:                                        ; preds = %handle_init_end313, %if_end303
  %879 = icmp eq ptr %K, null, !dbg !31
  br i1 %879, label %if_then318, label %if_end319, !dbg !31, !prof !33

handle_init312:                                   ; preds = %if_then310
  %880 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %881 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %882 = call i32 %881(ptr %880, ptr nonnull @.str.33, ptr nonnull %62), !dbg !31
  %883 = icmp eq i32 %882, 0, !dbg !31
  br i1 %883, label %call_end315, label %common.ret, !dbg !31, !prof !33

handle_init_end313:                               ; preds = %call_end315, %if_then310
  %884 = phi ptr [ %878, %if_then310 ], [ %887, %call_end315 ], !dbg !31
  %885 = call i32 %877(ptr %884, ptr nonnull %stack_ffi_any1228, i32 4, ptr nonnull %876), !dbg !31
  %886 = icmp eq i32 %885, 0, !dbg !31
  br i1 %886, label %if_end311, label %common.ret, !dbg !31, !prof !33

call_end315:                                      ; preds = %handle_init312
  %887 = load ptr, ptr %62, align 8, !dbg !31
  store ptr %887, ptr @.tvm_func.__tvm_error_device_type_mismatch, align 8, !dbg !31
  br label %handle_init_end313, !dbg !31

if_then318:                                       ; preds = %if_end311
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %888 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %888, align 8, !dbg !31
  %889 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %889, align 8, !dbg !31
  %890 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.18, ptr %890, align 8, !dbg !31
  %891 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %891, align 8, !dbg !31
  %892 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store ptr @.str.34, ptr %892, align 8, !dbg !31
  %893 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %893, i8 0, i64 16, i1 false), !dbg !31
  %894 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %895 = load ptr, ptr @.tvm_func.__tvm_error_null_ptr, align 8, !dbg !31
  %.not1395 = icmp eq ptr %895, null, !dbg !31
  br i1 %.not1395, label %handle_init320, label %handle_init_end321, !dbg !31, !prof !37

if_end319:                                        ; preds = %handle_init_end321, %if_end311
  %896 = getelementptr inbounds %1, ptr %V_handle, i64 0, i32 3, i32 2, !dbg !31
  %897 = load i16, ptr %896, align 2, !dbg !31
  %898 = icmp ne i16 %897, 1, !dbg !31
  %899 = getelementptr inbounds %1, ptr %V_handle, i64 0, i32 3, i32 1, !dbg !31
  %900 = load i8, ptr %899, align 1, !dbg !31
  %901 = icmp ne i8 %900, 16, !dbg !31
  %902 = getelementptr inbounds %1, ptr %V_handle, i64 0, i32 3, i32 0, !dbg !31
  %903 = load i8, ptr %902, align 1, !dbg !31
  %904 = icmp ne i8 %903, 2, !dbg !31
  %905 = or i1 %901, %904, !dbg !31
  %906 = or i1 %898, %905, !dbg !31
  br i1 %906, label %if_then326, label %if_end327, !dbg !31, !prof !33

handle_init320:                                   ; preds = %if_then318
  %907 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %908 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %909 = call i32 %908(ptr %907, ptr nonnull @.str.35, ptr nonnull %61), !dbg !31
  %910 = icmp eq i32 %909, 0, !dbg !31
  br i1 %910, label %call_end323, label %common.ret, !dbg !31, !prof !33

handle_init_end321:                               ; preds = %call_end323, %if_then318
  %911 = phi ptr [ %895, %if_then318 ], [ %914, %call_end323 ], !dbg !31
  %912 = call i32 %894(ptr %911, ptr nonnull %stack_ffi_any1228, i32 3, ptr nonnull %893), !dbg !31
  %913 = icmp eq i32 %912, 0, !dbg !31
  br i1 %913, label %if_end319, label %common.ret, !dbg !31, !prof !33

call_end323:                                      ; preds = %handle_init320
  %914 = load ptr, ptr %61, align 8, !dbg !31
  store ptr %914, ptr @.tvm_func.__tvm_error_null_ptr, align 8, !dbg !31
  br label %handle_init_end321, !dbg !31

if_then326:                                       ; preds = %if_end319
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %915 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %915, align 8, !dbg !31
  %916 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %916, align 8, !dbg !31
  %917 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.19, ptr %917, align 8, !dbg !31
  %918 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %918, align 8, !dbg !31
  %919 = load i8, ptr %902, align 1, !dbg !31
  %920 = zext i8 %919 to i64, !dbg !31
  %921 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store i64 %920, ptr %921, align 8, !dbg !31
  %922 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %922, align 8, !dbg !31
  %923 = load i8, ptr %899, align 1, !dbg !31
  %924 = zext i8 %923 to i64, !dbg !31
  %925 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 %924, ptr %925, align 8, !dbg !31
  %926 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %926, align 8, !dbg !31
  %927 = load i16, ptr %896, align 2, !dbg !31
  %928 = zext i16 %927 to i64, !dbg !31
  %929 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 2, !dbg !31
  store i64 %928, ptr %929, align 8, !dbg !31
  %930 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 5, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %930, align 8, !dbg !31
  %931 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 5, i32 2, !dbg !31
  store i64 2, ptr %931, align 8, !dbg !31
  %932 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 6, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %932, align 8, !dbg !31
  %933 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 6, i32 2, !dbg !31
  store i64 16, ptr %933, align 8, !dbg !31
  %934 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 7, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %934, align 8, !dbg !31
  %935 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 7, i32 2, !dbg !31
  store i64 1, ptr %935, align 8, !dbg !31
  %936 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 8, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %936, i8 0, i64 16, i1 false), !dbg !31
  %937 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %938 = load ptr, ptr @.tvm_func.__tvm_error_dtype_mismatch, align 8, !dbg !31
  %.not1394 = icmp eq ptr %938, null, !dbg !31
  br i1 %.not1394, label %handle_init328, label %handle_init_end329, !dbg !31, !prof !37

if_end327:                                        ; preds = %handle_init_end329, %if_end319
  %939 = load i64, ptr %flashattn_gqa_decode_no_split.V.shape, align 8, !dbg !31, !tbaa !84
  %940 = and i64 %939, 4294967295, !dbg !31
  %.not1253 = icmp eq i64 %940, 1, !dbg !31
  br i1 %.not1253, label %if_end335, label %if_then334, !dbg !31, !prof !37

handle_init328:                                   ; preds = %if_then326
  %941 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %942 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %943 = call i32 %942(ptr %941, ptr nonnull @.str.24, ptr nonnull %60), !dbg !31
  %944 = icmp eq i32 %943, 0, !dbg !31
  br i1 %944, label %call_end331, label %common.ret, !dbg !31, !prof !33

handle_init_end329:                               ; preds = %call_end331, %if_then326
  %945 = phi ptr [ %938, %if_then326 ], [ %948, %call_end331 ], !dbg !31
  %946 = call i32 %937(ptr %945, ptr nonnull %stack_ffi_any1228, i32 8, ptr nonnull %936), !dbg !31
  %947 = icmp eq i32 %946, 0, !dbg !31
  br i1 %947, label %if_end327, label %common.ret, !dbg !31, !prof !33

call_end331:                                      ; preds = %handle_init328
  %948 = load ptr, ptr %60, align 8, !dbg !31
  store ptr %948, ptr @.tvm_func.__tvm_error_dtype_mismatch, align 8, !dbg !31
  br label %handle_init_end329, !dbg !31

if_then334:                                       ; preds = %if_end327
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %949 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %949, align 8, !dbg !31
  %950 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %950, align 8, !dbg !31
  %951 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.19, ptr %951, align 8, !dbg !31
  %952 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %952, align 8, !dbg !31
  %953 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store ptr @.str.25, ptr %953, align 8, !dbg !31
  %954 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %954, align 8, !dbg !31
  %955 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 1, ptr %955, align 8, !dbg !31
  %956 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %956, align 8, !dbg !31
  %957 = load i64, ptr %flashattn_gqa_decode_no_split.V.shape, align 8, !dbg !31, !tbaa !84
  %sext1392 = shl i64 %957, 32, !dbg !31
  %958 = ashr exact i64 %sext1392, 32, !dbg !31
  %959 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 2, !dbg !31
  store i64 %958, ptr %959, align 8, !dbg !31
  %960 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 5, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %960, i8 0, i64 16, i1 false), !dbg !31
  %961 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %962 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  %.not1393 = icmp eq ptr %962, null, !dbg !31
  br i1 %.not1393, label %handle_init336, label %handle_init_end337, !dbg !31, !prof !37

if_end335:                                        ; preds = %handle_init_end337, %if_end327
  %963 = getelementptr inbounds i64, ptr %flashattn_gqa_decode_no_split.V.shape, i64 1, !dbg !31
  %964 = load i64, ptr %963, align 8, !dbg !31, !tbaa !84
  %965 = and i64 %964, 4294967295, !dbg !31
  %.not1254 = icmp eq i64 %965, 8192, !dbg !31
  br i1 %.not1254, label %if_end343, label %if_then342, !dbg !31, !prof !37

handle_init336:                                   ; preds = %if_then334
  %966 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %967 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %968 = call i32 %967(ptr %966, ptr nonnull @.str.26, ptr nonnull %59), !dbg !31
  %969 = icmp eq i32 %968, 0, !dbg !31
  br i1 %969, label %call_end339, label %common.ret, !dbg !31, !prof !33

handle_init_end337:                               ; preds = %call_end339, %if_then334
  %970 = phi ptr [ %962, %if_then334 ], [ %973, %call_end339 ], !dbg !31
  %971 = call i32 %961(ptr %970, ptr nonnull %stack_ffi_any1228, i32 5, ptr nonnull %960), !dbg !31
  %972 = icmp eq i32 %971, 0, !dbg !31
  br i1 %972, label %if_end335, label %common.ret, !dbg !31, !prof !33

call_end339:                                      ; preds = %handle_init336
  %973 = load ptr, ptr %59, align 8, !dbg !31
  store ptr %973, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  br label %handle_init_end337, !dbg !31

if_then342:                                       ; preds = %if_end335
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %974 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %974, align 8, !dbg !31
  %975 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %975, align 8, !dbg !31
  %976 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.19, ptr %976, align 8, !dbg !31
  %977 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %977, align 8, !dbg !31
  %978 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store ptr @.str.27, ptr %978, align 8, !dbg !31
  %979 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %979, align 8, !dbg !31
  %980 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 8192, ptr %980, align 8, !dbg !31
  %981 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %981, align 8, !dbg !31
  %982 = load i64, ptr %963, align 8, !dbg !31, !tbaa !84
  %sext1390 = shl i64 %982, 32, !dbg !31
  %983 = ashr exact i64 %sext1390, 32, !dbg !31
  %984 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 2, !dbg !31
  store i64 %983, ptr %984, align 8, !dbg !31
  %985 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 5, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %985, i8 0, i64 16, i1 false), !dbg !31
  %986 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %987 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  %.not1391 = icmp eq ptr %987, null, !dbg !31
  br i1 %.not1391, label %handle_init344, label %handle_init_end345, !dbg !31, !prof !37

if_end343:                                        ; preds = %handle_init_end345, %if_end335
  %988 = getelementptr inbounds i64, ptr %flashattn_gqa_decode_no_split.V.shape, i64 2, !dbg !31
  %989 = load i64, ptr %988, align 8, !dbg !31, !tbaa !84
  %990 = and i64 %989, 4294967295, !dbg !31
  %.not1255 = icmp eq i64 %990, 8, !dbg !31
  br i1 %.not1255, label %if_end351, label %if_then350, !dbg !31, !prof !37

handle_init344:                                   ; preds = %if_then342
  %991 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %992 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %993 = call i32 %992(ptr %991, ptr nonnull @.str.26, ptr nonnull %58), !dbg !31
  %994 = icmp eq i32 %993, 0, !dbg !31
  br i1 %994, label %call_end347, label %common.ret, !dbg !31, !prof !33

handle_init_end345:                               ; preds = %call_end347, %if_then342
  %995 = phi ptr [ %987, %if_then342 ], [ %998, %call_end347 ], !dbg !31
  %996 = call i32 %986(ptr %995, ptr nonnull %stack_ffi_any1228, i32 5, ptr nonnull %985), !dbg !31
  %997 = icmp eq i32 %996, 0, !dbg !31
  br i1 %997, label %if_end343, label %common.ret, !dbg !31, !prof !33

call_end347:                                      ; preds = %handle_init344
  %998 = load ptr, ptr %58, align 8, !dbg !31
  store ptr %998, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  br label %handle_init_end345, !dbg !31

if_then350:                                       ; preds = %if_end343
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %999 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %999, align 8, !dbg !31
  %1000 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %1000, align 8, !dbg !31
  %1001 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.19, ptr %1001, align 8, !dbg !31
  %1002 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %1002, align 8, !dbg !31
  %1003 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store ptr @.str.28, ptr %1003, align 8, !dbg !31
  %1004 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1004, align 8, !dbg !31
  %1005 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 8, ptr %1005, align 8, !dbg !31
  %1006 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1006, align 8, !dbg !31
  %1007 = load i64, ptr %988, align 8, !dbg !31, !tbaa !84
  %sext1388 = shl i64 %1007, 32, !dbg !31
  %1008 = ashr exact i64 %sext1388, 32, !dbg !31
  %1009 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 2, !dbg !31
  store i64 %1008, ptr %1009, align 8, !dbg !31
  %1010 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 5, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %1010, i8 0, i64 16, i1 false), !dbg !31
  %1011 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %1012 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  %.not1389 = icmp eq ptr %1012, null, !dbg !31
  br i1 %.not1389, label %handle_init352, label %handle_init_end353, !dbg !31, !prof !37

if_end351:                                        ; preds = %handle_init_end353, %if_end343
  %1013 = getelementptr inbounds i64, ptr %flashattn_gqa_decode_no_split.V.shape, i64 3, !dbg !31
  %1014 = load i64, ptr %1013, align 8, !dbg !31, !tbaa !84
  %1015 = and i64 %1014, 4294967295, !dbg !31
  %.not1256 = icmp eq i64 %1015, 128, !dbg !31
  br i1 %.not1256, label %if_end359, label %if_then358, !dbg !31, !prof !37

handle_init352:                                   ; preds = %if_then350
  %1016 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %1017 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %1018 = call i32 %1017(ptr %1016, ptr nonnull @.str.26, ptr nonnull %57), !dbg !31
  %1019 = icmp eq i32 %1018, 0, !dbg !31
  br i1 %1019, label %call_end355, label %common.ret, !dbg !31, !prof !33

handle_init_end353:                               ; preds = %call_end355, %if_then350
  %1020 = phi ptr [ %1012, %if_then350 ], [ %1023, %call_end355 ], !dbg !31
  %1021 = call i32 %1011(ptr %1020, ptr nonnull %stack_ffi_any1228, i32 5, ptr nonnull %1010), !dbg !31
  %1022 = icmp eq i32 %1021, 0, !dbg !31
  br i1 %1022, label %if_end351, label %common.ret, !dbg !31, !prof !33

call_end355:                                      ; preds = %handle_init352
  %1023 = load ptr, ptr %57, align 8, !dbg !31
  store ptr %1023, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  br label %handle_init_end353, !dbg !31

if_then358:                                       ; preds = %if_end351
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %1024 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %1024, align 8, !dbg !31
  %1025 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %1025, align 8, !dbg !31
  %1026 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.19, ptr %1026, align 8, !dbg !31
  %1027 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %1027, align 8, !dbg !31
  %1028 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store ptr @.str.36, ptr %1028, align 8, !dbg !31
  %1029 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1029, align 8, !dbg !31
  %1030 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 128, ptr %1030, align 8, !dbg !31
  %1031 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1031, align 8, !dbg !31
  %1032 = load i64, ptr %1013, align 8, !dbg !31, !tbaa !84
  %sext1386 = shl i64 %1032, 32, !dbg !31
  %1033 = ashr exact i64 %sext1386, 32, !dbg !31
  %1034 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 2, !dbg !31
  store i64 %1033, ptr %1034, align 8, !dbg !31
  %1035 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 5, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %1035, i8 0, i64 16, i1 false), !dbg !31
  %1036 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %1037 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  %.not1387 = icmp eq ptr %1037, null, !dbg !31
  br i1 %.not1387, label %handle_init360, label %handle_init_end361, !dbg !31, !prof !37

if_end359:                                        ; preds = %handle_init_end361, %if_end351
  %1038 = icmp eq ptr %flashattn_gqa_decode_no_split.V.strides, null, !dbg !31
  br i1 %1038, label %if_then383, label %if_end368, !dbg !31

handle_init360:                                   ; preds = %if_then358
  %1039 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %1040 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %1041 = call i32 %1040(ptr %1039, ptr nonnull @.str.26, ptr nonnull %56), !dbg !31
  %1042 = icmp eq i32 %1041, 0, !dbg !31
  br i1 %1042, label %call_end363, label %common.ret, !dbg !31, !prof !33

handle_init_end361:                               ; preds = %call_end363, %if_then358
  %1043 = phi ptr [ %1037, %if_then358 ], [ %1046, %call_end363 ], !dbg !31
  %1044 = call i32 %1036(ptr %1043, ptr nonnull %stack_ffi_any1228, i32 5, ptr nonnull %1035), !dbg !31
  %1045 = icmp eq i32 %1044, 0, !dbg !31
  br i1 %1045, label %if_end359, label %common.ret, !dbg !31, !prof !33

call_end363:                                      ; preds = %handle_init360
  %1046 = load ptr, ptr %56, align 8, !dbg !31
  store ptr %1046, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  br label %handle_init_end361, !dbg !31

if_end368:                                        ; preds = %if_end359
  %1047 = getelementptr inbounds i64, ptr %flashattn_gqa_decode_no_split.V.strides, i64 3, !dbg !31
  %1048 = load i64, ptr %1047, align 8, !dbg !31, !tbaa !84
  %1049 = and i64 %1048, 4294967295, !dbg !31
  %.not1257 = icmp eq i64 %1049, 1, !dbg !31
  br i1 %.not1257, label %if_end382, label %if_end373, !dbg !31, !prof !37

if_end373:                                        ; preds = %if_end368
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %1050 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %1050, align 8, !dbg !31
  %1051 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %1051, align 8, !dbg !31
  %1052 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.19, ptr %1052, align 8, !dbg !31
  %1053 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %1053, align 8, !dbg !31
  %1054 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store ptr @.str.37, ptr %1054, align 8, !dbg !31
  %1055 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1055, align 8, !dbg !31
  %1056 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 1, ptr %1056, align 8, !dbg !31
  %1057 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1057, align 8, !dbg !31
  %1058 = load i64, ptr %1047, align 8, !dbg !31, !tbaa !84
  %sext1384 = shl i64 %1058, 32, !dbg !31
  %1059 = ashr exact i64 %sext1384, 32, !dbg !31
  %1060 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 2, !dbg !31
  store i64 %1059, ptr %1060, align 8, !dbg !31
  %1061 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 5, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %1061, i8 0, i64 16, i1 false), !dbg !31
  %1062 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %1063 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  %.not1385 = icmp eq ptr %1063, null, !dbg !31
  br i1 %.not1385, label %handle_init374, label %handle_init_end375, !dbg !31, !prof !37

handle_init374:                                   ; preds = %if_end373
  %1064 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %1065 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %1066 = call i32 %1065(ptr %1064, ptr nonnull @.str.26, ptr nonnull %55), !dbg !31
  %1067 = icmp eq i32 %1066, 0, !dbg !31
  br i1 %1067, label %call_end377, label %common.ret, !dbg !31, !prof !33

handle_init_end375:                               ; preds = %call_end377, %if_end373
  %1068 = phi ptr [ %1063, %if_end373 ], [ %1071, %call_end377 ], !dbg !31
  %1069 = call i32 %1062(ptr %1068, ptr nonnull %stack_ffi_any1228, i32 5, ptr nonnull %1061), !dbg !31
  %1070 = icmp eq i32 %1069, 0, !dbg !31
  br i1 %1070, label %if_end382, label %common.ret, !dbg !31, !prof !33

call_end377:                                      ; preds = %handle_init374
  %1071 = load ptr, ptr %55, align 8, !dbg !31
  store ptr %1071, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  br label %handle_init_end375, !dbg !31

if_end382:                                        ; preds = %if_end368, %handle_init_end375
  %1072 = getelementptr inbounds i64, ptr %flashattn_gqa_decode_no_split.V.strides, i64 2, !dbg !31
  %1073 = load i64, ptr %1072, align 8, !dbg !31, !tbaa !84
  %1074 = and i64 %1073, 4294967295, !dbg !31
  %.not1258 = icmp eq i64 %1074, 128, !dbg !31
  br i1 %.not1258, label %if_end396, label %if_then383, !dbg !31, !prof !37

if_then383:                                       ; preds = %if_end359, %if_end382
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %1075 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %1075, align 8, !dbg !31
  %1076 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %1076, align 8, !dbg !31
  %1077 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.19, ptr %1077, align 8, !dbg !31
  %1078 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %1078, align 8, !dbg !31
  %1079 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store ptr @.str.29, ptr %1079, align 8, !dbg !31
  %1080 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1080, align 8, !dbg !31
  %1081 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 128, ptr %1081, align 8, !dbg !31
  %1082 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1082, align 8, !dbg !31
  br i1 %1038, label %if_end387, label %if_else386, !dbg !31

if_end384:                                        ; preds = %handle_init_end389
  br i1 %1038, label %if_then397, label %if_end396, !dbg !31

if_else386:                                       ; preds = %if_then383
  %1083 = getelementptr inbounds i64, ptr %flashattn_gqa_decode_no_split.V.strides, i64 2, !dbg !31
  %1084 = load i64, ptr %1083, align 8, !dbg !31, !tbaa !84
  br label %if_end387, !dbg !31

if_end387:                                        ; preds = %if_then383, %if_else386
  %1085 = phi i64 [ %1084, %if_else386 ], [ 1, %if_then383 ], !dbg !31
  %sext1382 = shl i64 %1085, 32, !dbg !31
  %1086 = ashr exact i64 %sext1382, 32, !dbg !31
  %1087 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 2, !dbg !31
  store i64 %1086, ptr %1087, align 8, !dbg !31
  %1088 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 5, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %1088, i8 0, i64 16, i1 false), !dbg !31
  %1089 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %1090 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  %.not1383 = icmp eq ptr %1090, null, !dbg !31
  br i1 %.not1383, label %handle_init388, label %handle_init_end389, !dbg !31, !prof !37

handle_init388:                                   ; preds = %if_end387
  %1091 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %1092 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %1093 = call i32 %1092(ptr %1091, ptr nonnull @.str.26, ptr nonnull %54), !dbg !31
  %1094 = icmp eq i32 %1093, 0, !dbg !31
  br i1 %1094, label %call_end391, label %common.ret, !dbg !31, !prof !33

handle_init_end389:                               ; preds = %call_end391, %if_end387
  %1095 = phi ptr [ %1090, %if_end387 ], [ %1098, %call_end391 ], !dbg !31
  %1096 = call i32 %1089(ptr %1095, ptr nonnull %stack_ffi_any1228, i32 5, ptr nonnull %1088), !dbg !31
  %1097 = icmp eq i32 %1096, 0, !dbg !31
  br i1 %1097, label %if_end384, label %common.ret, !dbg !31, !prof !33

call_end391:                                      ; preds = %handle_init388
  %1098 = load ptr, ptr %54, align 8, !dbg !31
  store ptr %1098, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  br label %handle_init_end389, !dbg !31

if_end396:                                        ; preds = %if_end382, %if_end384
  %1099 = getelementptr inbounds i64, ptr %flashattn_gqa_decode_no_split.V.strides, i64 1, !dbg !31
  %1100 = load i64, ptr %1099, align 8, !dbg !31, !tbaa !84
  %1101 = and i64 %1100, 4294967295, !dbg !31
  %.not1259 = icmp eq i64 %1101, 1024, !dbg !31
  br i1 %.not1259, label %if_end410, label %if_then397, !dbg !31, !prof !37

if_then397:                                       ; preds = %if_end384, %if_end396
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %1102 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %1102, align 8, !dbg !31
  %1103 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %1103, align 8, !dbg !31
  %1104 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.19, ptr %1104, align 8, !dbg !31
  %1105 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %1105, align 8, !dbg !31
  %1106 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store ptr @.str.30, ptr %1106, align 8, !dbg !31
  %1107 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1107, align 8, !dbg !31
  %1108 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 1024, ptr %1108, align 8, !dbg !31
  %1109 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1109, align 8, !dbg !31
  br i1 %1038, label %if_end401, label %if_else400, !dbg !31

if_end398:                                        ; preds = %handle_init_end403
  br i1 %1038, label %if_then411, label %if_end410, !dbg !31

if_else400:                                       ; preds = %if_then397
  %1110 = getelementptr inbounds i64, ptr %flashattn_gqa_decode_no_split.V.strides, i64 1, !dbg !31
  %1111 = load i64, ptr %1110, align 8, !dbg !31, !tbaa !84
  br label %if_end401, !dbg !31

if_end401:                                        ; preds = %if_then397, %if_else400
  %1112 = phi i64 [ %1111, %if_else400 ], [ 1, %if_then397 ], !dbg !31
  %sext1380 = shl i64 %1112, 32, !dbg !31
  %1113 = ashr exact i64 %sext1380, 32, !dbg !31
  %1114 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 2, !dbg !31
  store i64 %1113, ptr %1114, align 8, !dbg !31
  %1115 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 5, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %1115, i8 0, i64 16, i1 false), !dbg !31
  %1116 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %1117 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  %.not1381 = icmp eq ptr %1117, null, !dbg !31
  br i1 %.not1381, label %handle_init402, label %handle_init_end403, !dbg !31, !prof !37

handle_init402:                                   ; preds = %if_end401
  %1118 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %1119 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %1120 = call i32 %1119(ptr %1118, ptr nonnull @.str.26, ptr nonnull %53), !dbg !31
  %1121 = icmp eq i32 %1120, 0, !dbg !31
  br i1 %1121, label %call_end405, label %common.ret, !dbg !31, !prof !33

handle_init_end403:                               ; preds = %call_end405, %if_end401
  %1122 = phi ptr [ %1117, %if_end401 ], [ %1125, %call_end405 ], !dbg !31
  %1123 = call i32 %1116(ptr %1122, ptr nonnull %stack_ffi_any1228, i32 5, ptr nonnull %1115), !dbg !31
  %1124 = icmp eq i32 %1123, 0, !dbg !31
  br i1 %1124, label %if_end398, label %common.ret, !dbg !31, !prof !33

call_end405:                                      ; preds = %handle_init402
  %1125 = load ptr, ptr %53, align 8, !dbg !31
  store ptr %1125, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  br label %handle_init_end403, !dbg !31

if_end410:                                        ; preds = %if_end396, %if_end398
  %1126 = load i64, ptr %flashattn_gqa_decode_no_split.V.strides, align 8, !dbg !31, !tbaa !84
  %1127 = and i64 %1126, 4294967295, !dbg !31
  %.not1260 = icmp eq i64 %1127, 8388608, !dbg !31
  br i1 %.not1260, label %if_end412, label %if_then411, !dbg !31, !prof !37

if_then411:                                       ; preds = %if_end398, %if_end410
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %1128 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %1128, align 8, !dbg !31
  %1129 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %1129, align 8, !dbg !31
  %1130 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.19, ptr %1130, align 8, !dbg !31
  %1131 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %1131, align 8, !dbg !31
  %1132 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store ptr @.str.31, ptr %1132, align 8, !dbg !31
  %1133 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1133, align 8, !dbg !31
  %1134 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 8388608, ptr %1134, align 8, !dbg !31
  %1135 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1135, align 8, !dbg !31
  br i1 %1038, label %if_end415, label %if_else414, !dbg !31

if_end412:                                        ; preds = %handle_init_end417, %if_end410
  %1136 = getelementptr inbounds %1, ptr %V_handle, i64 0, i32 6, !dbg !31
  %1137 = load i64, ptr %1136, align 8, !dbg !31
  %.not1261 = icmp eq i64 %1137, 0, !dbg !31
  br i1 %.not1261, label %if_end423, label %if_then422, !dbg !31, !prof !37

if_else414:                                       ; preds = %if_then411
  %1138 = load i64, ptr %flashattn_gqa_decode_no_split.V.strides, align 8, !dbg !31, !tbaa !84
  br label %if_end415, !dbg !31

if_end415:                                        ; preds = %if_then411, %if_else414
  %1139 = phi i64 [ %1138, %if_else414 ], [ 1, %if_then411 ], !dbg !31
  %sext1378 = shl i64 %1139, 32, !dbg !31
  %1140 = ashr exact i64 %sext1378, 32, !dbg !31
  %1141 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 2, !dbg !31
  store i64 %1140, ptr %1141, align 8, !dbg !31
  %1142 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 5, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %1142, i8 0, i64 16, i1 false), !dbg !31
  %1143 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %1144 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  %.not1379 = icmp eq ptr %1144, null, !dbg !31
  br i1 %.not1379, label %handle_init416, label %handle_init_end417, !dbg !31, !prof !37

handle_init416:                                   ; preds = %if_end415
  %1145 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %1146 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %1147 = call i32 %1146(ptr %1145, ptr nonnull @.str.26, ptr nonnull %52), !dbg !31
  %1148 = icmp eq i32 %1147, 0, !dbg !31
  br i1 %1148, label %call_end419, label %common.ret, !dbg !31, !prof !33

handle_init_end417:                               ; preds = %call_end419, %if_end415
  %1149 = phi ptr [ %1144, %if_end415 ], [ %1152, %call_end419 ], !dbg !31
  %1150 = call i32 %1143(ptr %1149, ptr nonnull %stack_ffi_any1228, i32 5, ptr nonnull %1142), !dbg !31
  %1151 = icmp eq i32 %1150, 0, !dbg !31
  br i1 %1151, label %if_end412, label %common.ret, !dbg !31, !prof !33

call_end419:                                      ; preds = %handle_init416
  %1152 = load ptr, ptr %52, align 8, !dbg !31
  store ptr %1152, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  br label %handle_init_end417, !dbg !31

if_then422:                                       ; preds = %if_end412
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %1153 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %1153, align 8, !dbg !31
  %1154 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %1154, align 8, !dbg !31
  %1155 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.19, ptr %1155, align 8, !dbg !31
  %1156 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1156, align 8, !dbg !31
  %1157 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store i64 0, ptr %1157, align 8, !dbg !31
  %1158 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1158, align 8, !dbg !31
  %1159 = load i64, ptr %1136, align 8, !dbg !31
  %1160 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 %1159, ptr %1160, align 8, !dbg !31
  %1161 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %1161, i8 0, i64 16, i1 false), !dbg !31
  %1162 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %1163 = load ptr, ptr @.tvm_func.__tvm_error_byte_offset_mismatch, align 8, !dbg !31
  %.not1377 = icmp eq ptr %1163, null, !dbg !31
  br i1 %.not1377, label %handle_init424, label %handle_init_end425, !dbg !31, !prof !37

if_end423:                                        ; preds = %handle_init_end425, %if_end412
  %1164 = getelementptr inbounds %1, ptr %V_handle, i64 0, i32 1, i32 1, !dbg !31
  %1165 = load i32, ptr %1164, align 4, !dbg !31
  %1166 = load i32, ptr %164, align 4, !dbg !31
  %.not1262 = icmp eq i32 %1165, %1166, !dbg !31
  br i1 %.not1262, label %if_end431, label %if_then430, !dbg !31, !prof !37

handle_init424:                                   ; preds = %if_then422
  %1167 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %1168 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %1169 = call i32 %1168(ptr %1167, ptr nonnull @.str.32, ptr nonnull %51), !dbg !31
  %1170 = icmp eq i32 %1169, 0, !dbg !31
  br i1 %1170, label %call_end427, label %common.ret, !dbg !31, !prof !33

handle_init_end425:                               ; preds = %call_end427, %if_then422
  %1171 = phi ptr [ %1163, %if_then422 ], [ %1174, %call_end427 ], !dbg !31
  %1172 = call i32 %1162(ptr %1171, ptr nonnull %stack_ffi_any1228, i32 4, ptr nonnull %1161), !dbg !31
  %1173 = icmp eq i32 %1172, 0, !dbg !31
  br i1 %1173, label %if_end423, label %common.ret, !dbg !31, !prof !33

call_end427:                                      ; preds = %handle_init424
  %1174 = load ptr, ptr %51, align 8, !dbg !31
  store ptr %1174, ptr @.tvm_func.__tvm_error_byte_offset_mismatch, align 8, !dbg !31
  br label %handle_init_end425, !dbg !31

if_then430:                                       ; preds = %if_end423
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %1175 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %1175, align 8, !dbg !31
  %1176 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %1176, align 8, !dbg !31
  %1177 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.19, ptr %1177, align 8, !dbg !31
  %1178 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %1178, align 8, !dbg !31
  %1179 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store ptr @.str.38, ptr %1179, align 8, !dbg !31
  %1180 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1180, align 8, !dbg !31
  %1181 = load i32, ptr %164, align 4, !dbg !31
  %1182 = sext i32 %1181 to i64, !dbg !31
  %1183 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 %1182, ptr %1183, align 8, !dbg !31
  %1184 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1184, align 8, !dbg !31
  %1185 = load i32, ptr %1164, align 4, !dbg !31
  %1186 = sext i32 %1185 to i64, !dbg !31
  %1187 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 2, !dbg !31
  store i64 %1186, ptr %1187, align 8, !dbg !31
  %1188 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 5, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %1188, i8 0, i64 16, i1 false), !dbg !31
  %1189 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %1190 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  %.not1376 = icmp eq ptr %1190, null, !dbg !31
  br i1 %.not1376, label %handle_init432, label %handle_init_end433, !dbg !31, !prof !37

if_end431:                                        ; preds = %handle_init_end433, %if_end423
  %1191 = getelementptr inbounds %1, ptr %V_handle, i64 0, i32 1, i32 0, !dbg !31
  %1192 = load i32, ptr %1191, align 4, !dbg !31
  %.not1263 = icmp eq i32 %1192, 2, !dbg !31
  br i1 %.not1263, label %if_end439, label %if_then438, !dbg !31, !prof !37

handle_init432:                                   ; preds = %if_then430
  %1193 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %1194 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %1195 = call i32 %1194(ptr %1193, ptr nonnull @.str.26, ptr nonnull %50), !dbg !31
  %1196 = icmp eq i32 %1195, 0, !dbg !31
  br i1 %1196, label %call_end435, label %common.ret, !dbg !31, !prof !33

handle_init_end433:                               ; preds = %call_end435, %if_then430
  %1197 = phi ptr [ %1190, %if_then430 ], [ %1200, %call_end435 ], !dbg !31
  %1198 = call i32 %1189(ptr %1197, ptr nonnull %stack_ffi_any1228, i32 5, ptr nonnull %1188), !dbg !31
  %1199 = icmp eq i32 %1198, 0, !dbg !31
  br i1 %1199, label %if_end431, label %common.ret, !dbg !31, !prof !33

call_end435:                                      ; preds = %handle_init432
  %1200 = load ptr, ptr %50, align 8, !dbg !31
  store ptr %1200, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  br label %handle_init_end433, !dbg !31

if_then438:                                       ; preds = %if_end431
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %1201 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %1201, align 8, !dbg !31
  %1202 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %1202, align 8, !dbg !31
  %1203 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.19, ptr %1203, align 8, !dbg !31
  %1204 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1204, align 8, !dbg !31
  %1205 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store i64 2, ptr %1205, align 8, !dbg !31
  %1206 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1206, align 8, !dbg !31
  %1207 = load i32, ptr %1191, align 4, !dbg !31
  %1208 = sext i32 %1207 to i64, !dbg !31
  %1209 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 %1208, ptr %1209, align 8, !dbg !31
  %1210 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %1210, i8 0, i64 16, i1 false), !dbg !31
  %1211 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %1212 = load ptr, ptr @.tvm_func.__tvm_error_device_type_mismatch, align 8, !dbg !31
  %.not1375 = icmp eq ptr %1212, null, !dbg !31
  br i1 %.not1375, label %handle_init440, label %handle_init_end441, !dbg !31, !prof !37

if_end439:                                        ; preds = %handle_init_end441, %if_end431
  %1213 = icmp eq ptr %V, null, !dbg !31
  br i1 %1213, label %if_then446, label %if_end447, !dbg !31, !prof !33

handle_init440:                                   ; preds = %if_then438
  %1214 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %1215 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %1216 = call i32 %1215(ptr %1214, ptr nonnull @.str.33, ptr nonnull %49), !dbg !31
  %1217 = icmp eq i32 %1216, 0, !dbg !31
  br i1 %1217, label %call_end443, label %common.ret, !dbg !31, !prof !33

handle_init_end441:                               ; preds = %call_end443, %if_then438
  %1218 = phi ptr [ %1212, %if_then438 ], [ %1221, %call_end443 ], !dbg !31
  %1219 = call i32 %1211(ptr %1218, ptr nonnull %stack_ffi_any1228, i32 4, ptr nonnull %1210), !dbg !31
  %1220 = icmp eq i32 %1219, 0, !dbg !31
  br i1 %1220, label %if_end439, label %common.ret, !dbg !31, !prof !33

call_end443:                                      ; preds = %handle_init440
  %1221 = load ptr, ptr %49, align 8, !dbg !31
  store ptr %1221, ptr @.tvm_func.__tvm_error_device_type_mismatch, align 8, !dbg !31
  br label %handle_init_end441, !dbg !31

if_then446:                                       ; preds = %if_end439
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %1222 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %1222, align 8, !dbg !31
  %1223 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %1223, align 8, !dbg !31
  %1224 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.19, ptr %1224, align 8, !dbg !31
  %1225 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %1225, align 8, !dbg !31
  %1226 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store ptr @.str.34, ptr %1226, align 8, !dbg !31
  %1227 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %1227, i8 0, i64 16, i1 false), !dbg !31
  %1228 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %1229 = load ptr, ptr @.tvm_func.__tvm_error_null_ptr, align 8, !dbg !31
  %.not1374 = icmp eq ptr %1229, null, !dbg !31
  br i1 %.not1374, label %handle_init448, label %handle_init_end449, !dbg !31, !prof !37

if_end447:                                        ; preds = %handle_init_end449, %if_end439
  %1230 = getelementptr inbounds %1, ptr %mask_handle, i64 0, i32 3, i32 2, !dbg !31
  %1231 = load i16, ptr %1230, align 2, !dbg !31
  %1232 = icmp ne i16 %1231, 1, !dbg !31
  %1233 = getelementptr inbounds %1, ptr %mask_handle, i64 0, i32 3, i32 1, !dbg !31
  %1234 = load i8, ptr %1233, align 1, !dbg !31
  %1235 = icmp ne i8 %1234, 8, !dbg !31
  %1236 = getelementptr inbounds %1, ptr %mask_handle, i64 0, i32 3, i32 0, !dbg !31
  %1237 = load i8, ptr %1236, align 1, !dbg !31
  %1238 = icmp ne i8 %1237, 1, !dbg !31
  %1239 = or i1 %1235, %1238, !dbg !31
  %1240 = or i1 %1232, %1239, !dbg !31
  br i1 %1240, label %if_then454, label %if_end455, !dbg !31, !prof !33

handle_init448:                                   ; preds = %if_then446
  %1241 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %1242 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %1243 = call i32 %1242(ptr %1241, ptr nonnull @.str.35, ptr nonnull %48), !dbg !31
  %1244 = icmp eq i32 %1243, 0, !dbg !31
  br i1 %1244, label %call_end451, label %common.ret, !dbg !31, !prof !33

handle_init_end449:                               ; preds = %call_end451, %if_then446
  %1245 = phi ptr [ %1229, %if_then446 ], [ %1248, %call_end451 ], !dbg !31
  %1246 = call i32 %1228(ptr %1245, ptr nonnull %stack_ffi_any1228, i32 3, ptr nonnull %1227), !dbg !31
  %1247 = icmp eq i32 %1246, 0, !dbg !31
  br i1 %1247, label %if_end447, label %common.ret, !dbg !31, !prof !33

call_end451:                                      ; preds = %handle_init448
  %1248 = load ptr, ptr %48, align 8, !dbg !31
  store ptr %1248, ptr @.tvm_func.__tvm_error_null_ptr, align 8, !dbg !31
  br label %handle_init_end449, !dbg !31

if_then454:                                       ; preds = %if_end447
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %1249 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %1249, align 8, !dbg !31
  %1250 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %1250, align 8, !dbg !31
  %1251 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.20, ptr %1251, align 8, !dbg !31
  %1252 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1252, align 8, !dbg !31
  %1253 = load i8, ptr %1236, align 1, !dbg !31
  %1254 = zext i8 %1253 to i64, !dbg !31
  %1255 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store i64 %1254, ptr %1255, align 8, !dbg !31
  %1256 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1256, align 8, !dbg !31
  %1257 = load i8, ptr %1233, align 1, !dbg !31
  %1258 = zext i8 %1257 to i64, !dbg !31
  %1259 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 %1258, ptr %1259, align 8, !dbg !31
  %1260 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1260, align 8, !dbg !31
  %1261 = load i16, ptr %1230, align 2, !dbg !31
  %1262 = zext i16 %1261 to i64, !dbg !31
  %1263 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 2, !dbg !31
  store i64 %1262, ptr %1263, align 8, !dbg !31
  %1264 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 5, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1264, align 8, !dbg !31
  %1265 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 5, i32 2, !dbg !31
  store i64 1, ptr %1265, align 8, !dbg !31
  %1266 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 6, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1266, align 8, !dbg !31
  %1267 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 6, i32 2, !dbg !31
  store i64 8, ptr %1267, align 8, !dbg !31
  %1268 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 7, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1268, align 8, !dbg !31
  %1269 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 7, i32 2, !dbg !31
  store i64 1, ptr %1269, align 8, !dbg !31
  %1270 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 8, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %1270, i8 0, i64 16, i1 false), !dbg !31
  %1271 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %1272 = load ptr, ptr @.tvm_func.__tvm_error_dtype_mismatch, align 8, !dbg !31
  %.not1373 = icmp eq ptr %1272, null, !dbg !31
  br i1 %.not1373, label %handle_init456, label %handle_init_end457, !dbg !31, !prof !37

if_end455:                                        ; preds = %handle_init_end457, %if_end447
  %1273 = load i64, ptr %flashattn_gqa_decode_no_split.mask.shape, align 8, !dbg !31, !tbaa !84
  %1274 = and i64 %1273, 4294967295, !dbg !31
  %.not1264 = icmp eq i64 %1274, 1, !dbg !31
  br i1 %.not1264, label %if_end463, label %if_then462, !dbg !31, !prof !37

handle_init456:                                   ; preds = %if_then454
  %1275 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %1276 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %1277 = call i32 %1276(ptr %1275, ptr nonnull @.str.24, ptr nonnull %47), !dbg !31
  %1278 = icmp eq i32 %1277, 0, !dbg !31
  br i1 %1278, label %call_end459, label %common.ret, !dbg !31, !prof !33

handle_init_end457:                               ; preds = %call_end459, %if_then454
  %1279 = phi ptr [ %1272, %if_then454 ], [ %1282, %call_end459 ], !dbg !31
  %1280 = call i32 %1271(ptr %1279, ptr nonnull %stack_ffi_any1228, i32 8, ptr nonnull %1270), !dbg !31
  %1281 = icmp eq i32 %1280, 0, !dbg !31
  br i1 %1281, label %if_end455, label %common.ret, !dbg !31, !prof !33

call_end459:                                      ; preds = %handle_init456
  %1282 = load ptr, ptr %47, align 8, !dbg !31
  store ptr %1282, ptr @.tvm_func.__tvm_error_dtype_mismatch, align 8, !dbg !31
  br label %handle_init_end457, !dbg !31

if_then462:                                       ; preds = %if_end455
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %1283 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %1283, align 8, !dbg !31
  %1284 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %1284, align 8, !dbg !31
  %1285 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.20, ptr %1285, align 8, !dbg !31
  %1286 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %1286, align 8, !dbg !31
  %1287 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store ptr @.str.25, ptr %1287, align 8, !dbg !31
  %1288 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1288, align 8, !dbg !31
  %1289 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 1, ptr %1289, align 8, !dbg !31
  %1290 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1290, align 8, !dbg !31
  %1291 = load i64, ptr %flashattn_gqa_decode_no_split.mask.shape, align 8, !dbg !31, !tbaa !84
  %sext1371 = shl i64 %1291, 32, !dbg !31
  %1292 = ashr exact i64 %sext1371, 32, !dbg !31
  %1293 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 2, !dbg !31
  store i64 %1292, ptr %1293, align 8, !dbg !31
  %1294 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 5, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %1294, i8 0, i64 16, i1 false), !dbg !31
  %1295 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %1296 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  %.not1372 = icmp eq ptr %1296, null, !dbg !31
  br i1 %.not1372, label %handle_init464, label %handle_init_end465, !dbg !31, !prof !37

if_end463:                                        ; preds = %handle_init_end465, %if_end455
  %1297 = getelementptr inbounds i64, ptr %flashattn_gqa_decode_no_split.mask.shape, i64 1, !dbg !31
  %1298 = load i64, ptr %1297, align 8, !dbg !31, !tbaa !84
  %1299 = and i64 %1298, 4294967295, !dbg !31
  %.not1265 = icmp eq i64 %1299, 8192, !dbg !31
  br i1 %.not1265, label %if_end471, label %if_then470, !dbg !31, !prof !37

handle_init464:                                   ; preds = %if_then462
  %1300 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %1301 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %1302 = call i32 %1301(ptr %1300, ptr nonnull @.str.26, ptr nonnull %46), !dbg !31
  %1303 = icmp eq i32 %1302, 0, !dbg !31
  br i1 %1303, label %call_end467, label %common.ret, !dbg !31, !prof !33

handle_init_end465:                               ; preds = %call_end467, %if_then462
  %1304 = phi ptr [ %1296, %if_then462 ], [ %1307, %call_end467 ], !dbg !31
  %1305 = call i32 %1295(ptr %1304, ptr nonnull %stack_ffi_any1228, i32 5, ptr nonnull %1294), !dbg !31
  %1306 = icmp eq i32 %1305, 0, !dbg !31
  br i1 %1306, label %if_end463, label %common.ret, !dbg !31, !prof !33

call_end467:                                      ; preds = %handle_init464
  %1307 = load ptr, ptr %46, align 8, !dbg !31
  store ptr %1307, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  br label %handle_init_end465, !dbg !31

if_then470:                                       ; preds = %if_end463
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %1308 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %1308, align 8, !dbg !31
  %1309 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %1309, align 8, !dbg !31
  %1310 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.20, ptr %1310, align 8, !dbg !31
  %1311 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %1311, align 8, !dbg !31
  %1312 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store ptr @.str.27, ptr %1312, align 8, !dbg !31
  %1313 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1313, align 8, !dbg !31
  %1314 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 8192, ptr %1314, align 8, !dbg !31
  %1315 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1315, align 8, !dbg !31
  %1316 = load i64, ptr %1297, align 8, !dbg !31, !tbaa !84
  %sext1369 = shl i64 %1316, 32, !dbg !31
  %1317 = ashr exact i64 %sext1369, 32, !dbg !31
  %1318 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 2, !dbg !31
  store i64 %1317, ptr %1318, align 8, !dbg !31
  %1319 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 5, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %1319, i8 0, i64 16, i1 false), !dbg !31
  %1320 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %1321 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  %.not1370 = icmp eq ptr %1321, null, !dbg !31
  br i1 %.not1370, label %handle_init472, label %handle_init_end473, !dbg !31, !prof !37

if_end471:                                        ; preds = %handle_init_end473, %if_end463
  %1322 = getelementptr inbounds i64, ptr %flashattn_gqa_decode_no_split.mask.shape, i64 2, !dbg !31
  %1323 = load i64, ptr %1322, align 8, !dbg !31, !tbaa !84
  %1324 = and i64 %1323, 4294967295, !dbg !31
  %.not1266 = icmp eq i64 %1324, 8, !dbg !31
  br i1 %.not1266, label %if_end479, label %if_then478, !dbg !31, !prof !37

handle_init472:                                   ; preds = %if_then470
  %1325 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %1326 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %1327 = call i32 %1326(ptr %1325, ptr nonnull @.str.26, ptr nonnull %45), !dbg !31
  %1328 = icmp eq i32 %1327, 0, !dbg !31
  br i1 %1328, label %call_end475, label %common.ret, !dbg !31, !prof !33

handle_init_end473:                               ; preds = %call_end475, %if_then470
  %1329 = phi ptr [ %1321, %if_then470 ], [ %1332, %call_end475 ], !dbg !31
  %1330 = call i32 %1320(ptr %1329, ptr nonnull %stack_ffi_any1228, i32 5, ptr nonnull %1319), !dbg !31
  %1331 = icmp eq i32 %1330, 0, !dbg !31
  br i1 %1331, label %if_end471, label %common.ret, !dbg !31, !prof !33

call_end475:                                      ; preds = %handle_init472
  %1332 = load ptr, ptr %45, align 8, !dbg !31
  store ptr %1332, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  br label %handle_init_end473, !dbg !31

if_then478:                                       ; preds = %if_end471
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %1333 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %1333, align 8, !dbg !31
  %1334 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %1334, align 8, !dbg !31
  %1335 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.20, ptr %1335, align 8, !dbg !31
  %1336 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %1336, align 8, !dbg !31
  %1337 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store ptr @.str.28, ptr %1337, align 8, !dbg !31
  %1338 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1338, align 8, !dbg !31
  %1339 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 8, ptr %1339, align 8, !dbg !31
  %1340 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1340, align 8, !dbg !31
  %1341 = load i64, ptr %1322, align 8, !dbg !31, !tbaa !84
  %sext1367 = shl i64 %1341, 32, !dbg !31
  %1342 = ashr exact i64 %sext1367, 32, !dbg !31
  %1343 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 2, !dbg !31
  store i64 %1342, ptr %1343, align 8, !dbg !31
  %1344 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 5, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %1344, i8 0, i64 16, i1 false), !dbg !31
  %1345 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %1346 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  %.not1368 = icmp eq ptr %1346, null, !dbg !31
  br i1 %.not1368, label %handle_init480, label %handle_init_end481, !dbg !31, !prof !37

if_end479:                                        ; preds = %handle_init_end481, %if_end471
  %1347 = icmp eq ptr %flashattn_gqa_decode_no_split.mask.strides, null, !dbg !31
  br i1 %1347, label %if_then503, label %if_end488, !dbg !31

handle_init480:                                   ; preds = %if_then478
  %1348 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %1349 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %1350 = call i32 %1349(ptr %1348, ptr nonnull @.str.26, ptr nonnull %44), !dbg !31
  %1351 = icmp eq i32 %1350, 0, !dbg !31
  br i1 %1351, label %call_end483, label %common.ret, !dbg !31, !prof !33

handle_init_end481:                               ; preds = %call_end483, %if_then478
  %1352 = phi ptr [ %1346, %if_then478 ], [ %1355, %call_end483 ], !dbg !31
  %1353 = call i32 %1345(ptr %1352, ptr nonnull %stack_ffi_any1228, i32 5, ptr nonnull %1344), !dbg !31
  %1354 = icmp eq i32 %1353, 0, !dbg !31
  br i1 %1354, label %if_end479, label %common.ret, !dbg !31, !prof !33

call_end483:                                      ; preds = %handle_init480
  %1355 = load ptr, ptr %44, align 8, !dbg !31
  store ptr %1355, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  br label %handle_init_end481, !dbg !31

if_end488:                                        ; preds = %if_end479
  %1356 = getelementptr inbounds i64, ptr %flashattn_gqa_decode_no_split.mask.strides, i64 2, !dbg !31
  %1357 = load i64, ptr %1356, align 8, !dbg !31, !tbaa !84
  %1358 = and i64 %1357, 4294967295, !dbg !31
  %.not1267 = icmp eq i64 %1358, 1, !dbg !31
  br i1 %.not1267, label %if_end502, label %if_end493, !dbg !31, !prof !37

if_end493:                                        ; preds = %if_end488
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %1359 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %1359, align 8, !dbg !31
  %1360 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %1360, align 8, !dbg !31
  %1361 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.20, ptr %1361, align 8, !dbg !31
  %1362 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %1362, align 8, !dbg !31
  %1363 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store ptr @.str.29, ptr %1363, align 8, !dbg !31
  %1364 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1364, align 8, !dbg !31
  %1365 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 1, ptr %1365, align 8, !dbg !31
  %1366 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1366, align 8, !dbg !31
  %1367 = load i64, ptr %1356, align 8, !dbg !31, !tbaa !84
  %sext1365 = shl i64 %1367, 32, !dbg !31
  %1368 = ashr exact i64 %sext1365, 32, !dbg !31
  %1369 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 2, !dbg !31
  store i64 %1368, ptr %1369, align 8, !dbg !31
  %1370 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 5, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %1370, i8 0, i64 16, i1 false), !dbg !31
  %1371 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %1372 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  %.not1366 = icmp eq ptr %1372, null, !dbg !31
  br i1 %.not1366, label %handle_init494, label %handle_init_end495, !dbg !31, !prof !37

handle_init494:                                   ; preds = %if_end493
  %1373 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %1374 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %1375 = call i32 %1374(ptr %1373, ptr nonnull @.str.26, ptr nonnull %43), !dbg !31
  %1376 = icmp eq i32 %1375, 0, !dbg !31
  br i1 %1376, label %call_end497, label %common.ret, !dbg !31, !prof !33

handle_init_end495:                               ; preds = %call_end497, %if_end493
  %1377 = phi ptr [ %1372, %if_end493 ], [ %1380, %call_end497 ], !dbg !31
  %1378 = call i32 %1371(ptr %1377, ptr nonnull %stack_ffi_any1228, i32 5, ptr nonnull %1370), !dbg !31
  %1379 = icmp eq i32 %1378, 0, !dbg !31
  br i1 %1379, label %if_end502, label %common.ret, !dbg !31, !prof !33

call_end497:                                      ; preds = %handle_init494
  %1380 = load ptr, ptr %43, align 8, !dbg !31
  store ptr %1380, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  br label %handle_init_end495, !dbg !31

if_end502:                                        ; preds = %if_end488, %handle_init_end495
  %1381 = getelementptr inbounds i64, ptr %flashattn_gqa_decode_no_split.mask.strides, i64 1, !dbg !31
  %1382 = load i64, ptr %1381, align 8, !dbg !31, !tbaa !84
  %1383 = and i64 %1382, 4294967295, !dbg !31
  %.not1268 = icmp eq i64 %1383, 8, !dbg !31
  br i1 %.not1268, label %if_end516, label %if_then503, !dbg !31, !prof !37

if_then503:                                       ; preds = %if_end479, %if_end502
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %1384 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %1384, align 8, !dbg !31
  %1385 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %1385, align 8, !dbg !31
  %1386 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.20, ptr %1386, align 8, !dbg !31
  %1387 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %1387, align 8, !dbg !31
  %1388 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store ptr @.str.30, ptr %1388, align 8, !dbg !31
  %1389 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1389, align 8, !dbg !31
  %1390 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 8, ptr %1390, align 8, !dbg !31
  %1391 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1391, align 8, !dbg !31
  br i1 %1347, label %if_end507, label %if_else506, !dbg !31

if_end504:                                        ; preds = %handle_init_end509
  br i1 %1347, label %if_then517, label %if_end516, !dbg !31

if_else506:                                       ; preds = %if_then503
  %1392 = getelementptr inbounds i64, ptr %flashattn_gqa_decode_no_split.mask.strides, i64 1, !dbg !31
  %1393 = load i64, ptr %1392, align 8, !dbg !31, !tbaa !84
  br label %if_end507, !dbg !31

if_end507:                                        ; preds = %if_then503, %if_else506
  %1394 = phi i64 [ %1393, %if_else506 ], [ 1, %if_then503 ], !dbg !31
  %sext1363 = shl i64 %1394, 32, !dbg !31
  %1395 = ashr exact i64 %sext1363, 32, !dbg !31
  %1396 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 2, !dbg !31
  store i64 %1395, ptr %1396, align 8, !dbg !31
  %1397 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 5, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %1397, i8 0, i64 16, i1 false), !dbg !31
  %1398 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %1399 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  %.not1364 = icmp eq ptr %1399, null, !dbg !31
  br i1 %.not1364, label %handle_init508, label %handle_init_end509, !dbg !31, !prof !37

handle_init508:                                   ; preds = %if_end507
  %1400 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %1401 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %1402 = call i32 %1401(ptr %1400, ptr nonnull @.str.26, ptr nonnull %42), !dbg !31
  %1403 = icmp eq i32 %1402, 0, !dbg !31
  br i1 %1403, label %call_end511, label %common.ret, !dbg !31, !prof !33

handle_init_end509:                               ; preds = %call_end511, %if_end507
  %1404 = phi ptr [ %1399, %if_end507 ], [ %1407, %call_end511 ], !dbg !31
  %1405 = call i32 %1398(ptr %1404, ptr nonnull %stack_ffi_any1228, i32 5, ptr nonnull %1397), !dbg !31
  %1406 = icmp eq i32 %1405, 0, !dbg !31
  br i1 %1406, label %if_end504, label %common.ret, !dbg !31, !prof !33

call_end511:                                      ; preds = %handle_init508
  %1407 = load ptr, ptr %42, align 8, !dbg !31
  store ptr %1407, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  br label %handle_init_end509, !dbg !31

if_end516:                                        ; preds = %if_end502, %if_end504
  %1408 = load i64, ptr %flashattn_gqa_decode_no_split.mask.strides, align 8, !dbg !31, !tbaa !84
  %1409 = and i64 %1408, 4294967295, !dbg !31
  %.not1269 = icmp eq i64 %1409, 65536, !dbg !31
  br i1 %.not1269, label %if_end518, label %if_then517, !dbg !31, !prof !37

if_then517:                                       ; preds = %if_end504, %if_end516
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %1410 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %1410, align 8, !dbg !31
  %1411 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %1411, align 8, !dbg !31
  %1412 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.20, ptr %1412, align 8, !dbg !31
  %1413 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %1413, align 8, !dbg !31
  %1414 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store ptr @.str.31, ptr %1414, align 8, !dbg !31
  %1415 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1415, align 8, !dbg !31
  %1416 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 65536, ptr %1416, align 8, !dbg !31
  %1417 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1417, align 8, !dbg !31
  br i1 %1347, label %if_end521, label %if_else520, !dbg !31

if_end518:                                        ; preds = %handle_init_end523, %if_end516
  %1418 = getelementptr inbounds %1, ptr %mask_handle, i64 0, i32 6, !dbg !31
  %1419 = load i64, ptr %1418, align 8, !dbg !31
  %.not1270 = icmp eq i64 %1419, 0, !dbg !31
  br i1 %.not1270, label %if_end529, label %if_then528, !dbg !31, !prof !37

if_else520:                                       ; preds = %if_then517
  %1420 = load i64, ptr %flashattn_gqa_decode_no_split.mask.strides, align 8, !dbg !31, !tbaa !84
  br label %if_end521, !dbg !31

if_end521:                                        ; preds = %if_then517, %if_else520
  %1421 = phi i64 [ %1420, %if_else520 ], [ 1, %if_then517 ], !dbg !31
  %sext1361 = shl i64 %1421, 32, !dbg !31
  %1422 = ashr exact i64 %sext1361, 32, !dbg !31
  %1423 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 2, !dbg !31
  store i64 %1422, ptr %1423, align 8, !dbg !31
  %1424 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 5, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %1424, i8 0, i64 16, i1 false), !dbg !31
  %1425 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %1426 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  %.not1362 = icmp eq ptr %1426, null, !dbg !31
  br i1 %.not1362, label %handle_init522, label %handle_init_end523, !dbg !31, !prof !37

handle_init522:                                   ; preds = %if_end521
  %1427 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %1428 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %1429 = call i32 %1428(ptr %1427, ptr nonnull @.str.26, ptr nonnull %41), !dbg !31
  %1430 = icmp eq i32 %1429, 0, !dbg !31
  br i1 %1430, label %call_end525, label %common.ret, !dbg !31, !prof !33

handle_init_end523:                               ; preds = %call_end525, %if_end521
  %1431 = phi ptr [ %1426, %if_end521 ], [ %1434, %call_end525 ], !dbg !31
  %1432 = call i32 %1425(ptr %1431, ptr nonnull %stack_ffi_any1228, i32 5, ptr nonnull %1424), !dbg !31
  %1433 = icmp eq i32 %1432, 0, !dbg !31
  br i1 %1433, label %if_end518, label %common.ret, !dbg !31, !prof !33

call_end525:                                      ; preds = %handle_init522
  %1434 = load ptr, ptr %41, align 8, !dbg !31
  store ptr %1434, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  br label %handle_init_end523, !dbg !31

if_then528:                                       ; preds = %if_end518
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %1435 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %1435, align 8, !dbg !31
  %1436 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %1436, align 8, !dbg !31
  %1437 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.20, ptr %1437, align 8, !dbg !31
  %1438 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1438, align 8, !dbg !31
  %1439 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store i64 0, ptr %1439, align 8, !dbg !31
  %1440 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1440, align 8, !dbg !31
  %1441 = load i64, ptr %1418, align 8, !dbg !31
  %1442 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 %1441, ptr %1442, align 8, !dbg !31
  %1443 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %1443, i8 0, i64 16, i1 false), !dbg !31
  %1444 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %1445 = load ptr, ptr @.tvm_func.__tvm_error_byte_offset_mismatch, align 8, !dbg !31
  %.not1360 = icmp eq ptr %1445, null, !dbg !31
  br i1 %.not1360, label %handle_init530, label %handle_init_end531, !dbg !31, !prof !37

if_end529:                                        ; preds = %handle_init_end531, %if_end518
  %1446 = getelementptr inbounds %1, ptr %mask_handle, i64 0, i32 1, i32 1, !dbg !31
  %1447 = load i32, ptr %1446, align 4, !dbg !31
  %1448 = load i32, ptr %164, align 4, !dbg !31
  %.not1271 = icmp eq i32 %1447, %1448, !dbg !31
  br i1 %.not1271, label %if_end537, label %if_then536, !dbg !31, !prof !37

handle_init530:                                   ; preds = %if_then528
  %1449 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %1450 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %1451 = call i32 %1450(ptr %1449, ptr nonnull @.str.32, ptr nonnull %40), !dbg !31
  %1452 = icmp eq i32 %1451, 0, !dbg !31
  br i1 %1452, label %call_end533, label %common.ret, !dbg !31, !prof !33

handle_init_end531:                               ; preds = %call_end533, %if_then528
  %1453 = phi ptr [ %1445, %if_then528 ], [ %1456, %call_end533 ], !dbg !31
  %1454 = call i32 %1444(ptr %1453, ptr nonnull %stack_ffi_any1228, i32 4, ptr nonnull %1443), !dbg !31
  %1455 = icmp eq i32 %1454, 0, !dbg !31
  br i1 %1455, label %if_end529, label %common.ret, !dbg !31, !prof !33

call_end533:                                      ; preds = %handle_init530
  %1456 = load ptr, ptr %40, align 8, !dbg !31
  store ptr %1456, ptr @.tvm_func.__tvm_error_byte_offset_mismatch, align 8, !dbg !31
  br label %handle_init_end531, !dbg !31

if_then536:                                       ; preds = %if_end529
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %1457 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %1457, align 8, !dbg !31
  %1458 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %1458, align 8, !dbg !31
  %1459 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.20, ptr %1459, align 8, !dbg !31
  %1460 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %1460, align 8, !dbg !31
  %1461 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store ptr @.str.38, ptr %1461, align 8, !dbg !31
  %1462 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1462, align 8, !dbg !31
  %1463 = load i32, ptr %164, align 4, !dbg !31
  %1464 = sext i32 %1463 to i64, !dbg !31
  %1465 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 %1464, ptr %1465, align 8, !dbg !31
  %1466 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1466, align 8, !dbg !31
  %1467 = load i32, ptr %1446, align 4, !dbg !31
  %1468 = sext i32 %1467 to i64, !dbg !31
  %1469 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 2, !dbg !31
  store i64 %1468, ptr %1469, align 8, !dbg !31
  %1470 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 5, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %1470, i8 0, i64 16, i1 false), !dbg !31
  %1471 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %1472 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  %.not1359 = icmp eq ptr %1472, null, !dbg !31
  br i1 %.not1359, label %handle_init538, label %handle_init_end539, !dbg !31, !prof !37

if_end537:                                        ; preds = %handle_init_end539, %if_end529
  %1473 = getelementptr inbounds %1, ptr %mask_handle, i64 0, i32 1, i32 0, !dbg !31
  %1474 = load i32, ptr %1473, align 4, !dbg !31
  %.not1272 = icmp eq i32 %1474, 2, !dbg !31
  br i1 %.not1272, label %if_end545, label %if_then544, !dbg !31, !prof !37

handle_init538:                                   ; preds = %if_then536
  %1475 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %1476 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %1477 = call i32 %1476(ptr %1475, ptr nonnull @.str.26, ptr nonnull %39), !dbg !31
  %1478 = icmp eq i32 %1477, 0, !dbg !31
  br i1 %1478, label %call_end541, label %common.ret, !dbg !31, !prof !33

handle_init_end539:                               ; preds = %call_end541, %if_then536
  %1479 = phi ptr [ %1472, %if_then536 ], [ %1482, %call_end541 ], !dbg !31
  %1480 = call i32 %1471(ptr %1479, ptr nonnull %stack_ffi_any1228, i32 5, ptr nonnull %1470), !dbg !31
  %1481 = icmp eq i32 %1480, 0, !dbg !31
  br i1 %1481, label %if_end537, label %common.ret, !dbg !31, !prof !33

call_end541:                                      ; preds = %handle_init538
  %1482 = load ptr, ptr %39, align 8, !dbg !31
  store ptr %1482, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  br label %handle_init_end539, !dbg !31

if_then544:                                       ; preds = %if_end537
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %1483 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %1483, align 8, !dbg !31
  %1484 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %1484, align 8, !dbg !31
  %1485 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.20, ptr %1485, align 8, !dbg !31
  %1486 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1486, align 8, !dbg !31
  %1487 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store i64 2, ptr %1487, align 8, !dbg !31
  %1488 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1488, align 8, !dbg !31
  %1489 = load i32, ptr %1473, align 4, !dbg !31
  %1490 = sext i32 %1489 to i64, !dbg !31
  %1491 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 %1490, ptr %1491, align 8, !dbg !31
  %1492 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %1492, i8 0, i64 16, i1 false), !dbg !31
  %1493 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %1494 = load ptr, ptr @.tvm_func.__tvm_error_device_type_mismatch, align 8, !dbg !31
  %.not1358 = icmp eq ptr %1494, null, !dbg !31
  br i1 %.not1358, label %handle_init546, label %handle_init_end547, !dbg !31, !prof !37

if_end545:                                        ; preds = %handle_init_end547, %if_end537
  %1495 = icmp eq ptr %mask, null, !dbg !31
  br i1 %1495, label %if_then552, label %if_end553, !dbg !31, !prof !33

handle_init546:                                   ; preds = %if_then544
  %1496 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %1497 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %1498 = call i32 %1497(ptr %1496, ptr nonnull @.str.33, ptr nonnull %38), !dbg !31
  %1499 = icmp eq i32 %1498, 0, !dbg !31
  br i1 %1499, label %call_end549, label %common.ret, !dbg !31, !prof !33

handle_init_end547:                               ; preds = %call_end549, %if_then544
  %1500 = phi ptr [ %1494, %if_then544 ], [ %1503, %call_end549 ], !dbg !31
  %1501 = call i32 %1493(ptr %1500, ptr nonnull %stack_ffi_any1228, i32 4, ptr nonnull %1492), !dbg !31
  %1502 = icmp eq i32 %1501, 0, !dbg !31
  br i1 %1502, label %if_end545, label %common.ret, !dbg !31, !prof !33

call_end549:                                      ; preds = %handle_init546
  %1503 = load ptr, ptr %38, align 8, !dbg !31
  store ptr %1503, ptr @.tvm_func.__tvm_error_device_type_mismatch, align 8, !dbg !31
  br label %handle_init_end547, !dbg !31

if_then552:                                       ; preds = %if_end545
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %1504 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %1504, align 8, !dbg !31
  %1505 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %1505, align 8, !dbg !31
  %1506 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.20, ptr %1506, align 8, !dbg !31
  %1507 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %1507, align 8, !dbg !31
  %1508 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store ptr @.str.34, ptr %1508, align 8, !dbg !31
  %1509 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %1509, i8 0, i64 16, i1 false), !dbg !31
  %1510 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %1511 = load ptr, ptr @.tvm_func.__tvm_error_null_ptr, align 8, !dbg !31
  %.not1357 = icmp eq ptr %1511, null, !dbg !31
  br i1 %.not1357, label %handle_init554, label %handle_init_end555, !dbg !31, !prof !37

if_end553:                                        ; preds = %handle_init_end555, %if_end545
  br i1 %flashattn_gqa_decode_no_split.glse_is_null.not, label %if_end678, label %if_end568, !dbg !31

handle_init554:                                   ; preds = %if_then552
  %1512 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %1513 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %1514 = call i32 %1513(ptr %1512, ptr nonnull @.str.35, ptr nonnull %37), !dbg !31
  %1515 = icmp eq i32 %1514, 0, !dbg !31
  br i1 %1515, label %call_end557, label %common.ret, !dbg !31, !prof !33

handle_init_end555:                               ; preds = %call_end557, %if_then552
  %1516 = phi ptr [ %1511, %if_then552 ], [ %1519, %call_end557 ], !dbg !31
  %1517 = call i32 %1510(ptr %1516, ptr nonnull %stack_ffi_any1228, i32 3, ptr nonnull %1509), !dbg !31
  %1518 = icmp eq i32 %1517, 0, !dbg !31
  br i1 %1518, label %if_end553, label %common.ret, !dbg !31, !prof !33

call_end557:                                      ; preds = %handle_init554
  %1519 = load ptr, ptr %37, align 8, !dbg !31
  store ptr %1519, ptr @.tvm_func.__tvm_error_null_ptr, align 8, !dbg !31
  br label %handle_init_end555, !dbg !31

if_end568:                                        ; preds = %if_end553
  %1520 = getelementptr inbounds %1, ptr %glse_handle, i64 0, i32 3, i32 2, !dbg !31
  %1521 = load i16, ptr %1520, align 2, !dbg !31
  %1522 = icmp ne i16 %1521, 1, !dbg !31
  %1523 = getelementptr inbounds %1, ptr %glse_handle, i64 0, i32 3, i32 1, !dbg !31
  %1524 = load i8, ptr %1523, align 1, !dbg !31
  %1525 = icmp ne i8 %1524, 16, !dbg !31
  %1526 = getelementptr inbounds %1, ptr %glse_handle, i64 0, i32 3, i32 0, !dbg !31
  %1527 = load i8, ptr %1526, align 1, !dbg !31
  %1528 = icmp ne i8 %1527, 2, !dbg !31
  %1529 = or i1 %1525, %1528, !dbg !31
  %1530 = or i1 %1522, %1529, !dbg !31
  br i1 %1530, label %if_then569, label %if_then577, !dbg !31, !prof !33

if_then569:                                       ; preds = %if_end568
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %1531 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %1531, align 8, !dbg !31
  %1532 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %1532, align 8, !dbg !31
  %1533 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.21, ptr %1533, align 8, !dbg !31
  %1534 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1534, align 8, !dbg !31
  %1535 = load i8, ptr %1526, align 1, !dbg !31
  %1536 = zext i8 %1535 to i64, !dbg !31
  %1537 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store i64 %1536, ptr %1537, align 8, !dbg !31
  %1538 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1538, align 8, !dbg !31
  %1539 = load i8, ptr %1523, align 1, !dbg !31
  %1540 = zext i8 %1539 to i64, !dbg !31
  %1541 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 %1540, ptr %1541, align 8, !dbg !31
  %1542 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1542, align 8, !dbg !31
  %1543 = load i16, ptr %1520, align 2, !dbg !31
  %1544 = zext i16 %1543 to i64, !dbg !31
  %1545 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 2, !dbg !31
  store i64 %1544, ptr %1545, align 8, !dbg !31
  %1546 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 5, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1546, align 8, !dbg !31
  %1547 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 5, i32 2, !dbg !31
  store i64 2, ptr %1547, align 8, !dbg !31
  %1548 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 6, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1548, align 8, !dbg !31
  %1549 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 6, i32 2, !dbg !31
  store i64 16, ptr %1549, align 8, !dbg !31
  %1550 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 7, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1550, align 8, !dbg !31
  %1551 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 7, i32 2, !dbg !31
  store i64 1, ptr %1551, align 8, !dbg !31
  %1552 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 8, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %1552, i8 0, i64 16, i1 false), !dbg !31
  %1553 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %1554 = load ptr, ptr @.tvm_func.__tvm_error_dtype_mismatch, align 8, !dbg !31
  %.not1356 = icmp eq ptr %1554, null, !dbg !31
  br i1 %.not1356, label %handle_init571, label %handle_init_end572, !dbg !31, !prof !37

handle_init571:                                   ; preds = %if_then569
  %1555 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %1556 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %1557 = call i32 %1556(ptr %1555, ptr nonnull @.str.24, ptr nonnull %36), !dbg !31
  %1558 = icmp eq i32 %1557, 0, !dbg !31
  br i1 %1558, label %call_end574, label %common.ret, !dbg !31, !prof !33

handle_init_end572:                               ; preds = %call_end574, %if_then569
  %1559 = phi ptr [ %1554, %if_then569 ], [ %1562, %call_end574 ], !dbg !31
  %1560 = call i32 %1553(ptr %1559, ptr nonnull %stack_ffi_any1228, i32 8, ptr nonnull %1552), !dbg !31
  %1561 = icmp eq i32 %1560, 0, !dbg !31
  br i1 %1561, label %if_then577, label %common.ret, !dbg !31, !prof !33

call_end574:                                      ; preds = %handle_init571
  %1562 = load ptr, ptr %36, align 8, !dbg !31
  store ptr %1562, ptr @.tvm_func.__tvm_error_dtype_mismatch, align 8, !dbg !31
  br label %handle_init_end572, !dbg !31

if_then577:                                       ; preds = %handle_init_end572, %if_end568
  %1563 = load i64, ptr %flashattn_gqa_decode_no_split.glse.shape, align 8, !dbg !31, !tbaa !84
  %1564 = and i64 %1563, 4294967295, !dbg !31
  %.not1338 = icmp eq i64 %1564, 1, !dbg !31
  br i1 %.not1338, label %if_end580, label %if_then579, !dbg !31, !prof !37

if_then579:                                       ; preds = %if_then577
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %1565 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %1565, align 8, !dbg !31
  %1566 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %1566, align 8, !dbg !31
  %1567 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.21, ptr %1567, align 8, !dbg !31
  %1568 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %1568, align 8, !dbg !31
  %1569 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store ptr @.str.25, ptr %1569, align 8, !dbg !31
  %1570 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1570, align 8, !dbg !31
  %1571 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 1, ptr %1571, align 8, !dbg !31
  %1572 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1572, align 8, !dbg !31
  %1573 = load i64, ptr %flashattn_gqa_decode_no_split.glse.shape, align 8, !dbg !31, !tbaa !84
  %sext1354 = shl i64 %1573, 32, !dbg !31
  %1574 = ashr exact i64 %sext1354, 32, !dbg !31
  %1575 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 2, !dbg !31
  store i64 %1574, ptr %1575, align 8, !dbg !31
  %1576 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 5, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %1576, i8 0, i64 16, i1 false), !dbg !31
  %1577 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %1578 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  %.not1355 = icmp eq ptr %1578, null, !dbg !31
  br i1 %.not1355, label %handle_init581, label %handle_init_end582, !dbg !31, !prof !37

if_end580:                                        ; preds = %handle_init_end582, %if_then577
  %1579 = getelementptr inbounds i64, ptr %flashattn_gqa_decode_no_split.glse.shape, i64 1, !dbg !31
  %1580 = load i64, ptr %1579, align 8, !dbg !31, !tbaa !84
  %1581 = and i64 %1580, 4294967295, !dbg !31
  %.not1339 = icmp eq i64 %1581, 32, !dbg !31
  br i1 %.not1339, label %if_end588, label %if_then587, !dbg !31, !prof !37

handle_init581:                                   ; preds = %if_then579
  %1582 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %1583 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %1584 = call i32 %1583(ptr %1582, ptr nonnull @.str.26, ptr nonnull %35), !dbg !31
  %1585 = icmp eq i32 %1584, 0, !dbg !31
  br i1 %1585, label %call_end584, label %common.ret, !dbg !31, !prof !33

handle_init_end582:                               ; preds = %call_end584, %if_then579
  %1586 = phi ptr [ %1578, %if_then579 ], [ %1589, %call_end584 ], !dbg !31
  %1587 = call i32 %1577(ptr %1586, ptr nonnull %stack_ffi_any1228, i32 5, ptr nonnull %1576), !dbg !31
  %1588 = icmp eq i32 %1587, 0, !dbg !31
  br i1 %1588, label %if_end580, label %common.ret, !dbg !31, !prof !33

call_end584:                                      ; preds = %handle_init581
  %1589 = load ptr, ptr %35, align 8, !dbg !31
  store ptr %1589, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  br label %handle_init_end582, !dbg !31

if_then587:                                       ; preds = %if_end580
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %1590 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %1590, align 8, !dbg !31
  %1591 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %1591, align 8, !dbg !31
  %1592 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.21, ptr %1592, align 8, !dbg !31
  %1593 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %1593, align 8, !dbg !31
  %1594 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store ptr @.str.27, ptr %1594, align 8, !dbg !31
  %1595 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1595, align 8, !dbg !31
  %1596 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 32, ptr %1596, align 8, !dbg !31
  %1597 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1597, align 8, !dbg !31
  %1598 = load i64, ptr %1579, align 8, !dbg !31, !tbaa !84
  %sext1352 = shl i64 %1598, 32, !dbg !31
  %1599 = ashr exact i64 %sext1352, 32, !dbg !31
  %1600 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 2, !dbg !31
  store i64 %1599, ptr %1600, align 8, !dbg !31
  %1601 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 5, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %1601, i8 0, i64 16, i1 false), !dbg !31
  %1602 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %1603 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  %.not1353 = icmp eq ptr %1603, null, !dbg !31
  br i1 %.not1353, label %handle_init589, label %handle_init_end590, !dbg !31, !prof !37

if_end588:                                        ; preds = %handle_init_end590, %if_end580
  %1604 = getelementptr inbounds i64, ptr %flashattn_gqa_decode_no_split.glse.shape, i64 2, !dbg !31
  %1605 = load i64, ptr %1604, align 8, !dbg !31, !tbaa !84
  %1606 = and i64 %1605, 4294967295, !dbg !31
  %.not1340 = icmp eq i64 %1606, 1, !dbg !31
  br i1 %.not1340, label %if_end596, label %if_then595, !dbg !31, !prof !37

handle_init589:                                   ; preds = %if_then587
  %1607 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %1608 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %1609 = call i32 %1608(ptr %1607, ptr nonnull @.str.26, ptr nonnull %34), !dbg !31
  %1610 = icmp eq i32 %1609, 0, !dbg !31
  br i1 %1610, label %call_end592, label %common.ret, !dbg !31, !prof !33

handle_init_end590:                               ; preds = %call_end592, %if_then587
  %1611 = phi ptr [ %1603, %if_then587 ], [ %1614, %call_end592 ], !dbg !31
  %1612 = call i32 %1602(ptr %1611, ptr nonnull %stack_ffi_any1228, i32 5, ptr nonnull %1601), !dbg !31
  %1613 = icmp eq i32 %1612, 0, !dbg !31
  br i1 %1613, label %if_end588, label %common.ret, !dbg !31, !prof !33

call_end592:                                      ; preds = %handle_init589
  %1614 = load ptr, ptr %34, align 8, !dbg !31
  store ptr %1614, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  br label %handle_init_end590, !dbg !31

if_then595:                                       ; preds = %if_end588
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %1615 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %1615, align 8, !dbg !31
  %1616 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %1616, align 8, !dbg !31
  %1617 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.21, ptr %1617, align 8, !dbg !31
  %1618 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %1618, align 8, !dbg !31
  %1619 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store ptr @.str.28, ptr %1619, align 8, !dbg !31
  %1620 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1620, align 8, !dbg !31
  %1621 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 1, ptr %1621, align 8, !dbg !31
  %1622 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1622, align 8, !dbg !31
  %1623 = load i64, ptr %1604, align 8, !dbg !31, !tbaa !84
  %sext1350 = shl i64 %1623, 32, !dbg !31
  %1624 = ashr exact i64 %sext1350, 32, !dbg !31
  %1625 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 2, !dbg !31
  store i64 %1624, ptr %1625, align 8, !dbg !31
  %1626 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 5, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %1626, i8 0, i64 16, i1 false), !dbg !31
  %1627 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %1628 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  %.not1351 = icmp eq ptr %1628, null, !dbg !31
  br i1 %.not1351, label %handle_init597, label %handle_init_end598, !dbg !31, !prof !37

if_end596:                                        ; preds = %handle_init_end598, %if_end588
  %1629 = icmp eq ptr %flashattn_gqa_decode_no_split.glse.strides1442, null, !dbg !31
  br i1 %1629, label %if_then634, label %if_end605, !dbg !31

handle_init597:                                   ; preds = %if_then595
  %1630 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %1631 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %1632 = call i32 %1631(ptr %1630, ptr nonnull @.str.26, ptr nonnull %33), !dbg !31
  %1633 = icmp eq i32 %1632, 0, !dbg !31
  br i1 %1633, label %call_end600, label %common.ret, !dbg !31, !prof !33

handle_init_end598:                               ; preds = %call_end600, %if_then595
  %1634 = phi ptr [ %1628, %if_then595 ], [ %1637, %call_end600 ], !dbg !31
  %1635 = call i32 %1627(ptr %1634, ptr nonnull %stack_ffi_any1228, i32 5, ptr nonnull %1626), !dbg !31
  %1636 = icmp eq i32 %1635, 0, !dbg !31
  br i1 %1636, label %if_end596, label %common.ret, !dbg !31, !prof !33

call_end600:                                      ; preds = %handle_init597
  %1637 = load ptr, ptr %33, align 8, !dbg !31
  store ptr %1637, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  br label %handle_init_end598, !dbg !31

if_end605:                                        ; preds = %if_end596
  %1638 = getelementptr inbounds i64, ptr %flashattn_gqa_decode_no_split.glse.strides1442, i64 2, !dbg !31
  %1639 = load i64, ptr %1638, align 8, !dbg !31, !tbaa !84
  %1640 = and i64 %1639, 4294967295, !dbg !31
  %.not1341 = icmp eq i64 %1640, 1, !dbg !31
  br i1 %.not1341, label %if_end619, label %if_end610, !dbg !31, !prof !37

if_end610:                                        ; preds = %if_end605
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %1641 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %1641, align 8, !dbg !31
  %1642 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %1642, align 8, !dbg !31
  %1643 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.21, ptr %1643, align 8, !dbg !31
  %1644 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %1644, align 8, !dbg !31
  %1645 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store ptr @.str.29, ptr %1645, align 8, !dbg !31
  %1646 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1646, align 8, !dbg !31
  %1647 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 1, ptr %1647, align 8, !dbg !31
  %1648 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1648, align 8, !dbg !31
  %1649 = load i64, ptr %1638, align 8, !dbg !31, !tbaa !84
  %sext1348 = shl i64 %1649, 32, !dbg !31
  %1650 = ashr exact i64 %sext1348, 32, !dbg !31
  %1651 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 2, !dbg !31
  store i64 %1650, ptr %1651, align 8, !dbg !31
  %1652 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 5, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %1652, i8 0, i64 16, i1 false), !dbg !31
  %1653 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %1654 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  %.not1349 = icmp eq ptr %1654, null, !dbg !31
  br i1 %.not1349, label %handle_init611, label %handle_init_end612, !dbg !31, !prof !37

handle_init611:                                   ; preds = %if_end610
  %1655 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %1656 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %1657 = call i32 %1656(ptr %1655, ptr nonnull @.str.26, ptr nonnull %32), !dbg !31
  %1658 = icmp eq i32 %1657, 0, !dbg !31
  br i1 %1658, label %call_end614, label %common.ret, !dbg !31, !prof !33

handle_init_end612:                               ; preds = %call_end614, %if_end610
  %1659 = phi ptr [ %1654, %if_end610 ], [ %1662, %call_end614 ], !dbg !31
  %1660 = call i32 %1653(ptr %1659, ptr nonnull %stack_ffi_any1228, i32 5, ptr nonnull %1652), !dbg !31
  %1661 = icmp eq i32 %1660, 0, !dbg !31
  br i1 %1661, label %if_end619, label %common.ret, !dbg !31, !prof !33

call_end614:                                      ; preds = %handle_init611
  %1662 = load ptr, ptr %32, align 8, !dbg !31
  store ptr %1662, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  br label %handle_init_end612, !dbg !31

if_end619:                                        ; preds = %if_end605, %handle_init_end612
  %1663 = getelementptr inbounds i64, ptr %flashattn_gqa_decode_no_split.glse.strides1442, i64 1, !dbg !31
  %1664 = load i64, ptr %1663, align 8, !dbg !31, !tbaa !84
  %1665 = and i64 %1664, 4294967295, !dbg !31
  %.not1342 = icmp eq i64 %1665, 1, !dbg !31
  br i1 %.not1342, label %if_end633, label %if_end624, !dbg !31, !prof !37

if_end624:                                        ; preds = %if_end619
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %1666 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %1666, align 8, !dbg !31
  %1667 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %1667, align 8, !dbg !31
  %1668 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.21, ptr %1668, align 8, !dbg !31
  %1669 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %1669, align 8, !dbg !31
  %1670 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store ptr @.str.30, ptr %1670, align 8, !dbg !31
  %1671 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1671, align 8, !dbg !31
  %1672 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 1, ptr %1672, align 8, !dbg !31
  %1673 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1673, align 8, !dbg !31
  %1674 = load i64, ptr %1663, align 8, !dbg !31, !tbaa !84
  %sext1346 = shl i64 %1674, 32, !dbg !31
  %1675 = ashr exact i64 %sext1346, 32, !dbg !31
  %1676 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 2, !dbg !31
  store i64 %1675, ptr %1676, align 8, !dbg !31
  %1677 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 5, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %1677, i8 0, i64 16, i1 false), !dbg !31
  %1678 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %1679 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  %.not1347 = icmp eq ptr %1679, null, !dbg !31
  br i1 %.not1347, label %handle_init625, label %handle_init_end626, !dbg !31, !prof !37

handle_init625:                                   ; preds = %if_end624
  %1680 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %1681 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %1682 = call i32 %1681(ptr %1680, ptr nonnull @.str.26, ptr nonnull %31), !dbg !31
  %1683 = icmp eq i32 %1682, 0, !dbg !31
  br i1 %1683, label %call_end628, label %common.ret, !dbg !31, !prof !33

handle_init_end626:                               ; preds = %call_end628, %if_end624
  %1684 = phi ptr [ %1679, %if_end624 ], [ %1687, %call_end628 ], !dbg !31
  %1685 = call i32 %1678(ptr %1684, ptr nonnull %stack_ffi_any1228, i32 5, ptr nonnull %1677), !dbg !31
  %1686 = icmp eq i32 %1685, 0, !dbg !31
  br i1 %1686, label %if_end633, label %common.ret, !dbg !31, !prof !33

call_end628:                                      ; preds = %handle_init625
  %1687 = load ptr, ptr %31, align 8, !dbg !31
  store ptr %1687, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  br label %handle_init_end626, !dbg !31

if_end633:                                        ; preds = %if_end619, %handle_init_end626
  %1688 = load i64, ptr %flashattn_gqa_decode_no_split.glse.strides1442, align 8, !dbg !31, !tbaa !84
  %1689 = and i64 %1688, 4294967295, !dbg !31
  %.not1343 = icmp eq i64 %1689, 32, !dbg !31
  br i1 %.not1343, label %if_then645, label %if_then634, !dbg !31, !prof !37

if_then634:                                       ; preds = %if_end596, %if_end633
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %1690 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %1690, align 8, !dbg !31
  %1691 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %1691, align 8, !dbg !31
  %1692 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.21, ptr %1692, align 8, !dbg !31
  %1693 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %1693, align 8, !dbg !31
  %1694 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store ptr @.str.31, ptr %1694, align 8, !dbg !31
  %1695 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1695, align 8, !dbg !31
  %1696 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 32, ptr %1696, align 8, !dbg !31
  %1697 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1697, align 8, !dbg !31
  br i1 %1629, label %if_end638, label %if_else637, !dbg !31

if_else637:                                       ; preds = %if_then634
  %1698 = load i64, ptr %flashattn_gqa_decode_no_split.glse.strides1442, align 8, !dbg !31, !tbaa !84
  br label %if_end638, !dbg !31

if_end638:                                        ; preds = %if_then634, %if_else637
  %1699 = phi i64 [ %1698, %if_else637 ], [ 1, %if_then634 ], !dbg !31
  %sext1344 = shl i64 %1699, 32, !dbg !31
  %1700 = ashr exact i64 %sext1344, 32, !dbg !31
  %1701 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 2, !dbg !31
  store i64 %1700, ptr %1701, align 8, !dbg !31
  %1702 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 5, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %1702, i8 0, i64 16, i1 false), !dbg !31
  %1703 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %1704 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  %.not1345 = icmp eq ptr %1704, null, !dbg !31
  br i1 %.not1345, label %handle_init639, label %handle_init_end640, !dbg !31, !prof !37

handle_init639:                                   ; preds = %if_end638
  %1705 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %1706 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %1707 = call i32 %1706(ptr %1705, ptr nonnull @.str.26, ptr nonnull %30), !dbg !31
  %1708 = icmp eq i32 %1707, 0, !dbg !31
  br i1 %1708, label %call_end642, label %common.ret, !dbg !31, !prof !33

handle_init_end640:                               ; preds = %call_end642, %if_end638
  %1709 = phi ptr [ %1704, %if_end638 ], [ %1712, %call_end642 ], !dbg !31
  %1710 = call i32 %1703(ptr %1709, ptr nonnull %stack_ffi_any1228, i32 5, ptr nonnull %1702), !dbg !31
  %1711 = icmp eq i32 %1710, 0, !dbg !31
  br i1 %1711, label %if_then645, label %common.ret, !dbg !31, !prof !33

call_end642:                                      ; preds = %handle_init639
  %1712 = load ptr, ptr %30, align 8, !dbg !31
  store ptr %1712, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  br label %handle_init_end640, !dbg !31

if_then645:                                       ; preds = %if_end633, %handle_init_end640
  %1713 = getelementptr inbounds %1, ptr %glse_handle, i64 0, i32 6, !dbg !31
  %1714 = load i64, ptr %1713, align 8, !dbg !31
  %.not1336 = icmp eq i64 %1714, 0, !dbg !31
  br i1 %.not1336, label %if_then656, label %if_then648, !dbg !31, !prof !37

if_then648:                                       ; preds = %if_then645
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %1715 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %1715, align 8, !dbg !31
  %1716 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %1716, align 8, !dbg !31
  %1717 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.21, ptr %1717, align 8, !dbg !31
  %1718 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1718, align 8, !dbg !31
  %1719 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store i64 0, ptr %1719, align 8, !dbg !31
  %1720 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1720, align 8, !dbg !31
  %1721 = load i64, ptr %1713, align 8, !dbg !31
  %1722 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 %1721, ptr %1722, align 8, !dbg !31
  %1723 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %1723, i8 0, i64 16, i1 false), !dbg !31
  %1724 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %1725 = load ptr, ptr @.tvm_func.__tvm_error_byte_offset_mismatch, align 8, !dbg !31
  %.not1337 = icmp eq ptr %1725, null, !dbg !31
  br i1 %.not1337, label %handle_init650, label %handle_init_end651, !dbg !31, !prof !37

handle_init650:                                   ; preds = %if_then648
  %1726 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %1727 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %1728 = call i32 %1727(ptr %1726, ptr nonnull @.str.32, ptr nonnull %29), !dbg !31
  %1729 = icmp eq i32 %1728, 0, !dbg !31
  br i1 %1729, label %call_end653, label %common.ret, !dbg !31, !prof !33

handle_init_end651:                               ; preds = %call_end653, %if_then648
  %1730 = phi ptr [ %1725, %if_then648 ], [ %1733, %call_end653 ], !dbg !31
  %1731 = call i32 %1724(ptr %1730, ptr nonnull %stack_ffi_any1228, i32 4, ptr nonnull %1723), !dbg !31
  %1732 = icmp eq i32 %1731, 0, !dbg !31
  br i1 %1732, label %if_then656, label %common.ret, !dbg !31, !prof !33

call_end653:                                      ; preds = %handle_init650
  %1733 = load ptr, ptr %29, align 8, !dbg !31
  store ptr %1733, ptr @.tvm_func.__tvm_error_byte_offset_mismatch, align 8, !dbg !31
  br label %handle_init_end651, !dbg !31

if_then656:                                       ; preds = %if_then645, %handle_init_end651
  %1734 = getelementptr inbounds %1, ptr %glse_handle, i64 0, i32 1, i32 1, !dbg !31
  %1735 = load i32, ptr %1734, align 4, !dbg !31
  %1736 = load i32, ptr %164, align 4, !dbg !31
  %.not1334 = icmp eq i32 %1735, %1736, !dbg !31
  br i1 %.not1334, label %if_then666, label %if_then658, !dbg !31, !prof !37

if_then658:                                       ; preds = %if_then656
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %1737 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %1737, align 8, !dbg !31
  %1738 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %1738, align 8, !dbg !31
  %1739 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.21, ptr %1739, align 8, !dbg !31
  %1740 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %1740, align 8, !dbg !31
  %1741 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store ptr @.str.38, ptr %1741, align 8, !dbg !31
  %1742 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1742, align 8, !dbg !31
  %1743 = load i32, ptr %164, align 4, !dbg !31
  %1744 = sext i32 %1743 to i64, !dbg !31
  %1745 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 %1744, ptr %1745, align 8, !dbg !31
  %1746 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1746, align 8, !dbg !31
  %1747 = load i32, ptr %1734, align 4, !dbg !31
  %1748 = sext i32 %1747 to i64, !dbg !31
  %1749 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 2, !dbg !31
  store i64 %1748, ptr %1749, align 8, !dbg !31
  %1750 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 5, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %1750, i8 0, i64 16, i1 false), !dbg !31
  %1751 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %1752 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  %.not1335 = icmp eq ptr %1752, null, !dbg !31
  br i1 %.not1335, label %handle_init660, label %handle_init_end661, !dbg !31, !prof !37

handle_init660:                                   ; preds = %if_then658
  %1753 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %1754 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %1755 = call i32 %1754(ptr %1753, ptr nonnull @.str.26, ptr nonnull %28), !dbg !31
  %1756 = icmp eq i32 %1755, 0, !dbg !31
  br i1 %1756, label %call_end663, label %common.ret, !dbg !31, !prof !33

handle_init_end661:                               ; preds = %call_end663, %if_then658
  %1757 = phi ptr [ %1752, %if_then658 ], [ %1760, %call_end663 ], !dbg !31
  %1758 = call i32 %1751(ptr %1757, ptr nonnull %stack_ffi_any1228, i32 5, ptr nonnull %1750), !dbg !31
  %1759 = icmp eq i32 %1758, 0, !dbg !31
  br i1 %1759, label %if_then666, label %common.ret, !dbg !31, !prof !33

call_end663:                                      ; preds = %handle_init660
  %1760 = load ptr, ptr %28, align 8, !dbg !31
  store ptr %1760, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  br label %handle_init_end661, !dbg !31

if_then666:                                       ; preds = %if_then656, %handle_init_end661
  %1761 = getelementptr inbounds %1, ptr %glse_handle, i64 0, i32 1, i32 0, !dbg !31
  %1762 = load i32, ptr %1761, align 4, !dbg !31
  %.not1332 = icmp eq i32 %1762, 2, !dbg !31
  br i1 %.not1332, label %if_end667, label %if_then669, !dbg !31, !prof !37

if_end667:                                        ; preds = %handle_init_end672, %if_then666
  %1763 = icmp eq ptr %glse, null, !dbg !31
  br i1 %1763, label %if_then680, label %if_end678, !dbg !31, !prof !86

if_then669:                                       ; preds = %if_then666
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %1764 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %1764, align 8, !dbg !31
  %1765 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %1765, align 8, !dbg !31
  %1766 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.21, ptr %1766, align 8, !dbg !31
  %1767 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1767, align 8, !dbg !31
  %1768 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store i64 2, ptr %1768, align 8, !dbg !31
  %1769 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1769, align 8, !dbg !31
  %1770 = load i32, ptr %1761, align 4, !dbg !31
  %1771 = sext i32 %1770 to i64, !dbg !31
  %1772 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 %1771, ptr %1772, align 8, !dbg !31
  %1773 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %1773, i8 0, i64 16, i1 false), !dbg !31
  %1774 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %1775 = load ptr, ptr @.tvm_func.__tvm_error_device_type_mismatch, align 8, !dbg !31
  %.not1333 = icmp eq ptr %1775, null, !dbg !31
  br i1 %.not1333, label %handle_init671, label %handle_init_end672, !dbg !31, !prof !37

handle_init671:                                   ; preds = %if_then669
  %1776 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %1777 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %1778 = call i32 %1777(ptr %1776, ptr nonnull @.str.33, ptr nonnull %27), !dbg !31
  %1779 = icmp eq i32 %1778, 0, !dbg !31
  br i1 %1779, label %call_end674, label %common.ret, !dbg !31, !prof !33

handle_init_end672:                               ; preds = %call_end674, %if_then669
  %1780 = phi ptr [ %1775, %if_then669 ], [ %1783, %call_end674 ], !dbg !31
  %1781 = call i32 %1774(ptr %1780, ptr nonnull %stack_ffi_any1228, i32 4, ptr nonnull %1773), !dbg !31
  %1782 = icmp eq i32 %1781, 0, !dbg !31
  br i1 %1782, label %if_end667, label %common.ret, !dbg !31, !prof !33

call_end674:                                      ; preds = %handle_init671
  %1783 = load ptr, ptr %27, align 8, !dbg !31
  store ptr %1783, ptr @.tvm_func.__tvm_error_device_type_mismatch, align 8, !dbg !31
  br label %handle_init_end672, !dbg !31

if_end678:                                        ; preds = %if_end553, %handle_init_end683, %if_end667
  br i1 %flashattn_gqa_decode_no_split.Output_partial_is_null.not, label %if_end828, label %if_end696, !dbg !31

if_then680:                                       ; preds = %if_end667
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %1784 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %1784, align 8, !dbg !31
  %1785 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %1785, align 8, !dbg !31
  %1786 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.21, ptr %1786, align 8, !dbg !31
  %1787 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %1787, align 8, !dbg !31
  %1788 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store ptr @.str.34, ptr %1788, align 8, !dbg !31
  %1789 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %1789, i8 0, i64 16, i1 false), !dbg !31
  %1790 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %1791 = load ptr, ptr @.tvm_func.__tvm_error_null_ptr, align 8, !dbg !31
  %.not1331 = icmp eq ptr %1791, null, !dbg !31
  br i1 %.not1331, label %handle_init682, label %handle_init_end683, !dbg !31, !prof !37

handle_init682:                                   ; preds = %if_then680
  %1792 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %1793 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %1794 = call i32 %1793(ptr %1792, ptr nonnull @.str.35, ptr nonnull %26), !dbg !31
  %1795 = icmp eq i32 %1794, 0, !dbg !31
  br i1 %1795, label %call_end685, label %common.ret, !dbg !31, !prof !33

handle_init_end683:                               ; preds = %call_end685, %if_then680
  %1796 = phi ptr [ %1791, %if_then680 ], [ %1799, %call_end685 ], !dbg !31
  %1797 = call i32 %1790(ptr %1796, ptr nonnull %stack_ffi_any1228, i32 3, ptr nonnull %1789), !dbg !31
  %1798 = icmp eq i32 %1797, 0, !dbg !31
  br i1 %1798, label %if_end678, label %common.ret, !dbg !31, !prof !33

call_end685:                                      ; preds = %handle_init682
  %1799 = load ptr, ptr %26, align 8, !dbg !31
  store ptr %1799, ptr @.tvm_func.__tvm_error_null_ptr, align 8, !dbg !31
  br label %handle_init_end683, !dbg !31

if_end696:                                        ; preds = %if_end678
  %1800 = getelementptr inbounds %1, ptr %Output_partial_handle, i64 0, i32 3, i32 2, !dbg !31
  %1801 = load i16, ptr %1800, align 2, !dbg !31
  %1802 = icmp ne i16 %1801, 1, !dbg !31
  %1803 = getelementptr inbounds %1, ptr %Output_partial_handle, i64 0, i32 3, i32 1, !dbg !31
  %1804 = load i8, ptr %1803, align 1, !dbg !31
  %1805 = icmp ne i8 %1804, 16, !dbg !31
  %1806 = getelementptr inbounds %1, ptr %Output_partial_handle, i64 0, i32 3, i32 0, !dbg !31
  %1807 = load i8, ptr %1806, align 1, !dbg !31
  %1808 = icmp ne i8 %1807, 2, !dbg !31
  %1809 = or i1 %1805, %1808, !dbg !31
  %1810 = or i1 %1802, %1809, !dbg !31
  br i1 %1810, label %if_then697, label %if_then705, !dbg !31, !prof !33

if_then697:                                       ; preds = %if_end696
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %1811 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %1811, align 8, !dbg !31
  %1812 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %1812, align 8, !dbg !31
  %1813 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.22, ptr %1813, align 8, !dbg !31
  %1814 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1814, align 8, !dbg !31
  %1815 = load i8, ptr %1806, align 1, !dbg !31
  %1816 = zext i8 %1815 to i64, !dbg !31
  %1817 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store i64 %1816, ptr %1817, align 8, !dbg !31
  %1818 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1818, align 8, !dbg !31
  %1819 = load i8, ptr %1803, align 1, !dbg !31
  %1820 = zext i8 %1819 to i64, !dbg !31
  %1821 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 %1820, ptr %1821, align 8, !dbg !31
  %1822 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1822, align 8, !dbg !31
  %1823 = load i16, ptr %1800, align 2, !dbg !31
  %1824 = zext i16 %1823 to i64, !dbg !31
  %1825 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 2, !dbg !31
  store i64 %1824, ptr %1825, align 8, !dbg !31
  %1826 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 5, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1826, align 8, !dbg !31
  %1827 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 5, i32 2, !dbg !31
  store i64 2, ptr %1827, align 8, !dbg !31
  %1828 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 6, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1828, align 8, !dbg !31
  %1829 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 6, i32 2, !dbg !31
  store i64 16, ptr %1829, align 8, !dbg !31
  %1830 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 7, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1830, align 8, !dbg !31
  %1831 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 7, i32 2, !dbg !31
  store i64 1, ptr %1831, align 8, !dbg !31
  %1832 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 8, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %1832, i8 0, i64 16, i1 false), !dbg !31
  %1833 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %1834 = load ptr, ptr @.tvm_func.__tvm_error_dtype_mismatch, align 8, !dbg !31
  %.not1330 = icmp eq ptr %1834, null, !dbg !31
  br i1 %.not1330, label %handle_init699, label %handle_init_end700, !dbg !31, !prof !37

handle_init699:                                   ; preds = %if_then697
  %1835 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %1836 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %1837 = call i32 %1836(ptr %1835, ptr nonnull @.str.24, ptr nonnull %25), !dbg !31
  %1838 = icmp eq i32 %1837, 0, !dbg !31
  br i1 %1838, label %call_end702, label %common.ret, !dbg !31, !prof !33

handle_init_end700:                               ; preds = %call_end702, %if_then697
  %1839 = phi ptr [ %1834, %if_then697 ], [ %1842, %call_end702 ], !dbg !31
  %1840 = call i32 %1833(ptr %1839, ptr nonnull %stack_ffi_any1228, i32 8, ptr nonnull %1832), !dbg !31
  %1841 = icmp eq i32 %1840, 0, !dbg !31
  br i1 %1841, label %if_then705, label %common.ret, !dbg !31, !prof !33

call_end702:                                      ; preds = %handle_init699
  %1842 = load ptr, ptr %25, align 8, !dbg !31
  store ptr %1842, ptr @.tvm_func.__tvm_error_dtype_mismatch, align 8, !dbg !31
  br label %handle_init_end700, !dbg !31

if_then705:                                       ; preds = %handle_init_end700, %if_end696
  %1843 = load i64, ptr %flashattn_gqa_decode_no_split.Output_partial.shape, align 8, !dbg !31, !tbaa !84
  %1844 = and i64 %1843, 4294967295, !dbg !31
  %.not1306 = icmp eq i64 %1844, 1, !dbg !31
  br i1 %.not1306, label %if_end708, label %if_then707, !dbg !31, !prof !37

if_then707:                                       ; preds = %if_then705
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %1845 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %1845, align 8, !dbg !31
  %1846 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %1846, align 8, !dbg !31
  %1847 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.22, ptr %1847, align 8, !dbg !31
  %1848 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %1848, align 8, !dbg !31
  %1849 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store ptr @.str.25, ptr %1849, align 8, !dbg !31
  %1850 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1850, align 8, !dbg !31
  %1851 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 1, ptr %1851, align 8, !dbg !31
  %1852 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1852, align 8, !dbg !31
  %1853 = load i64, ptr %flashattn_gqa_decode_no_split.Output_partial.shape, align 8, !dbg !31, !tbaa !84
  %sext1328 = shl i64 %1853, 32, !dbg !31
  %1854 = ashr exact i64 %sext1328, 32, !dbg !31
  %1855 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 2, !dbg !31
  store i64 %1854, ptr %1855, align 8, !dbg !31
  %1856 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 5, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %1856, i8 0, i64 16, i1 false), !dbg !31
  %1857 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %1858 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  %.not1329 = icmp eq ptr %1858, null, !dbg !31
  br i1 %.not1329, label %handle_init709, label %handle_init_end710, !dbg !31, !prof !37

if_end708:                                        ; preds = %handle_init_end710, %if_then705
  %1859 = getelementptr inbounds i64, ptr %flashattn_gqa_decode_no_split.Output_partial.shape, i64 1, !dbg !31
  %1860 = load i64, ptr %1859, align 8, !dbg !31, !tbaa !84
  %1861 = and i64 %1860, 4294967295, !dbg !31
  %.not1307 = icmp eq i64 %1861, 32, !dbg !31
  br i1 %.not1307, label %if_end716, label %if_then715, !dbg !31, !prof !37

handle_init709:                                   ; preds = %if_then707
  %1862 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %1863 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %1864 = call i32 %1863(ptr %1862, ptr nonnull @.str.26, ptr nonnull %24), !dbg !31
  %1865 = icmp eq i32 %1864, 0, !dbg !31
  br i1 %1865, label %call_end712, label %common.ret, !dbg !31, !prof !33

handle_init_end710:                               ; preds = %call_end712, %if_then707
  %1866 = phi ptr [ %1858, %if_then707 ], [ %1869, %call_end712 ], !dbg !31
  %1867 = call i32 %1857(ptr %1866, ptr nonnull %stack_ffi_any1228, i32 5, ptr nonnull %1856), !dbg !31
  %1868 = icmp eq i32 %1867, 0, !dbg !31
  br i1 %1868, label %if_end708, label %common.ret, !dbg !31, !prof !33

call_end712:                                      ; preds = %handle_init709
  %1869 = load ptr, ptr %24, align 8, !dbg !31
  store ptr %1869, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  br label %handle_init_end710, !dbg !31

if_then715:                                       ; preds = %if_end708
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %1870 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %1870, align 8, !dbg !31
  %1871 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %1871, align 8, !dbg !31
  %1872 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.22, ptr %1872, align 8, !dbg !31
  %1873 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %1873, align 8, !dbg !31
  %1874 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store ptr @.str.27, ptr %1874, align 8, !dbg !31
  %1875 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1875, align 8, !dbg !31
  %1876 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 32, ptr %1876, align 8, !dbg !31
  %1877 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1877, align 8, !dbg !31
  %1878 = load i64, ptr %1859, align 8, !dbg !31, !tbaa !84
  %sext1326 = shl i64 %1878, 32, !dbg !31
  %1879 = ashr exact i64 %sext1326, 32, !dbg !31
  %1880 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 2, !dbg !31
  store i64 %1879, ptr %1880, align 8, !dbg !31
  %1881 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 5, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %1881, i8 0, i64 16, i1 false), !dbg !31
  %1882 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %1883 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  %.not1327 = icmp eq ptr %1883, null, !dbg !31
  br i1 %.not1327, label %handle_init717, label %handle_init_end718, !dbg !31, !prof !37

if_end716:                                        ; preds = %handle_init_end718, %if_end708
  %1884 = getelementptr inbounds i64, ptr %flashattn_gqa_decode_no_split.Output_partial.shape, i64 2, !dbg !31
  %1885 = load i64, ptr %1884, align 8, !dbg !31, !tbaa !84
  %1886 = and i64 %1885, 4294967295, !dbg !31
  %.not1308 = icmp eq i64 %1886, 1, !dbg !31
  br i1 %.not1308, label %if_end724, label %if_then723, !dbg !31, !prof !37

handle_init717:                                   ; preds = %if_then715
  %1887 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %1888 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %1889 = call i32 %1888(ptr %1887, ptr nonnull @.str.26, ptr nonnull %23), !dbg !31
  %1890 = icmp eq i32 %1889, 0, !dbg !31
  br i1 %1890, label %call_end720, label %common.ret, !dbg !31, !prof !33

handle_init_end718:                               ; preds = %call_end720, %if_then715
  %1891 = phi ptr [ %1883, %if_then715 ], [ %1894, %call_end720 ], !dbg !31
  %1892 = call i32 %1882(ptr %1891, ptr nonnull %stack_ffi_any1228, i32 5, ptr nonnull %1881), !dbg !31
  %1893 = icmp eq i32 %1892, 0, !dbg !31
  br i1 %1893, label %if_end716, label %common.ret, !dbg !31, !prof !33

call_end720:                                      ; preds = %handle_init717
  %1894 = load ptr, ptr %23, align 8, !dbg !31
  store ptr %1894, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  br label %handle_init_end718, !dbg !31

if_then723:                                       ; preds = %if_end716
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %1895 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %1895, align 8, !dbg !31
  %1896 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %1896, align 8, !dbg !31
  %1897 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.22, ptr %1897, align 8, !dbg !31
  %1898 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %1898, align 8, !dbg !31
  %1899 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store ptr @.str.28, ptr %1899, align 8, !dbg !31
  %1900 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1900, align 8, !dbg !31
  %1901 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 1, ptr %1901, align 8, !dbg !31
  %1902 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1902, align 8, !dbg !31
  %1903 = load i64, ptr %1884, align 8, !dbg !31, !tbaa !84
  %sext1324 = shl i64 %1903, 32, !dbg !31
  %1904 = ashr exact i64 %sext1324, 32, !dbg !31
  %1905 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 2, !dbg !31
  store i64 %1904, ptr %1905, align 8, !dbg !31
  %1906 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 5, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %1906, i8 0, i64 16, i1 false), !dbg !31
  %1907 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %1908 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  %.not1325 = icmp eq ptr %1908, null, !dbg !31
  br i1 %.not1325, label %handle_init725, label %handle_init_end726, !dbg !31, !prof !37

if_end724:                                        ; preds = %handle_init_end726, %if_end716
  %1909 = getelementptr inbounds i64, ptr %flashattn_gqa_decode_no_split.Output_partial.shape, i64 3, !dbg !31
  %1910 = load i64, ptr %1909, align 8, !dbg !31, !tbaa !84
  %1911 = and i64 %1910, 4294967295, !dbg !31
  %.not1309 = icmp eq i64 %1911, 128, !dbg !31
  br i1 %.not1309, label %if_end732, label %if_then731, !dbg !31, !prof !37

handle_init725:                                   ; preds = %if_then723
  %1912 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %1913 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %1914 = call i32 %1913(ptr %1912, ptr nonnull @.str.26, ptr nonnull %22), !dbg !31
  %1915 = icmp eq i32 %1914, 0, !dbg !31
  br i1 %1915, label %call_end728, label %common.ret, !dbg !31, !prof !33

handle_init_end726:                               ; preds = %call_end728, %if_then723
  %1916 = phi ptr [ %1908, %if_then723 ], [ %1919, %call_end728 ], !dbg !31
  %1917 = call i32 %1907(ptr %1916, ptr nonnull %stack_ffi_any1228, i32 5, ptr nonnull %1906), !dbg !31
  %1918 = icmp eq i32 %1917, 0, !dbg !31
  br i1 %1918, label %if_end724, label %common.ret, !dbg !31, !prof !33

call_end728:                                      ; preds = %handle_init725
  %1919 = load ptr, ptr %22, align 8, !dbg !31
  store ptr %1919, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  br label %handle_init_end726, !dbg !31

if_then731:                                       ; preds = %if_end724
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %1920 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %1920, align 8, !dbg !31
  %1921 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %1921, align 8, !dbg !31
  %1922 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.22, ptr %1922, align 8, !dbg !31
  %1923 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %1923, align 8, !dbg !31
  %1924 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store ptr @.str.36, ptr %1924, align 8, !dbg !31
  %1925 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1925, align 8, !dbg !31
  %1926 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 128, ptr %1926, align 8, !dbg !31
  %1927 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1927, align 8, !dbg !31
  %1928 = load i64, ptr %1909, align 8, !dbg !31, !tbaa !84
  %sext1322 = shl i64 %1928, 32, !dbg !31
  %1929 = ashr exact i64 %sext1322, 32, !dbg !31
  %1930 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 2, !dbg !31
  store i64 %1929, ptr %1930, align 8, !dbg !31
  %1931 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 5, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %1931, i8 0, i64 16, i1 false), !dbg !31
  %1932 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %1933 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  %.not1323 = icmp eq ptr %1933, null, !dbg !31
  br i1 %.not1323, label %handle_init733, label %handle_init_end734, !dbg !31, !prof !37

if_end732:                                        ; preds = %handle_init_end734, %if_end724
  %1934 = icmp eq ptr %flashattn_gqa_decode_no_split.Output_partial.strides1447, null, !dbg !31
  br i1 %1934, label %if_then756, label %if_end741, !dbg !31

handle_init733:                                   ; preds = %if_then731
  %1935 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %1936 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %1937 = call i32 %1936(ptr %1935, ptr nonnull @.str.26, ptr nonnull %21), !dbg !31
  %1938 = icmp eq i32 %1937, 0, !dbg !31
  br i1 %1938, label %call_end736, label %common.ret, !dbg !31, !prof !33

handle_init_end734:                               ; preds = %call_end736, %if_then731
  %1939 = phi ptr [ %1933, %if_then731 ], [ %1942, %call_end736 ], !dbg !31
  %1940 = call i32 %1932(ptr %1939, ptr nonnull %stack_ffi_any1228, i32 5, ptr nonnull %1931), !dbg !31
  %1941 = icmp eq i32 %1940, 0, !dbg !31
  br i1 %1941, label %if_end732, label %common.ret, !dbg !31, !prof !33

call_end736:                                      ; preds = %handle_init733
  %1942 = load ptr, ptr %21, align 8, !dbg !31
  store ptr %1942, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  br label %handle_init_end734, !dbg !31

if_end741:                                        ; preds = %if_end732
  %1943 = getelementptr inbounds i64, ptr %flashattn_gqa_decode_no_split.Output_partial.strides1447, i64 3, !dbg !31
  %1944 = load i64, ptr %1943, align 8, !dbg !31, !tbaa !84
  %1945 = and i64 %1944, 4294967295, !dbg !31
  %.not1310 = icmp eq i64 %1945, 1, !dbg !31
  br i1 %.not1310, label %if_end755, label %if_end746, !dbg !31, !prof !37

if_end746:                                        ; preds = %if_end741
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %1946 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %1946, align 8, !dbg !31
  %1947 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %1947, align 8, !dbg !31
  %1948 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.22, ptr %1948, align 8, !dbg !31
  %1949 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %1949, align 8, !dbg !31
  %1950 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store ptr @.str.37, ptr %1950, align 8, !dbg !31
  %1951 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1951, align 8, !dbg !31
  %1952 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 1, ptr %1952, align 8, !dbg !31
  %1953 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1953, align 8, !dbg !31
  %1954 = load i64, ptr %1943, align 8, !dbg !31, !tbaa !84
  %sext1320 = shl i64 %1954, 32, !dbg !31
  %1955 = ashr exact i64 %sext1320, 32, !dbg !31
  %1956 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 2, !dbg !31
  store i64 %1955, ptr %1956, align 8, !dbg !31
  %1957 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 5, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %1957, i8 0, i64 16, i1 false), !dbg !31
  %1958 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %1959 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  %.not1321 = icmp eq ptr %1959, null, !dbg !31
  br i1 %.not1321, label %handle_init747, label %handle_init_end748, !dbg !31, !prof !37

handle_init747:                                   ; preds = %if_end746
  %1960 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %1961 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %1962 = call i32 %1961(ptr %1960, ptr nonnull @.str.26, ptr nonnull %20), !dbg !31
  %1963 = icmp eq i32 %1962, 0, !dbg !31
  br i1 %1963, label %call_end750, label %common.ret, !dbg !31, !prof !33

handle_init_end748:                               ; preds = %call_end750, %if_end746
  %1964 = phi ptr [ %1959, %if_end746 ], [ %1967, %call_end750 ], !dbg !31
  %1965 = call i32 %1958(ptr %1964, ptr nonnull %stack_ffi_any1228, i32 5, ptr nonnull %1957), !dbg !31
  %1966 = icmp eq i32 %1965, 0, !dbg !31
  br i1 %1966, label %if_end755, label %common.ret, !dbg !31, !prof !33

call_end750:                                      ; preds = %handle_init747
  %1967 = load ptr, ptr %20, align 8, !dbg !31
  store ptr %1967, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  br label %handle_init_end748, !dbg !31

if_end755:                                        ; preds = %if_end741, %handle_init_end748
  %1968 = getelementptr inbounds i64, ptr %flashattn_gqa_decode_no_split.Output_partial.strides1447, i64 2, !dbg !31
  %1969 = load i64, ptr %1968, align 8, !dbg !31, !tbaa !84
  %1970 = and i64 %1969, 4294967295, !dbg !31
  %.not1311 = icmp eq i64 %1970, 128, !dbg !31
  br i1 %.not1311, label %if_end769, label %if_then756, !dbg !31, !prof !37

if_then756:                                       ; preds = %if_end732, %if_end755
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %1971 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %1971, align 8, !dbg !31
  %1972 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %1972, align 8, !dbg !31
  %1973 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.22, ptr %1973, align 8, !dbg !31
  %1974 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %1974, align 8, !dbg !31
  %1975 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store ptr @.str.29, ptr %1975, align 8, !dbg !31
  %1976 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1976, align 8, !dbg !31
  %1977 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 128, ptr %1977, align 8, !dbg !31
  %1978 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %1978, align 8, !dbg !31
  br i1 %1934, label %if_end760, label %if_else759, !dbg !31

if_end757:                                        ; preds = %handle_init_end762
  br i1 %1934, label %if_then770, label %if_end769, !dbg !31

if_else759:                                       ; preds = %if_then756
  %1979 = getelementptr inbounds i64, ptr %flashattn_gqa_decode_no_split.Output_partial.strides1447, i64 2, !dbg !31
  %1980 = load i64, ptr %1979, align 8, !dbg !31, !tbaa !84
  br label %if_end760, !dbg !31

if_end760:                                        ; preds = %if_then756, %if_else759
  %1981 = phi i64 [ %1980, %if_else759 ], [ 1, %if_then756 ], !dbg !31
  %sext1318 = shl i64 %1981, 32, !dbg !31
  %1982 = ashr exact i64 %sext1318, 32, !dbg !31
  %1983 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 2, !dbg !31
  store i64 %1982, ptr %1983, align 8, !dbg !31
  %1984 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 5, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %1984, i8 0, i64 16, i1 false), !dbg !31
  %1985 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %1986 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  %.not1319 = icmp eq ptr %1986, null, !dbg !31
  br i1 %.not1319, label %handle_init761, label %handle_init_end762, !dbg !31, !prof !37

handle_init761:                                   ; preds = %if_end760
  %1987 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %1988 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %1989 = call i32 %1988(ptr %1987, ptr nonnull @.str.26, ptr nonnull %19), !dbg !31
  %1990 = icmp eq i32 %1989, 0, !dbg !31
  br i1 %1990, label %call_end764, label %common.ret, !dbg !31, !prof !33

handle_init_end762:                               ; preds = %call_end764, %if_end760
  %1991 = phi ptr [ %1986, %if_end760 ], [ %1994, %call_end764 ], !dbg !31
  %1992 = call i32 %1985(ptr %1991, ptr nonnull %stack_ffi_any1228, i32 5, ptr nonnull %1984), !dbg !31
  %1993 = icmp eq i32 %1992, 0, !dbg !31
  br i1 %1993, label %if_end757, label %common.ret, !dbg !31, !prof !33

call_end764:                                      ; preds = %handle_init761
  %1994 = load ptr, ptr %19, align 8, !dbg !31
  store ptr %1994, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  br label %handle_init_end762, !dbg !31

if_end769:                                        ; preds = %if_end755, %if_end757
  %1995 = getelementptr inbounds i64, ptr %flashattn_gqa_decode_no_split.Output_partial.strides1447, i64 1, !dbg !31
  %1996 = load i64, ptr %1995, align 8, !dbg !31, !tbaa !84
  %1997 = and i64 %1996, 4294967295, !dbg !31
  %.not1312 = icmp eq i64 %1997, 128, !dbg !31
  br i1 %.not1312, label %if_end783, label %if_then770, !dbg !31, !prof !37

if_then770:                                       ; preds = %if_end757, %if_end769
  %flashattn_gqa_decode_no_split.Output_partial.strides145014881491 = phi ptr [ %flashattn_gqa_decode_no_split.Output_partial.strides1447, %if_end769 ], [ null, %if_end757 ]
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %1998 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %1998, align 8, !dbg !31
  %1999 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %1999, align 8, !dbg !31
  %2000 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.22, ptr %2000, align 8, !dbg !31
  %2001 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %2001, align 8, !dbg !31
  %2002 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store ptr @.str.30, ptr %2002, align 8, !dbg !31
  %2003 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %2003, align 8, !dbg !31
  %2004 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 128, ptr %2004, align 8, !dbg !31
  %2005 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %2005, align 8, !dbg !31
  br i1 %1934, label %if_end774, label %if_else773, !dbg !31

if_end771:                                        ; preds = %handle_init_end776
  br i1 %1934, label %if_then784, label %if_end783, !dbg !31

if_else773:                                       ; preds = %if_then770
  %2006 = getelementptr inbounds i64, ptr %flashattn_gqa_decode_no_split.Output_partial.strides145014881491, i64 1, !dbg !31
  %2007 = load i64, ptr %2006, align 8, !dbg !31, !tbaa !84
  br label %if_end774, !dbg !31

if_end774:                                        ; preds = %if_then770, %if_else773
  %2008 = phi i64 [ %2007, %if_else773 ], [ 1, %if_then770 ], !dbg !31
  %sext1316 = shl i64 %2008, 32, !dbg !31
  %2009 = ashr exact i64 %sext1316, 32, !dbg !31
  %2010 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 2, !dbg !31
  store i64 %2009, ptr %2010, align 8, !dbg !31
  %2011 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 5, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %2011, i8 0, i64 16, i1 false), !dbg !31
  %2012 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %2013 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  %.not1317 = icmp eq ptr %2013, null, !dbg !31
  br i1 %.not1317, label %handle_init775, label %handle_init_end776, !dbg !31, !prof !37

handle_init775:                                   ; preds = %if_end774
  %2014 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %2015 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %2016 = call i32 %2015(ptr %2014, ptr nonnull @.str.26, ptr nonnull %18), !dbg !31
  %2017 = icmp eq i32 %2016, 0, !dbg !31
  br i1 %2017, label %call_end778, label %common.ret, !dbg !31, !prof !33

handle_init_end776:                               ; preds = %call_end778, %if_end774
  %2018 = phi ptr [ %2013, %if_end774 ], [ %2021, %call_end778 ], !dbg !31
  %2019 = call i32 %2012(ptr %2018, ptr nonnull %stack_ffi_any1228, i32 5, ptr nonnull %2011), !dbg !31
  %2020 = icmp eq i32 %2019, 0, !dbg !31
  br i1 %2020, label %if_end771, label %common.ret, !dbg !31, !prof !33

call_end778:                                      ; preds = %handle_init775
  %2021 = load ptr, ptr %18, align 8, !dbg !31
  store ptr %2021, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  br label %handle_init_end776, !dbg !31

if_end783:                                        ; preds = %if_end769, %if_end771
  %flashattn_gqa_decode_no_split.Output_partial.strides14511493 = phi ptr [ %flashattn_gqa_decode_no_split.Output_partial.strides145014881491, %if_end771 ], [ %flashattn_gqa_decode_no_split.Output_partial.strides1447, %if_end769 ]
  %2022 = load i64, ptr %flashattn_gqa_decode_no_split.Output_partial.strides14511493, align 8, !dbg !31, !tbaa !84
  %2023 = and i64 %2022, 4294967295, !dbg !31
  %.not1313 = icmp eq i64 %2023, 4096, !dbg !31
  br i1 %.not1313, label %if_then795, label %if_then784, !dbg !31, !prof !37

if_then784:                                       ; preds = %if_end771, %if_end783
  %flashattn_gqa_decode_no_split.Output_partial.strides145114941497 = phi ptr [ %flashattn_gqa_decode_no_split.Output_partial.strides14511493, %if_end783 ], [ %flashattn_gqa_decode_no_split.Output_partial.strides145014881491, %if_end771 ]
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %2024 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %2024, align 8, !dbg !31
  %2025 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %2025, align 8, !dbg !31
  %2026 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.22, ptr %2026, align 8, !dbg !31
  %2027 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %2027, align 8, !dbg !31
  %2028 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store ptr @.str.31, ptr %2028, align 8, !dbg !31
  %2029 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %2029, align 8, !dbg !31
  %2030 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 4096, ptr %2030, align 8, !dbg !31
  %2031 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %2031, align 8, !dbg !31
  br i1 %1934, label %if_end788, label %if_else787, !dbg !31

if_else787:                                       ; preds = %if_then784
  %2032 = load i64, ptr %flashattn_gqa_decode_no_split.Output_partial.strides145114941497, align 8, !dbg !31, !tbaa !84
  br label %if_end788, !dbg !31

if_end788:                                        ; preds = %if_then784, %if_else787
  %2033 = phi i64 [ %2032, %if_else787 ], [ 1, %if_then784 ], !dbg !31
  %sext1314 = shl i64 %2033, 32, !dbg !31
  %2034 = ashr exact i64 %sext1314, 32, !dbg !31
  %2035 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 2, !dbg !31
  store i64 %2034, ptr %2035, align 8, !dbg !31
  %2036 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 5, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %2036, i8 0, i64 16, i1 false), !dbg !31
  %2037 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %2038 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  %.not1315 = icmp eq ptr %2038, null, !dbg !31
  br i1 %.not1315, label %handle_init789, label %handle_init_end790, !dbg !31, !prof !37

handle_init789:                                   ; preds = %if_end788
  %2039 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %2040 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %2041 = call i32 %2040(ptr %2039, ptr nonnull @.str.26, ptr nonnull %17), !dbg !31
  %2042 = icmp eq i32 %2041, 0, !dbg !31
  br i1 %2042, label %call_end792, label %common.ret, !dbg !31, !prof !33

handle_init_end790:                               ; preds = %call_end792, %if_end788
  %2043 = phi ptr [ %2038, %if_end788 ], [ %2046, %call_end792 ], !dbg !31
  %2044 = call i32 %2037(ptr %2043, ptr nonnull %stack_ffi_any1228, i32 5, ptr nonnull %2036), !dbg !31
  %2045 = icmp eq i32 %2044, 0, !dbg !31
  br i1 %2045, label %if_then795, label %common.ret, !dbg !31, !prof !33

call_end792:                                      ; preds = %handle_init789
  %2046 = load ptr, ptr %17, align 8, !dbg !31
  store ptr %2046, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  br label %handle_init_end790, !dbg !31

if_then795:                                       ; preds = %if_end783, %handle_init_end790
  %2047 = getelementptr inbounds %1, ptr %Output_partial_handle, i64 0, i32 6, !dbg !31
  %2048 = load i64, ptr %2047, align 8, !dbg !31
  %.not1304 = icmp eq i64 %2048, 0, !dbg !31
  br i1 %.not1304, label %if_then806, label %if_then798, !dbg !31, !prof !37

if_then798:                                       ; preds = %if_then795
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %2049 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %2049, align 8, !dbg !31
  %2050 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %2050, align 8, !dbg !31
  %2051 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.22, ptr %2051, align 8, !dbg !31
  %2052 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %2052, align 8, !dbg !31
  %2053 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store i64 0, ptr %2053, align 8, !dbg !31
  %2054 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %2054, align 8, !dbg !31
  %2055 = load i64, ptr %2047, align 8, !dbg !31
  %2056 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 %2055, ptr %2056, align 8, !dbg !31
  %2057 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %2057, i8 0, i64 16, i1 false), !dbg !31
  %2058 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %2059 = load ptr, ptr @.tvm_func.__tvm_error_byte_offset_mismatch, align 8, !dbg !31
  %.not1305 = icmp eq ptr %2059, null, !dbg !31
  br i1 %.not1305, label %handle_init800, label %handle_init_end801, !dbg !31, !prof !37

handle_init800:                                   ; preds = %if_then798
  %2060 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %2061 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %2062 = call i32 %2061(ptr %2060, ptr nonnull @.str.32, ptr nonnull %16), !dbg !31
  %2063 = icmp eq i32 %2062, 0, !dbg !31
  br i1 %2063, label %call_end803, label %common.ret, !dbg !31, !prof !33

handle_init_end801:                               ; preds = %call_end803, %if_then798
  %2064 = phi ptr [ %2059, %if_then798 ], [ %2067, %call_end803 ], !dbg !31
  %2065 = call i32 %2058(ptr %2064, ptr nonnull %stack_ffi_any1228, i32 4, ptr nonnull %2057), !dbg !31
  %2066 = icmp eq i32 %2065, 0, !dbg !31
  br i1 %2066, label %if_then806, label %common.ret, !dbg !31, !prof !33

call_end803:                                      ; preds = %handle_init800
  %2067 = load ptr, ptr %16, align 8, !dbg !31
  store ptr %2067, ptr @.tvm_func.__tvm_error_byte_offset_mismatch, align 8, !dbg !31
  br label %handle_init_end801, !dbg !31

if_then806:                                       ; preds = %if_then795, %handle_init_end801
  %2068 = getelementptr inbounds %1, ptr %Output_partial_handle, i64 0, i32 1, i32 1, !dbg !31
  %2069 = load i32, ptr %2068, align 4, !dbg !31
  %2070 = load i32, ptr %164, align 4, !dbg !31
  %.not1302 = icmp eq i32 %2069, %2070, !dbg !31
  br i1 %.not1302, label %if_then816, label %if_then808, !dbg !31, !prof !37

if_then808:                                       ; preds = %if_then806
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %2071 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %2071, align 8, !dbg !31
  %2072 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %2072, align 8, !dbg !31
  %2073 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.22, ptr %2073, align 8, !dbg !31
  %2074 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %2074, align 8, !dbg !31
  %2075 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store ptr @.str.38, ptr %2075, align 8, !dbg !31
  %2076 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %2076, align 8, !dbg !31
  %2077 = load i32, ptr %164, align 4, !dbg !31
  %2078 = sext i32 %2077 to i64, !dbg !31
  %2079 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 %2078, ptr %2079, align 8, !dbg !31
  %2080 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %2080, align 8, !dbg !31
  %2081 = load i32, ptr %2068, align 4, !dbg !31
  %2082 = sext i32 %2081 to i64, !dbg !31
  %2083 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 2, !dbg !31
  store i64 %2082, ptr %2083, align 8, !dbg !31
  %2084 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 5, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %2084, i8 0, i64 16, i1 false), !dbg !31
  %2085 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %2086 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  %.not1303 = icmp eq ptr %2086, null, !dbg !31
  br i1 %.not1303, label %handle_init810, label %handle_init_end811, !dbg !31, !prof !37

handle_init810:                                   ; preds = %if_then808
  %2087 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %2088 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %2089 = call i32 %2088(ptr %2087, ptr nonnull @.str.26, ptr nonnull %15), !dbg !31
  %2090 = icmp eq i32 %2089, 0, !dbg !31
  br i1 %2090, label %call_end813, label %common.ret, !dbg !31, !prof !33

handle_init_end811:                               ; preds = %call_end813, %if_then808
  %2091 = phi ptr [ %2086, %if_then808 ], [ %2094, %call_end813 ], !dbg !31
  %2092 = call i32 %2085(ptr %2091, ptr nonnull %stack_ffi_any1228, i32 5, ptr nonnull %2084), !dbg !31
  %2093 = icmp eq i32 %2092, 0, !dbg !31
  br i1 %2093, label %if_then816, label %common.ret, !dbg !31, !prof !33

call_end813:                                      ; preds = %handle_init810
  %2094 = load ptr, ptr %15, align 8, !dbg !31
  store ptr %2094, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  br label %handle_init_end811, !dbg !31

if_then816:                                       ; preds = %if_then806, %handle_init_end811
  %2095 = getelementptr inbounds %1, ptr %Output_partial_handle, i64 0, i32 1, i32 0, !dbg !31
  %2096 = load i32, ptr %2095, align 4, !dbg !31
  %.not1300 = icmp eq i32 %2096, 2, !dbg !31
  br i1 %.not1300, label %if_end817, label %if_then819, !dbg !31, !prof !37

if_end817:                                        ; preds = %handle_init_end822, %if_then816
  %2097 = icmp eq ptr %Output_partial, null, !dbg !31
  br i1 %2097, label %if_then830, label %if_end828, !dbg !31, !prof !86

if_then819:                                       ; preds = %if_then816
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %2098 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %2098, align 8, !dbg !31
  %2099 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %2099, align 8, !dbg !31
  %2100 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.22, ptr %2100, align 8, !dbg !31
  %2101 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %2101, align 8, !dbg !31
  %2102 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store i64 2, ptr %2102, align 8, !dbg !31
  %2103 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %2103, align 8, !dbg !31
  %2104 = load i32, ptr %2095, align 4, !dbg !31
  %2105 = sext i32 %2104 to i64, !dbg !31
  %2106 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 %2105, ptr %2106, align 8, !dbg !31
  %2107 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %2107, i8 0, i64 16, i1 false), !dbg !31
  %2108 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %2109 = load ptr, ptr @.tvm_func.__tvm_error_device_type_mismatch, align 8, !dbg !31
  %.not1301 = icmp eq ptr %2109, null, !dbg !31
  br i1 %.not1301, label %handle_init821, label %handle_init_end822, !dbg !31, !prof !37

handle_init821:                                   ; preds = %if_then819
  %2110 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %2111 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %2112 = call i32 %2111(ptr %2110, ptr nonnull @.str.33, ptr nonnull %14), !dbg !31
  %2113 = icmp eq i32 %2112, 0, !dbg !31
  br i1 %2113, label %call_end824, label %common.ret, !dbg !31, !prof !33

handle_init_end822:                               ; preds = %call_end824, %if_then819
  %2114 = phi ptr [ %2109, %if_then819 ], [ %2117, %call_end824 ], !dbg !31
  %2115 = call i32 %2108(ptr %2114, ptr nonnull %stack_ffi_any1228, i32 4, ptr nonnull %2107), !dbg !31
  %2116 = icmp eq i32 %2115, 0, !dbg !31
  br i1 %2116, label %if_end817, label %common.ret, !dbg !31, !prof !33

call_end824:                                      ; preds = %handle_init821
  %2117 = load ptr, ptr %14, align 8, !dbg !31
  store ptr %2117, ptr @.tvm_func.__tvm_error_device_type_mismatch, align 8, !dbg !31
  br label %handle_init_end822, !dbg !31

if_end828:                                        ; preds = %if_end678, %handle_init_end833, %if_end817
  %2118 = getelementptr inbounds %1, ptr %Output_handle, i64 0, i32 3, i32 2, !dbg !31
  %2119 = load i16, ptr %2118, align 2, !dbg !31
  %2120 = icmp ne i16 %2119, 1, !dbg !31
  %2121 = getelementptr inbounds %1, ptr %Output_handle, i64 0, i32 3, i32 1, !dbg !31
  %2122 = load i8, ptr %2121, align 1, !dbg !31
  %2123 = icmp ne i8 %2122, 16, !dbg !31
  %2124 = getelementptr inbounds %1, ptr %Output_handle, i64 0, i32 3, i32 0, !dbg !31
  %2125 = load i8, ptr %2124, align 1, !dbg !31
  %2126 = icmp ne i8 %2125, 2, !dbg !31
  %2127 = or i1 %2123, %2126, !dbg !31
  %2128 = or i1 %2120, %2127, !dbg !31
  br i1 %2128, label %if_then838, label %if_end839, !dbg !31, !prof !33

if_then830:                                       ; preds = %if_end817
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %2129 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %2129, align 8, !dbg !31
  %2130 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %2130, align 8, !dbg !31
  %2131 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.22, ptr %2131, align 8, !dbg !31
  %2132 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %2132, align 8, !dbg !31
  %2133 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store ptr @.str.34, ptr %2133, align 8, !dbg !31
  %2134 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %2134, i8 0, i64 16, i1 false), !dbg !31
  %2135 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %2136 = load ptr, ptr @.tvm_func.__tvm_error_null_ptr, align 8, !dbg !31
  %.not1299 = icmp eq ptr %2136, null, !dbg !31
  br i1 %.not1299, label %handle_init832, label %handle_init_end833, !dbg !31, !prof !37

handle_init832:                                   ; preds = %if_then830
  %2137 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %2138 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %2139 = call i32 %2138(ptr %2137, ptr nonnull @.str.35, ptr nonnull %13), !dbg !31
  %2140 = icmp eq i32 %2139, 0, !dbg !31
  br i1 %2140, label %call_end835, label %common.ret, !dbg !31, !prof !33

handle_init_end833:                               ; preds = %call_end835, %if_then830
  %2141 = phi ptr [ %2136, %if_then830 ], [ %2144, %call_end835 ], !dbg !31
  %2142 = call i32 %2135(ptr %2141, ptr nonnull %stack_ffi_any1228, i32 3, ptr nonnull %2134), !dbg !31
  %2143 = icmp eq i32 %2142, 0, !dbg !31
  br i1 %2143, label %if_end828, label %common.ret, !dbg !31, !prof !33

call_end835:                                      ; preds = %handle_init832
  %2144 = load ptr, ptr %13, align 8, !dbg !31
  store ptr %2144, ptr @.tvm_func.__tvm_error_null_ptr, align 8, !dbg !31
  br label %handle_init_end833, !dbg !31

if_then838:                                       ; preds = %if_end828
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %2145 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %2145, align 8, !dbg !31
  %2146 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %2146, align 8, !dbg !31
  %2147 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.23, ptr %2147, align 8, !dbg !31
  %2148 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %2148, align 8, !dbg !31
  %2149 = load i8, ptr %2124, align 1, !dbg !31
  %2150 = zext i8 %2149 to i64, !dbg !31
  %2151 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store i64 %2150, ptr %2151, align 8, !dbg !31
  %2152 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %2152, align 8, !dbg !31
  %2153 = load i8, ptr %2121, align 1, !dbg !31
  %2154 = zext i8 %2153 to i64, !dbg !31
  %2155 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 %2154, ptr %2155, align 8, !dbg !31
  %2156 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %2156, align 8, !dbg !31
  %2157 = load i16, ptr %2118, align 2, !dbg !31
  %2158 = zext i16 %2157 to i64, !dbg !31
  %2159 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 2, !dbg !31
  store i64 %2158, ptr %2159, align 8, !dbg !31
  %2160 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 5, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %2160, align 8, !dbg !31
  %2161 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 5, i32 2, !dbg !31
  store i64 2, ptr %2161, align 8, !dbg !31
  %2162 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 6, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %2162, align 8, !dbg !31
  %2163 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 6, i32 2, !dbg !31
  store i64 16, ptr %2163, align 8, !dbg !31
  %2164 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 7, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %2164, align 8, !dbg !31
  %2165 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 7, i32 2, !dbg !31
  store i64 1, ptr %2165, align 8, !dbg !31
  %2166 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 8, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %2166, i8 0, i64 16, i1 false), !dbg !31
  %2167 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %2168 = load ptr, ptr @.tvm_func.__tvm_error_dtype_mismatch, align 8, !dbg !31
  %.not1298 = icmp eq ptr %2168, null, !dbg !31
  br i1 %.not1298, label %handle_init840, label %handle_init_end841, !dbg !31, !prof !37

if_end839:                                        ; preds = %handle_init_end841, %if_end828
  %2169 = load i64, ptr %flashattn_gqa_decode_no_split.Output.shape, align 8, !dbg !31, !tbaa !84
  %2170 = and i64 %2169, 4294967295, !dbg !31
  %.not1273 = icmp eq i64 %2170, 1, !dbg !31
  br i1 %.not1273, label %if_end847, label %if_then846, !dbg !31, !prof !37

handle_init840:                                   ; preds = %if_then838
  %2171 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %2172 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %2173 = call i32 %2172(ptr %2171, ptr nonnull @.str.24, ptr nonnull %12), !dbg !31
  %2174 = icmp eq i32 %2173, 0, !dbg !31
  br i1 %2174, label %call_end843, label %common.ret, !dbg !31, !prof !33

handle_init_end841:                               ; preds = %call_end843, %if_then838
  %2175 = phi ptr [ %2168, %if_then838 ], [ %2178, %call_end843 ], !dbg !31
  %2176 = call i32 %2167(ptr %2175, ptr nonnull %stack_ffi_any1228, i32 8, ptr nonnull %2166), !dbg !31
  %2177 = icmp eq i32 %2176, 0, !dbg !31
  br i1 %2177, label %if_end839, label %common.ret, !dbg !31, !prof !33

call_end843:                                      ; preds = %handle_init840
  %2178 = load ptr, ptr %12, align 8, !dbg !31
  store ptr %2178, ptr @.tvm_func.__tvm_error_dtype_mismatch, align 8, !dbg !31
  br label %handle_init_end841, !dbg !31

if_then846:                                       ; preds = %if_end839
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %2179 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %2179, align 8, !dbg !31
  %2180 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %2180, align 8, !dbg !31
  %2181 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.23, ptr %2181, align 8, !dbg !31
  %2182 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %2182, align 8, !dbg !31
  %2183 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store ptr @.str.25, ptr %2183, align 8, !dbg !31
  %2184 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %2184, align 8, !dbg !31
  %2185 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 1, ptr %2185, align 8, !dbg !31
  %2186 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %2186, align 8, !dbg !31
  %2187 = load i64, ptr %flashattn_gqa_decode_no_split.Output.shape, align 8, !dbg !31, !tbaa !84
  %sext1296 = shl i64 %2187, 32, !dbg !31
  %2188 = ashr exact i64 %sext1296, 32, !dbg !31
  %2189 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 2, !dbg !31
  store i64 %2188, ptr %2189, align 8, !dbg !31
  %2190 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 5, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %2190, i8 0, i64 16, i1 false), !dbg !31
  %2191 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %2192 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  %.not1297 = icmp eq ptr %2192, null, !dbg !31
  br i1 %.not1297, label %handle_init848, label %handle_init_end849, !dbg !31, !prof !37

if_end847:                                        ; preds = %handle_init_end849, %if_end839
  %2193 = getelementptr inbounds i64, ptr %flashattn_gqa_decode_no_split.Output.shape, i64 1, !dbg !31
  %2194 = load i64, ptr %2193, align 8, !dbg !31, !tbaa !84
  %2195 = and i64 %2194, 4294967295, !dbg !31
  %.not1274 = icmp eq i64 %2195, 32, !dbg !31
  br i1 %.not1274, label %if_end855, label %if_then854, !dbg !31, !prof !37

handle_init848:                                   ; preds = %if_then846
  %2196 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %2197 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %2198 = call i32 %2197(ptr %2196, ptr nonnull @.str.26, ptr nonnull %11), !dbg !31
  %2199 = icmp eq i32 %2198, 0, !dbg !31
  br i1 %2199, label %call_end851, label %common.ret, !dbg !31, !prof !33

handle_init_end849:                               ; preds = %call_end851, %if_then846
  %2200 = phi ptr [ %2192, %if_then846 ], [ %2203, %call_end851 ], !dbg !31
  %2201 = call i32 %2191(ptr %2200, ptr nonnull %stack_ffi_any1228, i32 5, ptr nonnull %2190), !dbg !31
  %2202 = icmp eq i32 %2201, 0, !dbg !31
  br i1 %2202, label %if_end847, label %common.ret, !dbg !31, !prof !33

call_end851:                                      ; preds = %handle_init848
  %2203 = load ptr, ptr %11, align 8, !dbg !31
  store ptr %2203, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  br label %handle_init_end849, !dbg !31

if_then854:                                       ; preds = %if_end847
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %2204 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %2204, align 8, !dbg !31
  %2205 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %2205, align 8, !dbg !31
  %2206 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.23, ptr %2206, align 8, !dbg !31
  %2207 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %2207, align 8, !dbg !31
  %2208 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store ptr @.str.27, ptr %2208, align 8, !dbg !31
  %2209 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %2209, align 8, !dbg !31
  %2210 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 32, ptr %2210, align 8, !dbg !31
  %2211 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %2211, align 8, !dbg !31
  %2212 = load i64, ptr %2193, align 8, !dbg !31, !tbaa !84
  %sext1294 = shl i64 %2212, 32, !dbg !31
  %2213 = ashr exact i64 %sext1294, 32, !dbg !31
  %2214 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 2, !dbg !31
  store i64 %2213, ptr %2214, align 8, !dbg !31
  %2215 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 5, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %2215, i8 0, i64 16, i1 false), !dbg !31
  %2216 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %2217 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  %.not1295 = icmp eq ptr %2217, null, !dbg !31
  br i1 %.not1295, label %handle_init856, label %handle_init_end857, !dbg !31, !prof !37

if_end855:                                        ; preds = %handle_init_end857, %if_end847
  %2218 = getelementptr inbounds i64, ptr %flashattn_gqa_decode_no_split.Output.shape, i64 2, !dbg !31
  %2219 = load i64, ptr %2218, align 8, !dbg !31, !tbaa !84
  %2220 = and i64 %2219, 4294967295, !dbg !31
  %.not1275 = icmp eq i64 %2220, 128, !dbg !31
  br i1 %.not1275, label %if_end863, label %if_then862, !dbg !31, !prof !37

handle_init856:                                   ; preds = %if_then854
  %2221 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %2222 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %2223 = call i32 %2222(ptr %2221, ptr nonnull @.str.26, ptr nonnull %10), !dbg !31
  %2224 = icmp eq i32 %2223, 0, !dbg !31
  br i1 %2224, label %call_end859, label %common.ret, !dbg !31, !prof !33

handle_init_end857:                               ; preds = %call_end859, %if_then854
  %2225 = phi ptr [ %2217, %if_then854 ], [ %2228, %call_end859 ], !dbg !31
  %2226 = call i32 %2216(ptr %2225, ptr nonnull %stack_ffi_any1228, i32 5, ptr nonnull %2215), !dbg !31
  %2227 = icmp eq i32 %2226, 0, !dbg !31
  br i1 %2227, label %if_end855, label %common.ret, !dbg !31, !prof !33

call_end859:                                      ; preds = %handle_init856
  %2228 = load ptr, ptr %10, align 8, !dbg !31
  store ptr %2228, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  br label %handle_init_end857, !dbg !31

if_then862:                                       ; preds = %if_end855
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %2229 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %2229, align 8, !dbg !31
  %2230 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %2230, align 8, !dbg !31
  %2231 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.23, ptr %2231, align 8, !dbg !31
  %2232 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %2232, align 8, !dbg !31
  %2233 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store ptr @.str.28, ptr %2233, align 8, !dbg !31
  %2234 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %2234, align 8, !dbg !31
  %2235 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 128, ptr %2235, align 8, !dbg !31
  %2236 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %2236, align 8, !dbg !31
  %2237 = load i64, ptr %2218, align 8, !dbg !31, !tbaa !84
  %sext1292 = shl i64 %2237, 32, !dbg !31
  %2238 = ashr exact i64 %sext1292, 32, !dbg !31
  %2239 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 2, !dbg !31
  store i64 %2238, ptr %2239, align 8, !dbg !31
  %2240 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 5, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %2240, i8 0, i64 16, i1 false), !dbg !31
  %2241 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %2242 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  %.not1293 = icmp eq ptr %2242, null, !dbg !31
  br i1 %.not1293, label %handle_init864, label %handle_init_end865, !dbg !31, !prof !37

if_end863:                                        ; preds = %handle_init_end865, %if_end855
  %2243 = icmp eq ptr %flashattn_gqa_decode_no_split.Output.strides, null, !dbg !31
  br i1 %2243, label %if_then887, label %if_end872, !dbg !31

handle_init864:                                   ; preds = %if_then862
  %2244 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %2245 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %2246 = call i32 %2245(ptr %2244, ptr nonnull @.str.26, ptr nonnull %9), !dbg !31
  %2247 = icmp eq i32 %2246, 0, !dbg !31
  br i1 %2247, label %call_end867, label %common.ret, !dbg !31, !prof !33

handle_init_end865:                               ; preds = %call_end867, %if_then862
  %2248 = phi ptr [ %2242, %if_then862 ], [ %2251, %call_end867 ], !dbg !31
  %2249 = call i32 %2241(ptr %2248, ptr nonnull %stack_ffi_any1228, i32 5, ptr nonnull %2240), !dbg !31
  %2250 = icmp eq i32 %2249, 0, !dbg !31
  br i1 %2250, label %if_end863, label %common.ret, !dbg !31, !prof !33

call_end867:                                      ; preds = %handle_init864
  %2251 = load ptr, ptr %9, align 8, !dbg !31
  store ptr %2251, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  br label %handle_init_end865, !dbg !31

if_end872:                                        ; preds = %if_end863
  %2252 = getelementptr inbounds i64, ptr %flashattn_gqa_decode_no_split.Output.strides, i64 2, !dbg !31
  %2253 = load i64, ptr %2252, align 8, !dbg !31, !tbaa !84
  %2254 = and i64 %2253, 4294967295, !dbg !31
  %.not1276 = icmp eq i64 %2254, 1, !dbg !31
  br i1 %.not1276, label %if_end886, label %if_end877, !dbg !31, !prof !37

if_end877:                                        ; preds = %if_end872
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %2255 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %2255, align 8, !dbg !31
  %2256 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %2256, align 8, !dbg !31
  %2257 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.23, ptr %2257, align 8, !dbg !31
  %2258 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %2258, align 8, !dbg !31
  %2259 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store ptr @.str.29, ptr %2259, align 8, !dbg !31
  %2260 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %2260, align 8, !dbg !31
  %2261 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 1, ptr %2261, align 8, !dbg !31
  %2262 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %2262, align 8, !dbg !31
  %2263 = load i64, ptr %2252, align 8, !dbg !31, !tbaa !84
  %sext1290 = shl i64 %2263, 32, !dbg !31
  %2264 = ashr exact i64 %sext1290, 32, !dbg !31
  %2265 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 2, !dbg !31
  store i64 %2264, ptr %2265, align 8, !dbg !31
  %2266 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 5, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %2266, i8 0, i64 16, i1 false), !dbg !31
  %2267 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %2268 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  %.not1291 = icmp eq ptr %2268, null, !dbg !31
  br i1 %.not1291, label %handle_init878, label %handle_init_end879, !dbg !31, !prof !37

handle_init878:                                   ; preds = %if_end877
  %2269 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %2270 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %2271 = call i32 %2270(ptr %2269, ptr nonnull @.str.26, ptr nonnull %8), !dbg !31
  %2272 = icmp eq i32 %2271, 0, !dbg !31
  br i1 %2272, label %call_end881, label %common.ret, !dbg !31, !prof !33

handle_init_end879:                               ; preds = %call_end881, %if_end877
  %2273 = phi ptr [ %2268, %if_end877 ], [ %2276, %call_end881 ], !dbg !31
  %2274 = call i32 %2267(ptr %2273, ptr nonnull %stack_ffi_any1228, i32 5, ptr nonnull %2266), !dbg !31
  %2275 = icmp eq i32 %2274, 0, !dbg !31
  br i1 %2275, label %if_end886, label %common.ret, !dbg !31, !prof !33

call_end881:                                      ; preds = %handle_init878
  %2276 = load ptr, ptr %8, align 8, !dbg !31
  store ptr %2276, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  br label %handle_init_end879, !dbg !31

if_end886:                                        ; preds = %if_end872, %handle_init_end879
  %2277 = getelementptr inbounds i64, ptr %flashattn_gqa_decode_no_split.Output.strides, i64 1, !dbg !31
  %2278 = load i64, ptr %2277, align 8, !dbg !31, !tbaa !84
  %2279 = and i64 %2278, 4294967295, !dbg !31
  %.not1277 = icmp eq i64 %2279, 128, !dbg !31
  br i1 %.not1277, label %if_end900, label %if_then887, !dbg !31, !prof !37

if_then887:                                       ; preds = %if_end863, %if_end886
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %2280 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %2280, align 8, !dbg !31
  %2281 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %2281, align 8, !dbg !31
  %2282 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.23, ptr %2282, align 8, !dbg !31
  %2283 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %2283, align 8, !dbg !31
  %2284 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store ptr @.str.30, ptr %2284, align 8, !dbg !31
  %2285 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %2285, align 8, !dbg !31
  %2286 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 128, ptr %2286, align 8, !dbg !31
  %2287 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %2287, align 8, !dbg !31
  br i1 %2243, label %if_end891, label %if_else890, !dbg !31

if_end888:                                        ; preds = %handle_init_end893
  br i1 %2243, label %if_then901, label %if_end900, !dbg !31

if_else890:                                       ; preds = %if_then887
  %2288 = getelementptr inbounds i64, ptr %flashattn_gqa_decode_no_split.Output.strides, i64 1, !dbg !31
  %2289 = load i64, ptr %2288, align 8, !dbg !31, !tbaa !84
  br label %if_end891, !dbg !31

if_end891:                                        ; preds = %if_then887, %if_else890
  %2290 = phi i64 [ %2289, %if_else890 ], [ 1, %if_then887 ], !dbg !31
  %sext1288 = shl i64 %2290, 32, !dbg !31
  %2291 = ashr exact i64 %sext1288, 32, !dbg !31
  %2292 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 2, !dbg !31
  store i64 %2291, ptr %2292, align 8, !dbg !31
  %2293 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 5, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %2293, i8 0, i64 16, i1 false), !dbg !31
  %2294 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %2295 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  %.not1289 = icmp eq ptr %2295, null, !dbg !31
  br i1 %.not1289, label %handle_init892, label %handle_init_end893, !dbg !31, !prof !37

handle_init892:                                   ; preds = %if_end891
  %2296 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %2297 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %2298 = call i32 %2297(ptr %2296, ptr nonnull @.str.26, ptr nonnull %7), !dbg !31
  %2299 = icmp eq i32 %2298, 0, !dbg !31
  br i1 %2299, label %call_end895, label %common.ret, !dbg !31, !prof !33

handle_init_end893:                               ; preds = %call_end895, %if_end891
  %2300 = phi ptr [ %2295, %if_end891 ], [ %2303, %call_end895 ], !dbg !31
  %2301 = call i32 %2294(ptr %2300, ptr nonnull %stack_ffi_any1228, i32 5, ptr nonnull %2293), !dbg !31
  %2302 = icmp eq i32 %2301, 0, !dbg !31
  br i1 %2302, label %if_end888, label %common.ret, !dbg !31, !prof !33

call_end895:                                      ; preds = %handle_init892
  %2303 = load ptr, ptr %7, align 8, !dbg !31
  store ptr %2303, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  br label %handle_init_end893, !dbg !31

if_end900:                                        ; preds = %if_end886, %if_end888
  %2304 = load i64, ptr %flashattn_gqa_decode_no_split.Output.strides, align 8, !dbg !31, !tbaa !84
  %2305 = and i64 %2304, 4294967295, !dbg !31
  %.not1278 = icmp eq i64 %2305, 4096, !dbg !31
  br i1 %.not1278, label %if_end902, label %if_then901, !dbg !31, !prof !37

if_then901:                                       ; preds = %if_end888, %if_end900
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %2306 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %2306, align 8, !dbg !31
  %2307 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %2307, align 8, !dbg !31
  %2308 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.23, ptr %2308, align 8, !dbg !31
  %2309 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %2309, align 8, !dbg !31
  %2310 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store ptr @.str.31, ptr %2310, align 8, !dbg !31
  %2311 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %2311, align 8, !dbg !31
  %2312 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 4096, ptr %2312, align 8, !dbg !31
  %2313 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %2313, align 8, !dbg !31
  br i1 %2243, label %if_end905, label %if_else904, !dbg !31

if_end902:                                        ; preds = %handle_init_end907, %if_end900
  %2314 = getelementptr inbounds %1, ptr %Output_handle, i64 0, i32 6, !dbg !31
  %2315 = load i64, ptr %2314, align 8, !dbg !31
  %.not1279 = icmp eq i64 %2315, 0, !dbg !31
  br i1 %.not1279, label %if_end913, label %if_then912, !dbg !31, !prof !37

if_else904:                                       ; preds = %if_then901
  %2316 = load i64, ptr %flashattn_gqa_decode_no_split.Output.strides, align 8, !dbg !31, !tbaa !84
  br label %if_end905, !dbg !31

if_end905:                                        ; preds = %if_then901, %if_else904
  %2317 = phi i64 [ %2316, %if_else904 ], [ 1, %if_then901 ], !dbg !31
  %sext = shl i64 %2317, 32, !dbg !31
  %2318 = ashr exact i64 %sext, 32, !dbg !31
  %2319 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 2, !dbg !31
  store i64 %2318, ptr %2319, align 8, !dbg !31
  %2320 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 5, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %2320, i8 0, i64 16, i1 false), !dbg !31
  %2321 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %2322 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  %.not1287 = icmp eq ptr %2322, null, !dbg !31
  br i1 %.not1287, label %handle_init906, label %handle_init_end907, !dbg !31, !prof !37

handle_init906:                                   ; preds = %if_end905
  %2323 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %2324 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %2325 = call i32 %2324(ptr %2323, ptr nonnull @.str.26, ptr nonnull %6), !dbg !31
  %2326 = icmp eq i32 %2325, 0, !dbg !31
  br i1 %2326, label %call_end909, label %common.ret, !dbg !31, !prof !33

handle_init_end907:                               ; preds = %call_end909, %if_end905
  %2327 = phi ptr [ %2322, %if_end905 ], [ %2330, %call_end909 ], !dbg !31
  %2328 = call i32 %2321(ptr %2327, ptr nonnull %stack_ffi_any1228, i32 5, ptr nonnull %2320), !dbg !31
  %2329 = icmp eq i32 %2328, 0, !dbg !31
  br i1 %2329, label %if_end902, label %common.ret, !dbg !31, !prof !33

call_end909:                                      ; preds = %handle_init906
  %2330 = load ptr, ptr %6, align 8, !dbg !31
  store ptr %2330, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  br label %handle_init_end907, !dbg !31

if_then912:                                       ; preds = %if_end902
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %2331 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %2331, align 8, !dbg !31
  %2332 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %2332, align 8, !dbg !31
  %2333 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.23, ptr %2333, align 8, !dbg !31
  %2334 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %2334, align 8, !dbg !31
  %2335 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store i64 0, ptr %2335, align 8, !dbg !31
  %2336 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %2336, align 8, !dbg !31
  %2337 = load i64, ptr %2314, align 8, !dbg !31
  %2338 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 %2337, ptr %2338, align 8, !dbg !31
  %2339 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %2339, i8 0, i64 16, i1 false), !dbg !31
  %2340 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %2341 = load ptr, ptr @.tvm_func.__tvm_error_byte_offset_mismatch, align 8, !dbg !31
  %.not1286 = icmp eq ptr %2341, null, !dbg !31
  br i1 %.not1286, label %handle_init914, label %handle_init_end915, !dbg !31, !prof !37

if_end913:                                        ; preds = %handle_init_end915, %if_end902
  %2342 = getelementptr inbounds %1, ptr %Output_handle, i64 0, i32 1, i32 1, !dbg !31
  %2343 = load i32, ptr %2342, align 4, !dbg !31
  %2344 = load i32, ptr %164, align 4, !dbg !31
  %.not1280 = icmp eq i32 %2343, %2344, !dbg !31
  br i1 %.not1280, label %if_end921, label %if_then920, !dbg !31, !prof !37

handle_init914:                                   ; preds = %if_then912
  %2345 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %2346 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %2347 = call i32 %2346(ptr %2345, ptr nonnull @.str.32, ptr nonnull %5), !dbg !31
  %2348 = icmp eq i32 %2347, 0, !dbg !31
  br i1 %2348, label %call_end917, label %common.ret, !dbg !31, !prof !33

handle_init_end915:                               ; preds = %call_end917, %if_then912
  %2349 = phi ptr [ %2341, %if_then912 ], [ %2352, %call_end917 ], !dbg !31
  %2350 = call i32 %2340(ptr %2349, ptr nonnull %stack_ffi_any1228, i32 4, ptr nonnull %2339), !dbg !31
  %2351 = icmp eq i32 %2350, 0, !dbg !31
  br i1 %2351, label %if_end913, label %common.ret, !dbg !31, !prof !33

call_end917:                                      ; preds = %handle_init914
  %2352 = load ptr, ptr %5, align 8, !dbg !31
  store ptr %2352, ptr @.tvm_func.__tvm_error_byte_offset_mismatch, align 8, !dbg !31
  br label %handle_init_end915, !dbg !31

if_then920:                                       ; preds = %if_end913
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %2353 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %2353, align 8, !dbg !31
  %2354 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %2354, align 8, !dbg !31
  %2355 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.23, ptr %2355, align 8, !dbg !31
  %2356 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %2356, align 8, !dbg !31
  %2357 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store ptr @.str.38, ptr %2357, align 8, !dbg !31
  %2358 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %2358, align 8, !dbg !31
  %2359 = load i32, ptr %164, align 4, !dbg !31
  %2360 = sext i32 %2359 to i64, !dbg !31
  %2361 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 %2360, ptr %2361, align 8, !dbg !31
  %2362 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %2362, align 8, !dbg !31
  %2363 = load i32, ptr %2342, align 4, !dbg !31
  %2364 = sext i32 %2363 to i64, !dbg !31
  %2365 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 2, !dbg !31
  store i64 %2364, ptr %2365, align 8, !dbg !31
  %2366 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 5, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %2366, i8 0, i64 16, i1 false), !dbg !31
  %2367 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %2368 = load ptr, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  %.not1285 = icmp eq ptr %2368, null, !dbg !31
  br i1 %.not1285, label %handle_init922, label %handle_init_end923, !dbg !31, !prof !37

if_end921:                                        ; preds = %handle_init_end923, %if_end913
  %2369 = getelementptr inbounds %1, ptr %Output_handle, i64 0, i32 1, i32 0, !dbg !31
  %2370 = load i32, ptr %2369, align 4, !dbg !31
  %.not1281 = icmp eq i32 %2370, 2, !dbg !31
  br i1 %.not1281, label %if_end929, label %if_then928, !dbg !31, !prof !37

handle_init922:                                   ; preds = %if_then920
  %2371 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %2372 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %2373 = call i32 %2372(ptr %2371, ptr nonnull @.str.26, ptr nonnull %4), !dbg !31
  %2374 = icmp eq i32 %2373, 0, !dbg !31
  br i1 %2374, label %call_end925, label %common.ret, !dbg !31, !prof !33

handle_init_end923:                               ; preds = %call_end925, %if_then920
  %2375 = phi ptr [ %2368, %if_then920 ], [ %2378, %call_end925 ], !dbg !31
  %2376 = call i32 %2367(ptr %2375, ptr nonnull %stack_ffi_any1228, i32 5, ptr nonnull %2366), !dbg !31
  %2377 = icmp eq i32 %2376, 0, !dbg !31
  br i1 %2377, label %if_end921, label %common.ret, !dbg !31, !prof !33

call_end925:                                      ; preds = %handle_init922
  %2378 = load ptr, ptr %4, align 8, !dbg !31
  store ptr %2378, ptr @.tvm_func.__tvm_error_expect_eq, align 8, !dbg !31
  br label %handle_init_end923, !dbg !31

if_then928:                                       ; preds = %if_end921
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %2379 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %2379, align 8, !dbg !31
  %2380 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %2380, align 8, !dbg !31
  %2381 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.23, ptr %2381, align 8, !dbg !31
  %2382 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %2382, align 8, !dbg !31
  %2383 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store i64 2, ptr %2383, align 8, !dbg !31
  %2384 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %2384, align 8, !dbg !31
  %2385 = load i32, ptr %2369, align 4, !dbg !31
  %2386 = sext i32 %2385 to i64, !dbg !31
  %2387 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !31
  store i64 %2386, ptr %2387, align 8, !dbg !31
  %2388 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %2388, i8 0, i64 16, i1 false), !dbg !31
  %2389 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %2390 = load ptr, ptr @.tvm_func.__tvm_error_device_type_mismatch, align 8, !dbg !31
  %.not1284 = icmp eq ptr %2390, null, !dbg !31
  br i1 %.not1284, label %handle_init930, label %handle_init_end931, !dbg !31, !prof !37

if_end929:                                        ; preds = %handle_init_end931, %if_end921
  %2391 = icmp eq ptr %Output, null, !dbg !31
  br i1 %2391, label %if_then936, label %if_end937, !dbg !31, !prof !33

handle_init930:                                   ; preds = %if_then928
  %2392 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %2393 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %2394 = call i32 %2393(ptr %2392, ptr nonnull @.str.33, ptr nonnull %3), !dbg !31
  %2395 = icmp eq i32 %2394, 0, !dbg !31
  br i1 %2395, label %call_end933, label %common.ret, !dbg !31, !prof !33

handle_init_end931:                               ; preds = %call_end933, %if_then928
  %2396 = phi ptr [ %2390, %if_then928 ], [ %2399, %call_end933 ], !dbg !31
  %2397 = call i32 %2389(ptr %2396, ptr nonnull %stack_ffi_any1228, i32 4, ptr nonnull %2388), !dbg !31
  %2398 = icmp eq i32 %2397, 0, !dbg !31
  br i1 %2398, label %if_end929, label %common.ret, !dbg !31, !prof !33

call_end933:                                      ; preds = %handle_init930
  %2399 = load ptr, ptr %3, align 8, !dbg !31
  store ptr %2399, ptr @.tvm_func.__tvm_error_device_type_mismatch, align 8, !dbg !31
  br label %handle_init_end931, !dbg !31

if_then936:                                       ; preds = %if_end929
  store <2 x i32> <i32 8, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %2400 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store ptr @.str.15, ptr %2400, align 8, !dbg !31
  %2401 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %2401, align 8, !dbg !31
  %2402 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store ptr @.str.23, ptr %2402, align 8, !dbg !31
  %2403 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  store <2 x i32> <i32 8, i32 0>, ptr %2403, align 8, !dbg !31
  %2404 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  store ptr @.str.34, ptr %2404, align 8, !dbg !31
  %2405 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %2405, i8 0, i64 16, i1 false), !dbg !31
  %2406 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %2407 = load ptr, ptr @.tvm_func.__tvm_error_null_ptr, align 8, !dbg !31
  %.not1283 = icmp eq ptr %2407, null, !dbg !31
  br i1 %.not1283, label %handle_init938, label %handle_init_end939, !dbg !31, !prof !37

if_end937:                                        ; preds = %handle_init_end939, %if_end929
  %2408 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 1, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %stack_ffi_any1228, align 8, !dbg !31
  %2409 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 0, i32 2, !dbg !31
  store i64 2, ptr %2409, align 8, !dbg !31
  %2410 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 0, !dbg !31
  %2411 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 1, !dbg !31
  store <2 x i32> <i32 1, i32 0>, ptr %2410, align 8, !dbg !31
  %2412 = sext i32 %dev_id to i64, !dbg !31
  %2413 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 1, i32 2, !dbg !31
  store i64 %2412, ptr %2413, align 8, !dbg !31
  %2414 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 0, !dbg !31
  %2415 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 1, !dbg !31
  %2416 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 2, i32 2, !dbg !31
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %2414, i8 0, i64 16, i1 false), !dbg !31
  %2417 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !31, !tbaa !34
  %2418 = load ptr, ptr @.tvm_func.__tvm_set_device, align 8, !dbg !31
  %.not1282 = icmp eq ptr %2418, null, !dbg !31
  br i1 %.not1282, label %handle_init944, label %handle_init_end945, !dbg !31, !prof !37

handle_init938:                                   ; preds = %if_then936
  %2419 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %2420 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %2421 = call i32 %2420(ptr %2419, ptr nonnull @.str.35, ptr nonnull %2), !dbg !31
  %2422 = icmp eq i32 %2421, 0, !dbg !31
  br i1 %2422, label %call_end941, label %common.ret, !dbg !31, !prof !33

handle_init_end939:                               ; preds = %call_end941, %if_then936
  %2423 = phi ptr [ %2407, %if_then936 ], [ %2426, %call_end941 ], !dbg !31
  %2424 = call i32 %2406(ptr %2423, ptr nonnull %stack_ffi_any1228, i32 3, ptr nonnull %2405), !dbg !31
  %2425 = icmp eq i32 %2424, 0, !dbg !31
  br i1 %2425, label %if_end937, label %common.ret, !dbg !31, !prof !33

call_end941:                                      ; preds = %handle_init938
  %2426 = load ptr, ptr %2, align 8, !dbg !31
  store ptr %2426, ptr @.tvm_func.__tvm_error_null_ptr, align 8, !dbg !31
  br label %handle_init_end939, !dbg !31

handle_init944:                                   ; preds = %if_end937
  %2427 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !31, !tbaa !34
  %2428 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !31, !tbaa !34
  %2429 = call i32 %2428(ptr %2427, ptr nonnull @.str.39, ptr nonnull %1), !dbg !31
  %2430 = icmp eq i32 %2429, 0, !dbg !31
  br i1 %2430, label %call_end947, label %common.ret, !dbg !31, !prof !33

handle_init_end945:                               ; preds = %call_end947, %if_end937
  %2431 = phi ptr [ %2418, %if_end937 ], [ %2434, %call_end947 ], !dbg !31
  %2432 = call i32 %2417(ptr %2431, ptr nonnull %stack_ffi_any1228, i32 2, ptr nonnull %2414), !dbg !31
  %2433 = icmp eq i32 %2432, 0, !dbg !31
  br i1 %2433, label %call_end949, label %common.ret, !dbg !31, !prof !33

call_end947:                                      ; preds = %handle_init944
  %2434 = load ptr, ptr %1, align 8, !dbg !31
  store ptr %2434, ptr @.tvm_func.__tvm_set_device, align 8, !dbg !31
  br label %handle_init_end945, !dbg !31

call_end949:                                      ; preds = %handle_init_end945
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %0), !dbg !15
  call void @llvm.dbg.value(metadata ptr %K, metadata !24, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.value(metadata ptr %stack_ffi_any1228, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.value(metadata ptr %Output, metadata !26, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.value(metadata ptr %Q, metadata !27, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.value(metadata ptr %V, metadata !28, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.value(metadata ptr %mask, metadata !29, metadata !DIExpression()), !dbg !15
  %..i = select i1 %879, i32 0, i32 4, !dbg !15
  store i32 %..i, ptr %stack_ffi_any1228, align 8, !dbg !15
  store i32 0, ptr %2408, align 4, !dbg !15
  store ptr %K, ptr %2409, align 8, !dbg !15
  %spec.select.i = select i1 %2391, i32 0, i32 4, !dbg !15
  store i32 %spec.select.i, ptr %2410, align 8, !dbg !15
  store i32 0, ptr %2411, align 4, !dbg !15
  store ptr %Output, ptr %2413, align 8, !dbg !15
  %.sink17.i = select i1 %545, i32 0, i32 4, !dbg !15
  store i32 %.sink17.i, ptr %2414, align 8, !dbg !15
  store i32 0, ptr %2415, align 4, !dbg !15
  store ptr %Q, ptr %2416, align 8, !dbg !15
  %.sink18.i = select i1 %1213, i32 0, i32 4, !dbg !15
  %2435 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 0, !dbg !15
  store i32 %.sink18.i, ptr %2435, align 8, !dbg !15
  %2436 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 1, !dbg !15
  store i32 0, ptr %2436, align 4, !dbg !15
  %2437 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 3, i32 2, !dbg !15
  store ptr %V, ptr %2437, align 8, !dbg !15
  %.sink19.i = select i1 %1495, i32 0, i32 4, !dbg !15
  %2438 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 0, !dbg !15
  store i32 %.sink19.i, ptr %2438, align 8, !dbg !15
  %2439 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 1, !dbg !15
  store i32 0, ptr %2439, align 4, !dbg !15
  %2440 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 4, i32 2, !dbg !15
  store ptr %mask, ptr %2440, align 8, !dbg !15
  %2441 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 5, i32 0, !dbg !15
  store <2 x i32> <i32 1, i32 0>, ptr %2441, align 8, !dbg !15
  %2442 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 5, i32 2, !dbg !15
  store i64 1, ptr %2442, align 8, !dbg !15
  %2443 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 6, i32 0, !dbg !15
  store <2 x i32> <i32 1, i32 0>, ptr %2443, align 8, !dbg !15
  %2444 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 6, i32 2, !dbg !15
  store i64 8, ptr %2444, align 8, !dbg !15
  %2445 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 7, i32 0, !dbg !15
  store <2 x i32> <i32 1, i32 0>, ptr %2445, align 8, !dbg !15
  %2446 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 7, i32 2, !dbg !15
  store i64 1, ptr %2446, align 8, !dbg !15
  %2447 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 8, i32 0, !dbg !15
  store <2 x i32> <i32 1, i32 0>, ptr %2447, align 8, !dbg !15
  %2448 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 8, i32 2, !dbg !15
  store i64 128, ptr %2448, align 8, !dbg !15
  %2449 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 9, i32 0, !dbg !15
  store <2 x i32> <i32 1, i32 0>, ptr %2449, align 8, !dbg !15
  %2450 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 9, i32 2, !dbg !15
  store i64 1, ptr %2450, align 8, !dbg !15
  %2451 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 10, i32 0, !dbg !15
  store <2 x i32> <i32 1, i32 0>, ptr %2451, align 8, !dbg !15
  %2452 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 10, i32 2, !dbg !15
  store i64 1, ptr %2452, align 8, !dbg !15
  %2453 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 11, i32 0, !dbg !15
  store <2 x i32> <i32 1, i32 0>, ptr %2453, align 8, !dbg !15
  %2454 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 11, i32 2, !dbg !15
  store i64 81920, ptr %2454, align 8, !dbg !15
  %2455 = getelementptr inbounds %0, ptr %stack_ffi_any1228, i64 12, i32 0, !dbg !15
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %2455, i8 0, i64 16, i1 false), !dbg !15
  %2456 = load ptr, ptr @__TVMFFIFunctionCall, align 8, !dbg !15, !tbaa !34
  %2457 = load ptr, ptr @.tvm_func.flashattn_gqa_decode_no_split_kernel, align 8, !dbg !15
  %.not.i = icmp eq ptr %2457, null, !dbg !15
  br i1 %.not.i, label %handle_init.i, label %handle_init_end.i, !dbg !15, !prof !37

handle_init.i:                                    ; preds = %call_end949
  %2458 = load ptr, ptr @__tvm_ffi__library_ctx, align 8, !dbg !15, !tbaa !34
  %2459 = load ptr, ptr @__TVMBackendGetFuncFromEnv, align 8, !dbg !15, !tbaa !34
  %2460 = call i32 %2459(ptr %2458, ptr nonnull @.str.40, ptr nonnull %0), !dbg !15
  %2461 = icmp eq i32 %2460, 0, !dbg !15
  br i1 %2461, label %call_end.i, label %flashattn_gqa_decode_no_split_compute_.exit, !dbg !15, !prof !33

handle_init_end.i:                                ; preds = %call_end.i, %call_end949
  %2462 = phi ptr [ %2457, %call_end949 ], [ %2464, %call_end.i ], !dbg !15
  %2463 = call i32 %2456(ptr %2462, ptr nonnull %stack_ffi_any1228, i32 12, ptr nonnull %2455), !dbg !15
  br label %flashattn_gqa_decode_no_split_compute_.exit, !dbg !15

call_end.i:                                       ; preds = %handle_init.i
  %2464 = load ptr, ptr %0, align 8, !dbg !15
  store ptr %2464, ptr @.tvm_func.flashattn_gqa_decode_no_split_kernel, align 8, !dbg !15
  br label %handle_init_end.i, !dbg !15

flashattn_gqa_decode_no_split_compute_.exit:      ; preds = %handle_init.i, %handle_init_end.i
  %common.ret.op.i = phi i32 [ %2460, %handle_init.i ], [ %2463, %handle_init_end.i ]
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %0), !dbg !15
  br label %common.ret
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: inaccessiblememonly mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.assume(i1 noundef) #2

define weak dllexport i32 @__tvm_ffi_main(ptr %0, ptr %1, i32 %2, ptr %3) local_unnamed_addr {
entry:
  %4 = tail call i32 @flashattn_gqa_decode_no_split(ptr poison, ptr %1, i32 %2, ptr poison), !dbg !31
  ret i32 %4, !dbg !31
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
!5 = distinct !DISubprogram(name: "flashattn_gqa_decode_no_split", scope: !1, file: !1, type: !6, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !10)
!6 = !DISubroutineType(types: !7)
!7 = !{!8, !9, !9, !8, !9}
!8 = !DIBasicType(name: "int32", size: 32, encoding: DW_ATE_signed)
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null)
!10 = !{!11, !12, !13, !14}
!11 = !DILocalVariable(name: "self_handle", arg: 1, scope: !5, file: !1, type: !9)
!12 = !DILocalVariable(name: "args", arg: 2, scope: !5, file: !1, type: !9)
!13 = !DILocalVariable(name: "num_args", arg: 3, scope: !5, file: !1, type: !8)
!14 = !DILocalVariable(name: "result", arg: 4, scope: !5, file: !1, type: !9)
!15 = !DILocation(line: 0, scope: !16, inlinedAt: !30)
!16 = distinct !DISubprogram(name: "flashattn_gqa_decode_no_split_compute_", scope: !1, file: !1, type: !17, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !23)
!17 = !DISubroutineType(types: !18)
!18 = !{!8, !19, !9, !19, !19, !19, !21}
!19 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !20)
!20 = !DIBasicType(name: "float16", size: 16, encoding: DW_ATE_float)
!21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !22)
!22 = !DIBasicType(name: "uint8", size: 8, encoding: DW_ATE_unsigned)
!23 = !{!24, !25, !26, !27, !28, !29}
!24 = !DILocalVariable(name: "K", arg: 1, scope: !16, file: !1, type: !19)
!25 = !DILocalVariable(name: "stack_ffi_any", arg: 2, scope: !16, file: !1, type: !9)
!26 = !DILocalVariable(name: "Output", arg: 3, scope: !16, file: !1, type: !19)
!27 = !DILocalVariable(name: "Q", arg: 4, scope: !16, file: !1, type: !19)
!28 = !DILocalVariable(name: "V", arg: 5, scope: !16, file: !1, type: !19)
!29 = !DILocalVariable(name: "mask", arg: 6, scope: !16, file: !1, type: !21)
!30 = distinct !DILocation(line: 0, scope: !5)
!31 = !DILocation(line: 0, scope: !5)
!32 = !DILocalVariable(name: "stack_ffi_any", scope: !5, file: !1, type: !9)
!33 = !{!"branch_weights", i32 1048576, i32 1}
!34 = !{!35, !35, i64 0}
!35 = !{!"ctx_ptr", !36, i64 0}
!36 = !{!"tvm-tbaa"}
!37 = !{!"branch_weights", i32 1, i32 1048576}
!38 = !DILocalVariable(name: "Q_handle.type_index", scope: !5, file: !1, type: !8)
!39 = !DILocalVariable(name: "K_handle.type_index", scope: !5, file: !1, type: !8)
!40 = !DILocalVariable(name: "V_handle.type_index", scope: !5, file: !1, type: !8)
!41 = !DILocalVariable(name: "mask_handle.type_index", scope: !5, file: !1, type: !8)
!42 = !DILocalVariable(name: "glse_handle.type_index", scope: !5, file: !1, type: !8)
!43 = !DILocalVariable(name: "Output_partial_handle.type_index", scope: !5, file: !1, type: !8)
!44 = !DILocalVariable(name: "Output_handle.type_index", scope: !5, file: !1, type: !8)
!45 = !DILocalVariable(name: "Q_handle", scope: !5, file: !1, type: !9)
!46 = !DILocalVariable(name: "K_handle", scope: !5, file: !1, type: !9)
!47 = !DILocalVariable(name: "V_handle", scope: !5, file: !1, type: !9)
!48 = !DILocalVariable(name: "mask_handle", scope: !5, file: !1, type: !9)
!49 = !DILocalVariable(name: "glse_handle", scope: !5, file: !1, type: !9)
!50 = !DILocalVariable(name: "Output_partial_handle", scope: !5, file: !1, type: !9)
!51 = !DILocalVariable(name: "Output_handle", scope: !5, file: !1, type: !9)
!52 = !DILocalVariable(name: "flashattn_gqa_decode_no_split.Q_is_null", scope: !5, file: !1, type: !53)
!53 = !DIBasicType(name: "bool", size: 8, encoding: DW_ATE_boolean)
!54 = !DILocalVariable(name: "flashattn_gqa_decode_no_split.K_is_null", scope: !5, file: !1, type: !53)
!55 = !DILocalVariable(name: "flashattn_gqa_decode_no_split.V_is_null", scope: !5, file: !1, type: !53)
!56 = !DILocalVariable(name: "flashattn_gqa_decode_no_split.mask_is_null", scope: !5, file: !1, type: !53)
!57 = !DILocalVariable(name: "flashattn_gqa_decode_no_split.glse_is_null", scope: !5, file: !1, type: !53)
!58 = !DILocalVariable(name: "flashattn_gqa_decode_no_split.Output_partial_is_null", scope: !5, file: !1, type: !53)
!59 = !DILocalVariable(name: "flashattn_gqa_decode_no_split.Output_is_null", scope: !5, file: !1, type: !53)
!60 = !DILocalVariable(name: "flashattn_gqa_decode_no_split.Q.shape", scope: !5, file: !1, type: !61)
!61 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !62)
!62 = !DIBasicType(name: "int64", size: 64, encoding: DW_ATE_signed)
!63 = !DILocalVariable(name: "flashattn_gqa_decode_no_split.K.shape", scope: !5, file: !1, type: !61)
!64 = !DILocalVariable(name: "flashattn_gqa_decode_no_split.V.shape", scope: !5, file: !1, type: !61)
!65 = !DILocalVariable(name: "flashattn_gqa_decode_no_split.mask.shape", scope: !5, file: !1, type: !61)
!66 = !DILocalVariable(name: "flashattn_gqa_decode_no_split.glse.shape", scope: !5, file: !1, type: !61)
!67 = !DILocalVariable(name: "flashattn_gqa_decode_no_split.Output_partial.shape", scope: !5, file: !1, type: !61)
!68 = !DILocalVariable(name: "flashattn_gqa_decode_no_split.Output.shape", scope: !5, file: !1, type: !61)
!69 = !DILocalVariable(name: "flashattn_gqa_decode_no_split.Q.strides", scope: !5, file: !1, type: !61)
!70 = !DILocalVariable(name: "dev_id", scope: !5, file: !1, type: !8)
!71 = !DILocalVariable(name: "Q", scope: !5, file: !1, type: !19)
!72 = !DILocalVariable(name: "flashattn_gqa_decode_no_split.K.strides", scope: !5, file: !1, type: !61)
!73 = !DILocalVariable(name: "K", scope: !5, file: !1, type: !19)
!74 = !DILocalVariable(name: "flashattn_gqa_decode_no_split.V.strides", scope: !5, file: !1, type: !61)
!75 = !DILocalVariable(name: "V", scope: !5, file: !1, type: !19)
!76 = !DILocalVariable(name: "flashattn_gqa_decode_no_split.mask.strides", scope: !5, file: !1, type: !61)
!77 = !DILocalVariable(name: "mask", scope: !5, file: !1, type: !21)
!78 = !DILocalVariable(name: "flashattn_gqa_decode_no_split.glse.strides", scope: !5, file: !1, type: !61)
!79 = !DILocalVariable(name: "glse", scope: !5, file: !1, type: !19)
!80 = !DILocalVariable(name: "flashattn_gqa_decode_no_split.Output_partial.strides", scope: !5, file: !1, type: !61)
!81 = !DILocalVariable(name: "Output_partial", scope: !5, file: !1, type: !19)
!82 = !DILocalVariable(name: "flashattn_gqa_decode_no_split.Output.strides", scope: !5, file: !1, type: !61)
!83 = !DILocalVariable(name: "Output", scope: !5, file: !1, type: !19)
!84 = !{!85, !85, i64 0}
!85 = !{!"tvm-alias", !36}
!86 = !{!"branch_weights", i32 1048576, i32 1048578}
