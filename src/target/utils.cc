/*!
 * \file tl/target/utils.cc
 * \brief helper functions for target attributes.
 */

#include "utils.h"

#include "../support/ffi_aliases.h"
#include <tvm/node/node.h>

namespace tvm {
namespace tl {

bool TargetIsCuda(Target target) {
  return target->GetTargetDeviceType() == kDLCUDA;
}

int GetArchInt(Target target) {
  auto s = target->GetAttr<tvm::ffi::String>("arch");
  ICHECK(s.has_value());
  const std::string arch_str = s.value();
  ICHECK(arch_str.size() >= 3);
  ICHECK_EQ(arch_str.compare(0, 3, "sm_"), 0)
      << "arch string must start with sm_";
  return std::stoi(arch_str.substr(3));
}

bool TargetIsVolta(Target target) {
  if (!TargetIsCuda(target))
    return false;
  int arch = GetArchInt(target);
  return arch >= 70 && arch < 75;
}

bool TargetIsTuring(Target target) {
  if (!TargetIsCuda(target))
    return false;
  int arch = GetArchInt(target);
  return arch >= 75 && arch < 80;
}

bool TargetIsAmpere(Target target) {
  if (!TargetIsCuda(target))
    return false;
  int arch = GetArchInt(target);
  return arch >= 80 && arch < 90;
}

bool TargetIsHopper(Target target) {
  if (!TargetIsCuda(target))
    return false;
  int arch = GetArchInt(target);
  return arch >= 90 && arch < 100;
}

bool TargetIsSm100(Target target) {
  if (!TargetIsCuda(target))
    return false;
  int arch = GetArchInt(target);
  return arch >= 100 & arch <= 110;
}

bool TargetIsSM120(Target target) {
  if (!TargetIsCuda(target))
    return false;
  int arch = GetArchInt(target);
  return arch >= 120 && arch < 130;
}

bool TargetHasAsyncCopy(Target target) {
  if (TargetIsCuda(target)) {
    int arch = GetArchInt(target);
    return arch >= 80;
  }

  return false;
}
bool TargetHasLdmatrix(Target target) {
  if (!TargetIsCuda(target))
    return false;
  int arch = GetArchInt(target);
  return arch >= 75;
}

bool TargetHasStmatrix(Target target) {
  if (!TargetIsCuda(target))
    return false;
  int arch = GetArchInt(target);
  return arch >= 90;
}

bool TargetHasTmem(Target target) {
  if (!TargetIsCuda(target))
    return false;
  return TargetIsSm100(target);
}

bool TargetHasBulkCopy(Target target) {
  if (!TargetIsCuda(target))
    return false;
  int arch = GetArchInt(target);
  return arch >= 90;
}

int TargetGetWarpSize(Target target) {
  int res = 32;
  return res;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tl.TargetIsCuda",
           [](Target target) { return TargetIsCuda(target); })
      .def("tl.TargetIsVolta",
           [](Target target) { return TargetIsVolta(target); })
      .def("tl.TargetIsTuring",
           [](Target target) { return TargetIsTuring(target); })
      .def("tl.TargetIsAmpere",
           [](Target target) { return TargetIsAmpere(target); })
      .def("tl.TargetIsHopper",
           [](Target target) { return TargetIsHopper(target); })
      .def("tl.TargetIsSM120",
           [](Target target) { return TargetIsSM120(target); })
      .def("tl.TargetHasAsyncCopy",
           [](Target target) { return TargetHasAsyncCopy(target); })
      .def("tl.TargetHasLdmatrix",
           [](Target target) { return TargetHasLdmatrix(target); })
      .def("tl.TargetHasStmatrix",
           [](Target target) { return TargetHasStmatrix(target); })
      .def("tl.TargetHasBulkCopy",
           [](Target target) { return TargetHasBulkCopy(target); })
      .def("tl.TargetGetWarpSize",
           [](Target target) { return TargetGetWarpSize(target); });
}

} // namespace tl
} // namespace tvm
