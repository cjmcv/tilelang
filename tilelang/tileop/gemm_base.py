from __future__ import annotations
from enum import IntEnum

from dataclasses import dataclass
from tilelang import tvm as tvm
from tvm.target import Target
from tvm import tir
from tilelang import language as T
from tilelang.utils.language import is_shared, is_fragment
from tvm.ir.base import Node
from tvm.ir import PrimExpr


class GemmWarpPolicy(IntEnum):
    """
    Enumeration for GEMM Warp Partitioning Policies.
    """

    Square = 0  # Balance warps evenly in a "square" aspect ratio.
    FullRow = 1  # Assign all warps to rows.
    FullCol = 2  # Assign all warps to columns.

    def is_square(self) -> bool:
        """
        Check if the policy is a square partitioning.

        Returns:
            bool: True if the policy is square, False otherwise.
        """
        return self == GemmWarpPolicy.Square

    def is_full_row(self) -> bool:
        """
        Check if the policy is a full row partitioning.

        Returns:
            bool: True if the policy is full row, False otherwise.
        """
        return self == GemmWarpPolicy.FullRow

    def is_full_col(self) -> bool:
        """
        Check if the policy is a full column partitioning.

        Returns:
            bool: True if the policy is full column, False otherwise.
        """
        return self == GemmWarpPolicy.FullCol

    @staticmethod
    def to_prime_factors(num):
        """
        Compute the prime factorization of a given number.

        Args:
            num (int): The number to factorize.

        Returns:
            list: A list of prime factors of the number.
        """
        factors = []
        i = 2
        # Find all prime factors up to the square root of the number.
        while i * i <= num:
            while num % i == 0:  # Check divisibility by `i`.
                factors.append(i)
                num //= i
            i += 1
        # If the remaining number is greater than 1, it's a prime factor.
        if num > 1:
            factors.append(num)
        return factors

    def compute_warp_partition(self, M, N, num_warps):
        """
        Compute the warp partition (m_warp, n_warp) based on the given policy.

        Args:
            M (int): The number of rows in the GEMM workload.
            N (int): The number of columns in the GEMM workload.
            num_warps (int): The total number of warps available.

        Returns:
            tuple: A tuple (m_warp, n_warp) representing the partitioning of warps.

        Raises:
            ValueError: If the policy is invalid or the partitioning fails.
            AssertionError: If M or N is not divisible by the required factor for FullRow or FullCol policies.
        """
        m_warp = 1  # Initial warp count for rows.
        n_warp = 1  # Initial warp count for columns.

        if self.is_full_row():
            # FullRow policy: Allocate all warps to rows.
            m_warp = num_warps
            n_warp = 1

            # If M cannot be evenly divided by m_warp*16, try to split remaining warps to N
            if M % (m_warp * 16) != 0:
                # Calculate how many warps we can use for M
                max_m_warps = M // 16
                m_warp = max_m_warps
                # Use remaining warps for N
                n_warp = num_warps // m_warp
                if n_warp == 0:
                    n_warp = 1

        elif self.is_full_col():
            # FullCol policy: Allocate all warps to columns.
            m_warp = 1
            n_warp = num_warps

            # If N cannot be evenly divided by n_warp*8, try to split remaining warps to M
            if N % (n_warp * 8) != 0:
                # Calculate how many warps we can use for N
                max_n_warps = N // 8
                n_warp = max_n_warps
                # Use remaining warps for M
                m_warp = num_warps // n_warp
                if m_warp == 0:
                    m_warp = 1

        elif self.is_square():
            # First calculate the maximum possible warps for each dimension
            max_m_warps = M // 16  # Each warp needs at least 16 elements in M
            max_n_warps = N // 8  # Each warp needs at least 8 elements in N

            # Calculate the ideal ratio of M/N warps based on the matrix dimensions
            ideal_ratio = 1.0
            if N > 0:
                ideal_ratio = float(M) / N

            # Start with a balanced initial guess
            m_warp = 1
            n_warp = 1

            # Try to find the best balanced partition
            best_m = 1
            best_n = 1
            best_balance = float("inf")

            # Try all possible combinations that satisfy the constraints
            for m in range(1, min(max_m_warps, num_warps) + 1):
                n = num_warps // m
                if n > max_n_warps:
                    continue
                if m * n != num_warps:
                    continue

                # Calculate how balanced this partition is
                m_per_warp = float(M) / (m * 16)
                n_per_warp = float(N) / (n * 8)
                balance = abs(m_per_warp / n_per_warp - ideal_ratio)

                if balance < best_balance:
                    best_balance = balance
                    best_m = m
                    best_n = n

            m_warp = best_m
            n_warp = best_n

        else:
            # Raise an error for unknown policies.
            raise ValueError(f"Unknown GemmWarpPolicy: {self}")

        return m_warp, n_warp

    @classmethod
    def from_warp_partition(cls, m_warp: int, n_warp: int) -> GemmWarpPolicy:
        """
        Determine the warp policy based on the given warp partitioning.

        Args:
            m_warp (int): Number of warps in the row dimension
            n_warp (int): Number of warps in the column dimension

        Returns:
            GemmWarpPolicy: The corresponding warp policy

        Examples:
            >>> GemmWarpPolicy.from_block_row_cols(4, 1)  # All warps in rows
            GemmWarpPolicy.FullRow
            >>> GemmWarpPolicy.from_block_row_cols(1, 4)  # All warps in columns
            GemmWarpPolicy.FullCol
            >>> GemmWarpPolicy.from_block_row_cols(2, 2)  # Balanced distribution
            GemmWarpPolicy.Square
        """
        if n_warp == 1 and m_warp > 1:
            return cls.FullRow
        elif m_warp == 1 and n_warp > 1:
            return cls.FullCol
        else:
            return cls.Square


@dataclass
class GemmBase:
    gemm_node: Node

    def infer_layout(self, target: Target, thread_nums: int):
        raise NotImplementedError("infer_layout is not implemented")

    def lower(self, target: Target, thread_nums: int, thread_var: tir.Var):
        raise NotImplementedError("lower is not implemented")

    def is_gemm_ss(self) -> bool:
        return is_shared(self.A) and is_shared(self.B)

    def is_gemm_sr(self) -> bool:
        return is_shared(self.A) and is_fragment(self.B)

    def is_gemm_rs(self) -> bool:
        return is_fragment(self.A) and is_shared(self.B)

    def is_gemm_rr(self) -> bool:
        return is_fragment(self.A) and is_fragment(self.B)

    @property
    def M(self) -> int:
        return getattr(self.gemm_node, "m", None)

    @property
    def N(self) -> int:
        return getattr(self.gemm_node, "n", None)

    @property
    def K(self) -> int:
        return getattr(self.gemm_node, "k", None)

    @property
    def trans_A(self) -> bool:
        return getattr(self.gemm_node, "transA", None)

    @property
    def trans_B(self) -> bool:
        return getattr(self.gemm_node, "transB", None)

    @property
    def in_dtype(self) -> str:
        assert self.A.dtype == self.B.dtype, "A and B must have the same dtype"
        return self.A.dtype

    @property
    def accum_dtype(self) -> str:
        return self.C.dtype

    @property
    def chunk(self) -> int:
        return self.A.shape[-2] if self.trans_A else self.A.shape[-1]

    @property
    def A(self) -> tir.Buffer:
        return getattr(self.gemm_node, "a", None)

    @property
    def B(self) -> tir.Buffer:
        return getattr(self.gemm_node, "b", None)

    @property
    def C(self) -> tir.Buffer:
        return getattr(self.gemm_node, "c", None)

    @property
    def ARegion(self):
        return getattr(self.gemm_node, "aRegion", None)

    @property
    def BRegion(self):
        return getattr(self.gemm_node, "bRegion", None)

    @property
    def CRegion(self):
        return getattr(self.gemm_node, "cRegion", None)

    @property
    def stride_A(self) -> int:
        return getattr(self.gemm_node, "strideA", None)

    @property
    def stride_B(self) -> int:
        return getattr(self.gemm_node, "strideB", None)

    @property
    def offset_A(self) -> int:
        return getattr(self.gemm_node, "offsetA", None)

    @property
    def offset_B(self) -> int:
        return getattr(self.gemm_node, "offsetB", None)

    @property
    def clear_accum(self) -> PrimExpr:
        return getattr(self.gemm_node, "clearAccum", None)

    @property
    def k_pack(self) -> int:
        return getattr(self.gemm_node, "kPack", None)

    @property
    def wg_wait(self) -> int:
        return getattr(self.gemm_node, "wgWait", 0)

    @property
    def policy(self) -> GemmWarpPolicy:
        return getattr(self.gemm_node, "policy", None)

    @property
    def mbarptr(self) -> PrimExpr:
        return getattr(self.gemm_node, "mbarPtr", tvm.tir.const(0, T.uint32))

    @property
    def mbar(self) -> tir.Buffer:
        return getattr(self.gemm_node, "mbar", None)

    @property
    def C_coords(self):
        coords = getattr(self.gemm_node, "cCoords", None)
        if coords is None or len(coords) == 0:
            zero = tvm.tir.const(0, T.int32)
            return [zero, zero]
        return [coords[i] for i in range(len(coords))]

    def get_region_base_offsets(self, region):
        """
        Get the base offset (start index) for each dimension from a BufferRegion.

        For example, if region is A_shared[ko % 2, 0:128, 0:64],
        this returns [ko % 2, 0, 0]

        Args:
            region: BufferRegion object

        Returns:
            List of PrimExpr representing the base offset for each dimension
        """
        if region is None:
            return []
        return [r.min for r in region.region]

    @property
    def A_base_offsets(self):
        """Get base offsets for each dimension of A region"""
        return self.get_region_base_offsets(self.ARegion)

    @property
    def B_base_offsets(self):
        """Get base offsets for each dimension of B region"""
        return self.get_region_base_offsets(self.BRegion)

    @property
    def C_base_offsets(self):
        """Get base offsets for each dimension of C region"""
        return self.get_region_base_offsets(self.CRegion)
