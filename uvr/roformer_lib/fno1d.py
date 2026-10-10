import functools
from typing import (
    Tuple,
    List,
    Union,
    Optional,
    Callable,
    Any,
    Literal,
    Dict,
    Sequence,
    Set,
    Generator,
    FrozenSet,
    Collection,
    Iterable,
    Iterator,
    overload,
    Counter as CounterType,
)
from decimal import Decimal
from collections import namedtuple

TensorShapeType = Tuple[int, ...]
PathType = Collection[TensorShapeType]
ArrayType = Any
ArrayIndexType = FrozenSet[str]
ArrayShaped = namedtuple("ArrayShaped", ["shape"])
Number = Union[float, int]
import bisect
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
import inspect
from pathlib import Path
from abc import ABC, abstractmethod
from copy import deepcopy
import itertools
import random
import heapq
import random
import operator
import re
from scipy.optimize import brentq
import math

_MemoryLimit = Union[None, int, Decimal, Literal["max_input"]]
PathSearchFunctionType = Callable[
    [List[ArrayIndexType], ArrayIndexType, Dict[str, int], Optional[int]], PathType
]

_BaseTypes = (bool, int, float, complex, str, bytes)

__all__ = ["compute_size_by_dict", "find_contraction", "flop_count"]

_valid_chars = "abcdefghijklmopqABC"
_sizes = [2, 3, 4, 5, 4, 3, 2, 6, 5, 4, 3, 2, 5, 7, 4, 3, 2, 3, 4]
_default_dim_dict = dict(zip(_valid_chars, _sizes))

_einsum_symbols_base = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

_UNLIMITED_MEM = {-1, None, float("inf")}

from collections import Counter, defaultdict

GreedyCostType = Tuple[int, int, int]
GreedyContractionType = Tuple[
    GreedyCostType, ArrayIndexType, ArrayIndexType, ArrayIndexType
]  # Cost, t1,t2->t3


class PathOptimizer:
    r"""Base class for different path optimizers to inherit from.

    Subclassed optimizers should define a call method with signature:

    ```python
    def __call__(self, inputs: List[ArrayIndexType], output: ArrayIndexType, size_dict: dict[str, int], memory_limit: int | None = None) -> list[tuple[int, ...]]:
        \"\"\"
        Parameters:
            inputs: The indices of each input array.
            outputs: The output indices
            size_dict: The size of each index
            memory_limit: If given, the maximum allowed memory.
        \"\"\"
        # ... compute path here ...
        return path
    ```

    where `path` is a list of int-tuples specifying a contraction order.
    """

    def _check_args_against_first_call(
        self,
        inputs: List[ArrayIndexType],
        output: ArrayIndexType,
        size_dict: Dict[str, int],
    ) -> None:
        """Utility that stateful optimizers can use to ensure they are not
        called with different contractions across separate runs.
        """
        args = (inputs, output, size_dict)
        if not hasattr(self, "_first_call_args"):
            # simply set the attribute as currently there is no global PathOptimizer init
            self._first_call_args = args
        elif args != self._first_call_args:
            raise ValueError(
                "The arguments specifying the contraction that this path optimizer "
                "instance was called with have changed - try creating a new instance."
            )

    def __call__(
        self,
        inputs: List[ArrayIndexType],
        output: ArrayIndexType,
        size_dict: Dict[str, int],
        memory_limit: Optional[int] = None,
    ) -> PathType:
        raise NotImplementedError


def ssa_to_linear(ssa_path: PathType) -> PathType:
    """Convert a path with static single assignment ids to a path with recycled
    linear ids.

    Example:
        ```python
        ssa_to_linear([(0, 3), (2, 4), (1, 5)])
        #> [(0, 3), (1, 2), (0, 1)]
        ```
    """
    # ids = np.arange(1 + max(map(max, ssa_path)), dtype=np.int32)  # type: ignore
    # path = []
    # for ssa_ids in ssa_path:
    #     path.append(tuple(int(ids[ssa_id]) for ssa_id in ssa_ids))
    #     for ssa_id in ssa_ids:
    #         ids[ssa_id:] -= 1
    # return path

    n = sum(map(len, ssa_path)) - len(ssa_path) + 1
    ids = list(range(n))
    path = []
    ssa = n
    for scon in ssa_path:
        con = sorted([bisect.bisect_left(ids, s) for s in scon])
        for j in reversed(con):
            ids.pop(j)
        ids.append(ssa)
        path.append(con)
        ssa += 1
    return [tuple(x) for x in path]

    # N = sum(map(len, ssa_path)) - len(ssa_path) + 1
    # ids = list(range(N))
    # ids = np.arange(1 + max(map(max, ssa_path)), dtype=np.int32)
    # path = []
    # ssa = N
    # for scon in ssa_path:
    #     con = sorted(map(ids.index, scon))
    #     for j in reversed(con):
    #         ids.pop(j)
    #     ids.append(ssa)
    #     path.append(con)
    #     ssa += 1
    # return path


def linear_to_ssa(path: PathType) -> PathType:
    """Convert a path with recycled linear ids to a path with static single
    assignment ids.

    Exmaple:
        ```python
        linear_to_ssa([(0, 3), (1, 2), (0, 1)])
        #> [(0, 3), (2, 4), (1, 5)]
        ```
    """
    num_inputs = sum(map(len, path)) - len(path) + 1
    linear_to_ssa = list(range(num_inputs))
    new_ids = itertools.count(num_inputs)
    ssa_path = []
    for ids in path:
        ssa_path.append(tuple(linear_to_ssa[id_] for id_ in ids))
        for id_ in sorted(ids, reverse=True):
            del linear_to_ssa[id_]
        linear_to_ssa.append(next(new_ids))
    return ssa_path


def better_flops_first(flops: int, size: int, best_flops: int, best_size: int) -> bool:
    return (flops, size) < (best_flops, best_size)


def better_size_first(flops: int, size: int, best_flops: int, best_size: int) -> bool:
    return (size, flops) < (best_size, best_flops)


_BETTER_FNS = {
    "flops": better_flops_first,
    "size": better_size_first,
}


def cost_memory_removed(
    size12: int, size1: int, size2: int, k12: int, k1: int, k2: int
) -> float:
    """The default heuristic cost, corresponding to the total reduction in
    memory of performing a contraction.
    """
    return size12 - size1 - size2


def cost_memory_removed_jitter(
    size12: int, size1: int, size2: int, k12: int, k1: int, k2: int
) -> float:
    """Like memory-removed, but with a slight amount of noise that breaks ties
    and thus jumbles the contractions a bit.
    """
    return random.gauss(1.0, 0.01) * (size12 - size1 - size2)


_COST_FNS = {
    "memory-removed": cost_memory_removed,
    "memory-removed-jitter": cost_memory_removed_jitter,
}


def _tree_to_sequence(tree: Tuple[Any, ...]) -> PathType:
    """Converts a contraction tree to a contraction path as it has to be
    returned by path optimizers. A contraction tree can either be an int
    (=no contraction) or a tuple containing the terms to be contracted. An
    arbitrary number (>= 1) of terms can be contracted at once. Note that
    contractions are commutative, e.g. (j, k, l) = (k, l, j). Note that in
    general, solutions are not unique.

    Parameters:
        c: Contraction tree

    Returns:
        path: Contraction path

    Examples:
        ```python
        _tree_to_sequence(((1,2),(0,(4,5,3))))
        #> [(1, 2), (1, 2, 3), (0, 2), (0, 1)]
        ```
    """
    # ((1,2),(0,(4,5,3))) --> [(1, 2), (1, 2, 3), (0, 2), (0, 1)]
    #
    # 0     0         0           (1,2)       --> ((1,2),(0,(3,4,5)))
    # 1     3         (1,2)   --> (0,(3,4,5))
    # 2 --> 4     --> (3,4,5)
    # 3     5
    # 4     (1,2)
    # 5
    #
    # this function iterates through the table shown above from right to left;

    if type(tree) == int:  # noqa: E721
        return []

    c: List[Tuple[Any, ...]] = [
        tree
    ]  # list of remaining contractions (lower part of columns shown above)
    t: List[int] = []  # list of elementary tensors (upper part of columns)
    s: List[Tuple[int, ...]] = []  # resulting contraction sequence

    while len(c) > 0:
        j = c.pop(-1)
        s.insert(0, ())

        for i in sorted([i for i in j if type(i) == int]):  # noqa: E721
            s[0] += (sum(1 for q in t if q < i),)
            t.insert(s[0][-1], i)

        for i_tup in [i_tup for i_tup in j if type(i_tup) != int]:  # noqa: E721
            s[0] += (len(t) + len(c),)
            c.append(i_tup)

    return s


def _find_disconnected_subgraphs(
    inputs: List[FrozenSet[int]], output: FrozenSet[int]
) -> List[FrozenSet[int]]:
    """Finds disconnected subgraphs in the given list of inputs. Inputs are
    connected if they share summation indices. Note: Disconnected subgraphs
    can be contracted independently before forming outer products.

    Parameters:
        inputs: List of sets that represent the lhs side of the einsum subscript
        output: Set that represents the rhs side of the overall einsum subscript

    Returns:
        subgraphs: List containing sets of indices for each subgraph

    Examples:
        ```python
        _find_disconnected_subgraphs([set("ab"), set("c"), set("ad")], set("bd"))
        #> [{0, 2}, {1}]

        _find_disconnected_subgraphs([set("ab"), set("c"), set("ad")], set("abd"))
        #> [{0}, {1}, {2}]
        ```
    """
    subgraphs = []
    unused_inputs = set(range(len(inputs)))

    i_sum = frozenset.union(*inputs) - output  # all summation indices

    while len(unused_inputs) > 0:
        g = set()
        q = [unused_inputs.pop()]
        while len(q) > 0:
            j = q.pop()
            g.add(j)
            i_tmp = i_sum & inputs[j]
            n = {k for k in unused_inputs if len(i_tmp & inputs[k]) > 0}
            q.extend(n)
            unused_inputs.difference_update(n)

        subgraphs.append(g)

    return [frozenset(x) for x in subgraphs]


def _bitmap_select(
    s: int, seq: List[FrozenSet[int]]
) -> Generator[FrozenSet[int], None, None]:
    """Select elements of ``seq`` which are marked by the bitmap set ``s``.

    E.g.:

        >>> list(_bitmap_select(0b11010, ['A', 'B', 'C', 'D', 'E']))
        ['B', 'D', 'E']
    """
    return (x for x, b in zip(seq, bin(s)[:1:-1]) if b == "1")


def _dp_calc_legs(g, all_tensors, s, inputs, i1_cut_i2_wo_output, i1_union_i2):
    """Calculates the effective outer indices of the intermediate tensor
    corresponding to the subgraph ``s``.
    """
    # set of remaining tensors (=g-s)
    r = g & (all_tensors ^ s)
    # indices of remaining indices:
    if r:
        i_r = frozenset.union(*_bitmap_select(r, inputs))
    else:
        i_r = frozenset()
    # contraction indices:
    i_contract = i1_cut_i2_wo_output - i_r
    return i1_union_i2 - i_contract


def _dp_compare_flops(
    cost1: int,
    cost2: int,
    i1_union_i2: Set[int],
    size_dict: List[int],
    cost_cap: int,
    s1: int,
    s2: int,
    xn: Dict[int, Any],
    g: int,
    all_tensors: int,
    inputs: List[FrozenSet[int]],
    i1_cut_i2_wo_output: Set[int],
    memory_limit: Optional[int],
    contract1: Union[int, Tuple[int]],
    contract2: Union[int, Tuple[int]],
) -> None:
    """Performs the inner comparison of whether the two subgraphs (the bitmaps
    `s1` and `s2`) should be merged and added to the dynamic programming
    search. Will skip for a number of reasons:

    1. If the number of operations to form `s = s1 | s2` including previous
       contractions is above the cost-cap.
    2. If we've already found a better way of making `s`.
    3. If the intermediate tensor corresponding to `s` is going to break the
       memory limit.
    """
    # TODO: Odd usage with an Iterable[int] to map a dict of type List[int]
    cost = cost1 + cost2 + compute_size_by_dict(i1_union_i2, size_dict)
    if cost <= cost_cap:
        s = s1 | s2
        if s not in xn or cost < xn[s][1]:
            i = _dp_calc_legs(
                g, all_tensors, s, inputs, i1_cut_i2_wo_output, i1_union_i2
            )
            mem = compute_size_by_dict(i, size_dict)
            if memory_limit is None or mem <= memory_limit:
                xn[s] = (i, cost, (contract1, contract2))


def _dp_compare_size(
    cost1: int,
    cost2: int,
    i1_union_i2: Set[int],
    size_dict: List[int],
    cost_cap: int,
    s1: int,
    s2: int,
    xn: Dict[int, Any],
    g: int,
    all_tensors: int,
    inputs: List[FrozenSet[int]],
    i1_cut_i2_wo_output: Set[int],
    memory_limit: Optional[int],
    contract1: Union[int, Tuple[int]],
    contract2: Union[int, Tuple[int]],
) -> None:
    """Like `_dp_compare_flops` but sieves the potential contraction based
    on the size of the intermediate tensor created, rather than the number of
    operations, and so calculates that first.
    """
    s = s1 | s2
    i = _dp_calc_legs(g, all_tensors, s, inputs, i1_cut_i2_wo_output, i1_union_i2)
    mem = compute_size_by_dict(i, size_dict)
    cost = max(cost1, cost2, mem)
    if cost <= cost_cap:
        if s not in xn or cost < xn[s][1]:
            if memory_limit is None or mem <= memory_limit:
                xn[s] = (i, cost, (contract1, contract2))


def _dp_compare_write(
    cost1: int,
    cost2: int,
    i1_union_i2: Set[int],
    size_dict: List[int],
    cost_cap: int,
    s1: int,
    s2: int,
    xn: Dict[int, Any],
    g: int,
    all_tensors: int,
    inputs: List[FrozenSet[int]],
    i1_cut_i2_wo_output: Set[int],
    memory_limit: Optional[int],
    contract1: Union[int, Tuple[int]],
    contract2: Union[int, Tuple[int]],
) -> None:
    """Like ``_dp_compare_flops`` but sieves the potential contraction based
    on the total size of memory created, rather than the number of
    operations, and so calculates that first.
    """
    s = s1 | s2
    i = _dp_calc_legs(g, all_tensors, s, inputs, i1_cut_i2_wo_output, i1_union_i2)
    mem = compute_size_by_dict(i, size_dict)
    cost = cost1 + cost2 + mem
    if cost <= cost_cap:
        if s not in xn or cost < xn[s][1]:
            if memory_limit is None or mem <= memory_limit:
                xn[s] = (i, cost, (contract1, contract2))


DEFAULT_COMBO_FACTOR = 64


def _dp_compare_combo(
    cost1: int,
    cost2: int,
    i1_union_i2: Set[int],
    size_dict: List[int],
    cost_cap: int,
    s1: int,
    s2: int,
    xn: Dict[int, Any],
    g: int,
    all_tensors: int,
    inputs: List[FrozenSet[int]],
    i1_cut_i2_wo_output: Set[int],
    memory_limit: Optional[int],
    contract1: Union[int, Tuple[int]],
    contract2: Union[int, Tuple[int]],
    factor: Union[int, float] = DEFAULT_COMBO_FACTOR,
    combine: Callable = sum,
) -> None:
    """Like ``_dp_compare_flops`` but sieves the potential contraction based
    on some combination of both the flops and size,.
    """
    s = s1 | s2
    i = _dp_calc_legs(g, all_tensors, s, inputs, i1_cut_i2_wo_output, i1_union_i2)
    mem = compute_size_by_dict(i, size_dict)
    f = compute_size_by_dict(i1_union_i2, size_dict)
    cost = cost1 + cost2 + combine((f, factor * mem))
    if cost <= cost_cap:
        if s not in xn or cost < xn[s][1]:
            if memory_limit is None or mem <= memory_limit:
                xn[s] = (i, cost, (contract1, contract2))


def get_better_fn(key: str) -> Callable[[int, int, int, int], bool]:
    return _BETTER_FNS[key]


class BranchBound(PathOptimizer):
    def __init__(
        self,
        nbranch: Optional[int] = None,
        cutoff_flops_factor: int = 4,
        minimize: str = "flops",
        cost_fn: str = "memory-removed",
    ):
        """Explores possible pair contractions in a depth-first recursive manner like
        the `optimal` approach, but with extra heuristic early pruning of branches
        as well sieving by `memory_limit` and the best path found so far.


        Parameters:
            nbranch: How many branches to explore at each contraction step. If None, explore
                all possible branches. If an integer, branch into this many paths at
                each step. Defaults to None.
            cutoff_flops_factor: If at any point, a path is doing this much worse than the best path
                found so far was, terminate it. The larger this is made, the more paths
                will be fully explored and the slower the algorithm. Defaults to 4.
            minimize: Whether to optimize the path with regard primarily to the total
                estimated flop-count, or the size of the largest intermediate. The
                option not chosen will still be used as a secondary criterion.
            cost_fn: A function that returns a heuristic 'cost' of a potential contraction
                with which to sort candidates. Should have signature
                `cost_fn(size12, size1, size2, k12, k1, k2)`.
        """
        if (nbranch is not None) and nbranch < 1:
            raise ValueError(
                f"The number of branches must be at least one, `nbranch={nbranch}`."
            )

        self.nbranch = nbranch
        self.cutoff_flops_factor = cutoff_flops_factor
        self.minimize = minimize
        self.cost_fn: Any = _COST_FNS.get(cost_fn, cost_fn)

        self.better = get_better_fn(minimize)
        self.best: Dict[str, Any] = {"flops": float("inf"), "size": float("inf")}
        self.best_progress: Dict[int, float] = defaultdict(lambda: float("inf"))

    @property
    def path(self) -> PathType:
        return ssa_to_linear(self.best["ssa_path"])

    def __call__(
        self,
        inputs_: List[ArrayIndexType],
        output_: ArrayIndexType,
        size_dict: Dict[str, int],
        memory_limit: Optional[int] = None,
    ) -> PathType:
        """Parameters:
            inputs_: List of sets that represent the lhs side of the einsum subscript
            output_: Set that represents the rhs side of the overall einsum subscript
            size_dict: Dictionary of index sizes
            memory_limit: The maximum number of elements in a temporary array.

        Returns:
            path: The contraction order within the memory limit constraint.

        Examples:
        ```python
        isets = [set('abd'), set('ac'), set('bdc')]
        oset = set('')
        idx_sizes = {'a': 1, 'b':2, 'c':3, 'd':4}
        optimal(isets, oset, idx_sizes, 5000)
        #> [(0, 2), (0, 1)]
        """
        self._check_args_against_first_call(inputs_, output_, size_dict)

        inputs: Tuple[FrozenSet[str]] = tuple(map(frozenset, inputs_))  # type: ignore
        output: FrozenSet[str] = frozenset(output_)

        size_cache = {k: compute_size_by_dict(k, size_dict) for k in inputs}
        result_cache: Dict[
            Tuple[FrozenSet[str], FrozenSet[str]], Tuple[FrozenSet[str], int]
        ] = {}

        def _branch_iterate(path, inputs, remaining, flops, size):
            # reached end of path (only ever get here if flops is best found so far)
            if len(remaining) == 1:
                self.best["size"] = size
                self.best["flops"] = flops
                self.best["ssa_path"] = path
                return

            def _assess_candidate(
                k1: FrozenSet[str], k2: FrozenSet[str], i: int, j: int
            ) -> Any:
                # find resulting indices and flops
                try:
                    k12, flops12 = result_cache[k1, k2]
                except KeyError:
                    k12, flops12 = result_cache[k1, k2] = calc_k12_flops(
                        inputs, output, remaining, i, j, size_dict
                    )

                try:
                    size12 = size_cache[k12]
                except KeyError:
                    size12 = size_cache[k12] = compute_size_by_dict(k12, size_dict)

                new_flops = flops + flops12
                new_size = max(size, size12)

                # sieve based on current best i.e. check flops and size still better
                if not self.better(
                    new_flops, new_size, self.best["flops"], self.best["size"]
                ):
                    return None

                # compare to how the best method was doing as this point
                if new_flops < self.best_progress[len(inputs)]:
                    self.best_progress[len(inputs)] = new_flops
                # sieve based on current progress relative to best
                elif (
                    new_flops
                    > self.cutoff_flops_factor * self.best_progress[len(inputs)]
                ):
                    return None

                # sieve based on memory limit
                if (memory_limit not in _UNLIMITED_MEM) and (size12 > memory_limit):  # type: ignore
                    # terminate path here, but check all-terms contract first
                    new_flops = flops + _compute_oversize_flops(
                        inputs, remaining, output_, size_dict
                    )
                    if new_flops < self.best["flops"]:
                        self.best["flops"] = new_flops
                        self.best["ssa_path"] = path + (tuple(remaining),)
                    return None

                # set cost heuristic in order to locally sort possible contractions
                size1, size2 = size_cache[inputs[i]], size_cache[inputs[j]]
                cost = self.cost_fn(size12, size1, size2, k12, k1, k2)

                return cost, flops12, new_flops, new_size, (i, j), k12

            # check all possible remaining paths
            candidates = []
            for i, j in itertools.combinations(remaining, 2):
                if i > j:
                    i, j = j, i
                k1, k2 = inputs[i], inputs[j]

                # initially ignore outer products
                if k1.isdisjoint(k2):
                    continue

                candidate = _assess_candidate(k1, k2, i, j)
                if candidate:
                    heapq.heappush(candidates, candidate)

            # assess outer products if nothing left
            if not candidates:
                for i, j in itertools.combinations(remaining, 2):
                    if i > j:
                        i, j = j, i
                    k1, k2 = inputs[i], inputs[j]
                    candidate = _assess_candidate(k1, k2, i, j)
                    if candidate:
                        heapq.heappush(candidates, candidate)

            # recurse into all or some of the best candidate contractions
            bi = 0
            while (self.nbranch is None or bi < self.nbranch) and candidates:
                _, _, new_flops, new_size, (i, j), k12 = heapq.heappop(candidates)
                _branch_iterate(
                    path=path + ((i, j),),
                    inputs=inputs + (k12,),
                    remaining=(remaining - {i, j}) | {len(inputs)},
                    flops=new_flops,
                    size=new_size,
                )
                bi += 1

        _branch_iterate(
            path=(), inputs=inputs, remaining=set(range(len(inputs))), flops=0, size=0
        )

        return self.path


def branch(
    inputs: List[ArrayIndexType],
    output: ArrayIndexType,
    size_dict: Dict[str, int],
    memory_limit: Optional[int] = None,
    nbranch: Optional[int] = None,
    cutoff_flops_factor: int = 4,
    minimize: str = "flops",
    cost_fn: str = "memory-removed",
) -> PathType:
    optimizer = BranchBound(
        nbranch=nbranch,
        cutoff_flops_factor=cutoff_flops_factor,
        minimize=minimize,
        cost_fn=cost_fn,
    )
    return optimizer(inputs, output, size_dict, memory_limit)


branch_all = functools.partial(branch, nbranch=None)
branch_2 = functools.partial(branch, nbranch=2)
branch_1 = functools.partial(branch, nbranch=1)


def calc_k12_flops(
    inputs: Tuple[FrozenSet[str]],
    output: FrozenSet[str],
    remaining: FrozenSet[int],
    i: int,
    j: int,
    size_dict: Dict[str, int],
) -> Tuple[FrozenSet[str], int]:
    """Calculate the resulting indices and flops for a potential pairwise
    contraction - used in the recursive (optimal/branch) algorithms.

    Parameters:
        inputs: The indices of each tensor in this contraction, note this includes
            tensors unavailable to contract as static single assignment is used:>
            contracted tensors are not removed from the list.
        output: The set of output indices for the whole contraction.
        remaining: *The set of indices (corresponding to ``inputs``) of tensors still available to contract.
        i: Index of potential tensor to contract.
        j: Index of potential tensor to contract.
        size_dict: Size mapping of all the indices.

    Returns:
        k12: The resulting indices of the potential tensor.
        cost: Estimated flop count of operation.
    """
    k1, k2 = inputs[i], inputs[j]
    either = k1 | k2
    shared = k1 & k2
    keep = frozenset.union(output, *map(inputs.__getitem__, remaining - {i, j}))

    k12 = either & keep
    cost = flop_count(either, bool(shared - keep), 2, size_dict)

    return k12, cost


def _compute_oversize_flops(
    inputs: Tuple[FrozenSet[str]],
    remaining: List[ArrayIndexType],
    output: ArrayIndexType,
    size_dict: Dict[str, int],
) -> int:
    """Compute the flop count for a contraction of all remaining arguments. This
    is used when a memory limit means that no pairwise contractions can be made.
    """
    idx_contraction = frozenset.union(*map(inputs.__getitem__, remaining))  # type: ignore
    inner = idx_contraction - output
    num_terms = len(remaining)
    return flop_count(idx_contraction, bool(inner), num_terms, size_dict)


def optimal(
    inputs: List[ArrayIndexType],
    output: ArrayIndexType,
    size_dict: Dict[str, int],
    memory_limit: Optional[int] = None,
) -> PathType:
    """Computes all possible pair contractions in a depth-first recursive manner,
    sieving results based on `memory_limit` and the best path found so far.

    Parameters:
        inputs: List of sets that represent the lhs side of the einsum subscript.
        output: Set that represents the rhs side of the overall einsum subscript.
        size_dict: Dictionary of index sizes.
        memory_limit: The maximum number of elements in a temporary array.

    Returns:
        path: The optimal contraction order within the memory limit constraint.

    Examples:
    ```python
    isets = [set('abd'), set('ac'), set('bdc')]
    oset = set('')
    idx_sizes = {'a': 1, 'b':2, 'c':3, 'd':4}
    optimal(isets, oset, idx_sizes, 5000)
    #> [(0, 2), (0, 1)]
    ```
    """
    inputs_set = tuple(map(frozenset, inputs))
    output_set = frozenset(output)

    best_flops = {"flops": float("inf")}
    best_ssa_path = {"ssa_path": (tuple(range(len(inputs))),)}
    size_cache: Dict[FrozenSet[str], int] = {}
    result_cache: Dict[
        Tuple[ArrayIndexType, ArrayIndexType], Tuple[FrozenSet[str], int]
    ] = {}

    def _optimal_iterate(path, remaining, inputs, flops):
        # reached end of path (only ever get here if flops is best found so far)
        if len(remaining) == 1:
            best_flops["flops"] = flops
            best_ssa_path["ssa_path"] = path
            return

        # check all possible remaining paths
        for i, j in itertools.combinations(remaining, 2):
            if i > j:
                i, j = j, i
            key = (inputs[i], inputs[j])
            try:
                k12, flops12 = result_cache[key]
            except KeyError:
                k12, flops12 = result_cache[key] = calc_k12_flops(
                    inputs, output_set, remaining, i, j, size_dict
                )

            # sieve based on current best flops
            new_flops = flops + flops12
            if new_flops >= best_flops["flops"]:
                continue

            # sieve based on memory limit
            if memory_limit not in _UNLIMITED_MEM:
                try:
                    size12 = size_cache[k12]
                except KeyError:
                    size12 = size_cache[k12] = compute_size_by_dict(k12, size_dict)

                # possibly terminate this path with an all-terms einsum
                if size12 > memory_limit:
                    new_flops = flops + _compute_oversize_flops(
                        inputs, remaining, output_set, size_dict
                    )
                    if new_flops < best_flops["flops"]:
                        best_flops["flops"] = new_flops
                        best_ssa_path["ssa_path"] = path + (tuple(remaining),)
                    continue

            # add contraction and recurse into all remaining
            _optimal_iterate(
                path=path + ((i, j),),
                inputs=inputs + (k12,),
                remaining=remaining - {i, j} | {len(inputs)},
                flops=new_flops,
            )

    _optimal_iterate(
        path=(), inputs=inputs_set, remaining=set(range(len(inputs))), flops=0
    )

    return ssa_to_linear(best_ssa_path["ssa_path"])


minimize_finder = re.compile(r"(flops|size|write|combo|limit)-*(\d*)")


@functools.lru_cache(128)
def _parse_minimize(
    minimize: Union[str, Callable],
) -> Tuple[Callable, Union[int, float]]:
    """This works out what local scoring function to use for the dp algorithm
    as well as a `naive_scale` to account for the memory_limit checks.
    """
    if minimize == "flops":
        return _dp_compare_flops, 1
    elif minimize == "size":
        return _dp_compare_size, 1
    elif minimize == "write":
        return _dp_compare_write, 1
    elif callable(minimize):
        # default to naive_scale=inf for this and remaining options
        # as otherwise memory_limit check can cause problems
        return minimize, float("inf")

    # parse out a customized value for the combination factor
    match = minimize_finder.fullmatch(minimize)
    if match is None:
        raise ValueError(f"Couldn't parse `minimize` value: {minimize}.")

    minimize, custom_factor = match.groups()
    factor = float(custom_factor) if custom_factor else DEFAULT_COMBO_FACTOR
    if minimize == "combo":
        return functools.partial(_dp_compare_combo, factor=factor, combine=sum), float(
            "inf"
        )
    elif minimize == "limit":
        return functools.partial(_dp_compare_combo, factor=factor, combine=max), float(
            "inf"
        )
    else:
        raise ValueError(f"Couldn't parse `minimize` value: {minimize}.")


def simple_tree_tuple(seq: Sequence[Tuple[int, ...]]) -> Tuple[Any, ...]:
    """Make a simple left to right binary tree out of iterable `seq`.

    ```python
    tuple_nest([1, 2, 3, 4])
    #> (((1, 2), 3), 4)
    ```

    """
    return functools.reduce(lambda x, y: (x, y), seq)


def _dp_parse_out_single_term_ops(
    inputs: List[FrozenSet[int]],
    all_inds: Tuple[str, ...],
    ind_counts: CounterType[str],
) -> Tuple[List[FrozenSet[int]], List[Tuple[int]], List[Union[int, Tuple[int]]]]:
    """Take `inputs` and parse for single term index operations, i.e. where
    an index appears on one tensor and nowhere else.

    If a term is completely reduced to a scalar in this way it can be removed
    to `inputs_done`. If only some indices can be summed then add a 'single
    term contraction' that will perform this summation.
    """
    i_single = frozenset(i for i, c in enumerate(all_inds) if ind_counts[c] == 1)
    inputs_parsed: List[FrozenSet[int]] = []
    inputs_done: List[Tuple[int]] = []
    inputs_contractions: List[Union[int, Tuple[int]]] = []
    for j, i in enumerate(inputs):
        i_reduced = i - i_single
        if (not i_reduced) and (len(i) > 0):
            # input reduced to scalar already - remove
            inputs_done.append((j,))
        else:
            # if the input has any index reductions, add single contraction
            inputs_parsed.append(i_reduced)
            inputs_contractions.append((j,) if i_reduced != i else j)

    return inputs_parsed, inputs_done, inputs_contractions


class DynamicProgramming(PathOptimizer):
    """Finds the optimal path of pairwise contractions without intermediate outer
    products based a dynamic programming approach presented in
    Phys. Rev. E 90, 033315 (2014) (the corresponding preprint is publicly
    available at https://arxiv.org/abs/1304.6112). This method is especially
    well-suited in the area of tensor network states, where it usually
    outperforms all the other optimization strategies.

    This algorithm shows exponential scaling with the number of inputs
    in the worst case scenario (see example below). If the graph to be
    contracted consists of disconnected subgraphs, the algorithm scales
    linearly in the number of disconnected subgraphs and only exponentially
    with the number of inputs per subgraph.

    Parameters:
        minimize: What to minimize:
            - 'flops' - minimize the number of flops
            - 'size' - minimize the size of the largest intermediate
            - 'write' - minimize the size of all intermediate tensors
            - 'combo' - minimize `flops + alpha * write` summed over intermediates, a default ratio of alpha=64
            is used, or it can be customized with `f'combo-{alpha}'`
            - 'limit' - minimize `max(flops, alpha * write)` summed over intermediates, a default ratio of alpha=64
            is used, or it can be customized with `f'limit-{alpha}'`
            - callable - a custom local cost function

        cost_cap: How to implement cost-capping:
            - True - iteratively increase the cost-cap
            - False - implement no cost-cap at all
            - int - use explicit cost cap

        search_outer: In rare circumstances the optimal contraction may involve an outer
            product, this option allows searching such contractions but may well
            slow down the path finding considerably on all but very small graphs.
    """

    def __init__(
        self,
        minimize: str = "flops",
        cost_cap: Union[bool, int] = True,
        search_outer: bool = False,
    ) -> None:
        self.minimize = minimize
        self.search_outer = search_outer
        self.cost_cap = cost_cap

    def __call__(
        self,
        inputs_: List[ArrayIndexType],
        output_: ArrayIndexType,
        size_dict_: Dict[str, int],
        memory_limit_: Optional[int] = None,
    ) -> PathType:
        """Parameters:
            inputs_: List of sets that represent the lhs side of the einsum subscript
            output_: Set that represents the rhs side of the overall einsum subscript
            size_dict_: Dictionary of index sizes
            memory_limit_: The maximum number of elements in a temporary array.

        Returns:
            path: The contraction order (a list of tuples of ints).

        Examples:
            ```python
            n_in = 3  # exponential scaling
            n_out = 2 # linear scaling
            s = dict()
            i_all = []
            for _ in range(n_out):
                i = [set() for _ in range(n_in)]
                for j in range(n_in):
                    for k in range(j+1, n_in):
                        c = oe.get_symbol(len(s))
                        i[j].add(c)
                        i[k].add(c)
                        s[c] = 2
                i_all.extend(i)
            o = DynamicProgramming()
            o(i_all, set(), s)
            #> [(1, 2), (0, 4), (1, 2), (0, 2), (0, 1)]
            ```
        """
        _check_contraction, naive_scale = _parse_minimize(self.minimize)
        _check_outer = (lambda x: True) if self.search_outer else (lambda x: x)

        ind_counts = Counter(itertools.chain(*inputs_, output_))
        all_inds = tuple(ind_counts)

        # convert all indices to integers (makes set operations ~10 % faster)
        symbol2int = {c: j for j, c in enumerate(all_inds)}
        inputs = [frozenset(symbol2int[c] for c in i) for i in inputs_]
        output = frozenset(symbol2int[c] for c in output_)
        size_dict_canonical = {
            symbol2int[c]: v for c, v in size_dict_.items() if c in symbol2int
        }
        size_dict = [size_dict_canonical[j] for j in range(len(size_dict_canonical))]
        naive_cost = (
            naive_scale * len(inputs) * functools.reduce(operator.mul, size_dict, 1)
        )

        inputs, inputs_done, inputs_contractions = _dp_parse_out_single_term_ops(
            inputs, all_inds, ind_counts
        )

        if not inputs:
            # nothing left to do after single axis reductions!
            return _tree_to_sequence(simple_tree_tuple(inputs_done))

        # a list of all necessary contraction expressions for each of the
        # disconnected subgraphs and their size
        subgraph_contractions = inputs_done
        subgraph_contractions_size = [1] * len(inputs_done)

        if self.search_outer:
            # optimize everything together if we are considering outer products
            subgraphs = [frozenset(range(len(inputs)))]
        else:
            subgraphs = _find_disconnected_subgraphs(inputs, output)

        # the bitmap set of all tensors is computed as it is needed to
        # compute set differences: s1 - s2 transforms into
        # s1 & (all_tensors ^ s2)
        all_tensors = (1 << len(inputs)) - 1

        for g in subgraphs:
            # dynamic programming approach to compute x[n] for subgraph g;
            # x[n][set of n tensors] = (indices, cost, contraction)
            # the set of n tensors is represented by a bitmap: if bit j is 1,
            # tensor j is in the set, e.g. 0b100101 = {0,2,5}; set unions
            # (intersections) can then be computed by bitwise or (and);
            x: List[Any] = [None] * 2 + [{} for j in range(len(g) - 1)]
            x[1] = {1 << j: (inputs[j], 0, inputs_contractions[j]) for j in g}

            # convert set of tensors g to a bitmap set:
            bitmap_g = functools.reduce(lambda x, y: x | y, (1 << j for j in g))

            # try to find contraction with cost <= cost_cap and increase
            # cost_cap successively if no such contraction is found;
            # this is a major performance improvement; start with product of
            # output index dimensions as initial cost_cap
            subgraph_inds = frozenset.union(*_bitmap_select(bitmap_g, inputs))
            if self.cost_cap is True:
                cost_cap = compute_size_by_dict(subgraph_inds & output, size_dict)
            elif self.cost_cap is False:
                cost_cap = float("inf")  # type: ignore
            else:
                cost_cap = self.cost_cap
            # set the factor to increase the cost by each iteration (ensure > 1)
            if len(subgraph_inds) == 0:
                cost_increment = 2
            else:
                cost_increment = max(min(map(size_dict.__getitem__, subgraph_inds)), 2)

            while len(x[-1]) == 0:
                for n in range(2, len(x[1]) + 1):
                    xn = x[n]

                    # try to combine solutions from x[m] and x[n-m]
                    for m in range(1, n // 2 + 1):
                        for s1, (i1, cost1, contract1) in x[m].items():
                            for s2, (i2, cost2, contract2) in x[n - m].items():
                                # can only merge if s1 and s2 are disjoint
                                # and avoid e.g. s1={0}, s2={1} and s1={1}, s2={0}
                                if (not s1 & s2) and (m != n - m or s1 < s2):
                                    i1_cut_i2_wo_output = (i1 & i2) - output

                                    # maybe ignore outer products:
                                    if _check_outer(i1_cut_i2_wo_output):
                                        i1_union_i2 = i1 | i2
                                        _check_contraction(
                                            cost1,
                                            cost2,
                                            i1_union_i2,
                                            size_dict,
                                            cost_cap,
                                            s1,
                                            s2,
                                            xn,
                                            bitmap_g,
                                            all_tensors,
                                            inputs,
                                            i1_cut_i2_wo_output,
                                            memory_limit_,
                                            contract1,
                                            contract2,
                                        )

                if (cost_cap > naive_cost) and (len(x[-1]) == 0):
                    raise RuntimeError("No contraction found for given `memory_limit`.")

                # increase cost cap for next iteration:
                cost_cap = cost_increment * cost_cap

            i, cost, contraction = list(x[-1].values())[0]
            subgraph_contractions.append(contraction)
            subgraph_contractions_size.append(compute_size_by_dict(i, size_dict))

        # sort the subgraph contractions by the size of the subgraphs in
        # ascending order (will give the cheapest contractions); note that
        # outer products should be performed pairwise (to use BLAS functions)
        subgraph_contractions = [
            subgraph_contractions[j]
            for j in sorted(
                range(len(subgraph_contractions_size)),
                key=subgraph_contractions_size.__getitem__,
            )
        ]

        # build the final contraction tree
        tree = simple_tree_tuple(subgraph_contractions)
        return _tree_to_sequence(tree)


def dynamic_programming(
    inputs: List[ArrayIndexType],
    output: ArrayIndexType,
    size_dict: Dict[str, int],
    memory_limit: Optional[int] = None,
    **kwargs: Any,
) -> PathType:
    optimizer = DynamicProgramming(**kwargs)
    return optimizer(inputs, output, size_dict, memory_limit)


_AUTO_CHOICES = {}
for i in range(1, 5):
    _AUTO_CHOICES[i] = optimal
for i in range(5, 7):
    _AUTO_CHOICES[i] = branch_all
for i in range(7, 9):
    _AUTO_CHOICES[i] = branch_2
for i in range(9, 15):
    _AUTO_CHOICES[i] = branch_1


def auto(
    inputs: List[ArrayIndexType],
    output: ArrayIndexType,
    size_dict: Dict[str, int],
    memory_limit: Optional[int] = None,
) -> PathType:
    """Finds the contraction path by automatically choosing the method based on
    how many input arguments there are.
    """
    return _AUTO_CHOICES.get(len(inputs), greedy)(
        inputs, output, size_dict, memory_limit
    )


_AUTO_HQ_CHOICES = {}
for i in range(1, 6):
    _AUTO_HQ_CHOICES[i] = optimal
for i in range(6, 17):
    _AUTO_HQ_CHOICES[i] = dynamic_programming


def auto_hq(
    inputs: List[ArrayIndexType],
    output: ArrayIndexType,
    size_dict: Dict[str, int],
    memory_limit: Optional[int] = None,
) -> PathType:
    """Finds the contraction path by automatically choosing the method based on
    how many input arguments there are, but targeting a more generous
    amount of search time than ``'auto'``.
    """
    from opt_einsum.path_random import random_greedy_128

    return _AUTO_HQ_CHOICES.get(len(inputs), random_greedy_128)(
        inputs, output, size_dict, memory_limit
    )


def _get_candidate(
    output: ArrayIndexType,
    sizes: Dict[str, int],
    remaining: Dict[ArrayIndexType, int],
    footprints: Dict[ArrayIndexType, int],
    dim_ref_counts: Dict[int, Set[str]],
    k1: ArrayIndexType,
    k2: ArrayIndexType,
    cost_fn: Any,
) -> GreedyContractionType:
    either = k1 | k2
    two = k1 & k2
    one = either - two
    k12 = (either & output) | (two & dim_ref_counts[3]) | (one & dim_ref_counts[2])
    cost = cost_fn(
        compute_size_by_dict(k12, sizes),
        footprints[k1],
        footprints[k2],
        k12,
        k1,
        k2,
    )
    id1 = remaining[k1]
    id2 = remaining[k2]
    if id1 > id2:
        k1, id1, k2, id2 = k2, id2, k1, id1
    cost = cost, id2, id1  # break ties to ensure determinism
    return cost, k1, k2, k12


def _push_candidate(
    output: ArrayIndexType,
    sizes: Dict[str, Any],
    remaining: Dict[ArrayIndexType, int],
    footprints: Dict[ArrayIndexType, int],
    dim_ref_counts: Dict[int, Set[str]],
    k1: ArrayIndexType,
    k2s: List[ArrayIndexType],
    queue: List[GreedyContractionType],
    push_all: bool,
    cost_fn: Any,
) -> None:
    candidates = (
        _get_candidate(
            output, sizes, remaining, footprints, dim_ref_counts, k1, k2, cost_fn
        )
        for k2 in k2s
    )
    if push_all:
        # want to do this if we e.g. are using a custom 'choose_fn'
        for candidate in candidates:
            heapq.heappush(queue, candidate)
    else:
        heapq.heappush(queue, min(candidates))


def _update_ref_counts(
    dim_to_keys: Dict[str, Set[ArrayIndexType]],
    dim_ref_counts: Dict[int, Set[str]],
    dims: ArrayIndexType,
) -> None:
    for dim in dims:
        count = len(dim_to_keys[dim])
        if count <= 1:
            dim_ref_counts[2].discard(dim)
            dim_ref_counts[3].discard(dim)
        elif count == 2:
            dim_ref_counts[2].add(dim)
            dim_ref_counts[3].discard(dim)
        else:
            dim_ref_counts[2].add(dim)
            dim_ref_counts[3].add(dim)


def _simple_chooser(queue, remaining):
    """Default contraction chooser that simply takes the minimum cost option."""
    cost, k1, k2, k12 = heapq.heappop(queue)
    if k1 not in remaining or k2 not in remaining:
        return None  # candidate is obsolete
    return cost, k1, k2, k12


def ssa_greedy_optimize(
    inputs: List[ArrayIndexType],
    output: ArrayIndexType,
    sizes: Dict[str, int],
    choose_fn: Any = None,
    cost_fn: Any = "memory-removed",
) -> PathType:
    """This is the core function for :func:`greedy` but produces a path with
    static single assignment ids rather than recycled linear ids.
    SSA ids are cheaper to work with and easier to reason about.
    """
    if len(inputs) == 1:
        # Perform a single contraction to match output shape.
        return [(0,)]

    # set the function that assigns a heuristic cost to a possible contraction
    cost_fn = _COST_FNS.get(cost_fn, cost_fn)

    # set the function that chooses which contraction to take
    if choose_fn is None:
        choose_fn = _simple_chooser
        push_all = False
    else:
        # assume chooser wants access to all possible contractions
        push_all = True

    # A dim that is common to all tensors might as well be an output dim, since it
    # cannot be contracted until the final step. This avoids an expensive all-pairs
    # comparison to search for possible contractions at each step, leading to speedup
    # in many practical problems where all tensors share a common batch dimension.
    fs_inputs = [frozenset(x) for x in inputs]
    output = frozenset(output) | frozenset.intersection(*fs_inputs)

    # Deduplicate shapes by eagerly computing Hadamard products.
    remaining: Dict[ArrayIndexType, int] = {}  # key -> ssa_id
    ssa_ids = itertools.count(len(fs_inputs))
    ssa_path: List[TensorShapeType] = []
    for ssa_id, key in enumerate(fs_inputs):
        if key in remaining:
            ssa_path.append((remaining[key], ssa_id))
            remaining[key] = next(ssa_ids)
        else:
            remaining[key] = ssa_id

    # Keep track of possible contraction dims.
    dim_to_keys = defaultdict(set)
    for key in remaining:
        for dim in key - output:
            dim_to_keys[dim].add(key)

    # Keep track of the number of tensors using each dim; when the dim is no longer
    # used it can be contracted. Since we specialize to binary ops, we only care about
    # ref counts of >=2 or >=3.
    dim_ref_counts = {
        count: {dim for dim, keys in dim_to_keys.items() if len(keys) >= count} - output
        for count in [2, 3]
    }

    # Compute separable part of the objective function for contractions.
    footprints = {key: compute_size_by_dict(key, sizes) for key in remaining}

    # Find initial candidate contractions.
    queue: List[GreedyContractionType] = []
    for dim, dim_keys in dim_to_keys.items():
        dim_keys_list = sorted(dim_keys, key=remaining.__getitem__)
        for i, k1 in enumerate(dim_keys_list[:-1]):
            k2s_guess = dim_keys_list[1 + i :]
            _push_candidate(
                output,
                sizes,
                remaining,
                footprints,
                dim_ref_counts,
                k1,
                k2s_guess,
                queue,
                push_all,
                cost_fn,
            )

    # Greedily contract pairs of tensors.
    while queue:
        con = choose_fn(queue, remaining)
        if con is None:
            continue  # allow choose_fn to flag all candidates obsolete
        cost, k1, k2, k12 = con

        ssa_id1 = remaining.pop(k1)
        ssa_id2 = remaining.pop(k2)
        for dim in k1 - output:
            dim_to_keys[dim].remove(k1)
        for dim in k2 - output:
            dim_to_keys[dim].remove(k2)
        ssa_path.append((ssa_id1, ssa_id2))
        if k12 in remaining:
            ssa_path.append((remaining[k12], next(ssa_ids)))
        else:
            for dim in k12 - output:
                dim_to_keys[dim].add(k12)
        remaining[k12] = next(ssa_ids)
        _update_ref_counts(dim_to_keys, dim_ref_counts, k1 | k2 - output)
        footprints[k12] = compute_size_by_dict(k12, sizes)

        # Find new candidate contractions.
        k1 = k12
        k2s = {k2 for dim in k1 for k2 in dim_to_keys[dim]}
        k2s.discard(k1)
        if k2s:
            _push_candidate(
                output,
                sizes,
                remaining,
                footprints,
                dim_ref_counts,
                k1,
                list(k2s),
                queue,
                push_all,
                cost_fn,
            )

    # Greedily compute pairwise outer products.
    final_queue = [
        (compute_size_by_dict(key & output, sizes), ssa_id, key)
        for key, ssa_id in remaining.items()
    ]
    heapq.heapify(final_queue)
    _, ssa_id1, k1 = heapq.heappop(final_queue)
    while final_queue:
        _, ssa_id2, k2 = heapq.heappop(final_queue)
        ssa_path.append((min(ssa_id1, ssa_id2), max(ssa_id1, ssa_id2)))
        k12 = (k1 | k2) & output
        cost = compute_size_by_dict(k12, sizes)
        ssa_id12 = next(ssa_ids)
        _, ssa_id1, k1 = heapq.heappushpop(final_queue, (cost, ssa_id12, k12))

    return ssa_path


def greedy(
    inputs: List[ArrayIndexType],
    output: ArrayIndexType,
    size_dict: Dict[str, int],
    memory_limit: Optional[int] = None,
    choose_fn: Any = None,
    cost_fn: str = "memory-removed",
) -> PathType:
    """Finds the path by a three stage algorithm:

    1. Eagerly compute Hadamard products.
    2. Greedily compute contractions to maximize `removed_size`
    3. Greedily compute outer products.

    This algorithm scales quadratically with respect to the
    maximum number of elements sharing a common dim.

    Parameters:
        inputs: List of sets that represent the lhs side of the einsum subscript
        output: Set that represents the rhs side of the overall einsum subscript
        size_dict: Dictionary of index sizes
        memory_limit: The maximum number of elements in a temporary array
        choose_fn: A function that chooses which contraction to perform from the queue
        cost_fn: A function that assigns a potential contraction a cost.

    Returns:
        path: The contraction order (a list of tuples of ints).

    Examples:
        ```python
        isets = [set('abd'), set('ac'), set('bdc')]
        oset = set('')
        idx_sizes = {'a': 1, 'b':2, 'c':3, 'd':4}
        greedy(isets, oset, idx_sizes)
        #> [(0, 2), (0, 1)]
        ```
    """
    if memory_limit not in _UNLIMITED_MEM:
        return branch(inputs, output, size_dict, memory_limit, nbranch=1, cost_fn=cost_fn)  # type: ignore

    ssa_path = ssa_greedy_optimize(
        inputs, output, size_dict, cost_fn=cost_fn, choose_fn=choose_fn
    )
    return ssa_to_linear(ssa_path)


_PATH_OPTIONS: Dict[str, PathSearchFunctionType] = {
    "auto": auto,
    "auto-hq": auto_hq,
    "optimal": optimal,
    "branch-all": branch_all,
    "branch-2": branch_2,
    "branch-1": branch_1,
    "greedy": greedy,
    "eager": greedy,
    "opportunistic": greedy,
    "dp": dynamic_programming,
    "dynamic-programming": dynamic_programming,
}


def get_path_fn(path_type: str) -> PathSearchFunctionType:
    """Get the correct path finding function from str ``path_type``."""
    path_type = path_type.lower()
    if path_type not in _PATH_OPTIONS:
        raise KeyError(
            f"Path optimizer '{path_type}' not found, valid options are {set(_PATH_OPTIONS.keys())}."
        )

    return _PATH_OPTIONS[path_type]


@overload
def compute_size_by_dict(indices: Iterable[int], idx_dict: List[int]) -> int: ...


@overload
def compute_size_by_dict(indices: Collection[str], idx_dict: Dict[str, int]) -> int: ...


def compute_size_by_dict(indices: Any, idx_dict: Any) -> int:
    """Computes the product of the elements in indices based on the dictionary
    idx_dict.

    Parameters
    ----------
    indices : iterable
        Indices to base the product on.
    idx_dict : dictionary
        Dictionary of index _sizes

    Returns:
    -------
    ret : int
        The resulting product.

    Examples:
    --------
    >>> compute_size_by_dict('abbc', {'a': 2, 'b':3, 'c':5})
    90

    """
    ret = 1
    for i in indices:  # lgtm [py/iteration-string-and-sequence]
        ret *= idx_dict[i]
    return ret


def find_contraction(
    positions: Collection[int],
    input_sets: List[ArrayIndexType],
    output_set: ArrayIndexType,
) -> Tuple[FrozenSet[str], List[ArrayIndexType], ArrayIndexType, ArrayIndexType]:
    """Finds the contraction for a given set of input and output sets.

    Parameters
    ----------
    positions : iterable
        Integer positions of terms used in the contraction.
    input_sets : list
        List of sets that represent the lhs side of the einsum subscript
    output_set : set
        Set that represents the rhs side of the overall einsum subscript

    Returns:
    -------
    new_result : set
        The indices of the resulting contraction
    remaining : list
        List of sets that have not been contracted, the new set is appended to
        the end of this list
    idx_removed : set
        Indices removed from the entire contraction
    idx_contraction : set
        The indices used in the current contraction

    Examples:
    --------
    # A simple dot product test case
    >>> pos = (0, 1)
    >>> isets = [set('ab'), set('bc')]
    >>> oset = set('ac')
    >>> find_contraction(pos, isets, oset)
    ({'a', 'c'}, [{'a', 'c'}], {'b'}, {'a', 'b', 'c'})

    # A more complex case with additional terms in the contraction
    >>> pos = (0, 2)
    >>> isets = [set('abd'), set('ac'), set('bdc')]
    >>> oset = set('ac')
    >>> find_contraction(pos, isets, oset)
    ({'a', 'c'}, [{'a', 'c'}, {'a', 'c'}], {'b', 'd'}, {'a', 'b', 'c', 'd'})
    """
    remaining = list(input_sets)
    inputs = (remaining.pop(i) for i in sorted(positions, reverse=True))
    idx_contract = frozenset.union(*inputs)
    idx_remain = output_set.union(*remaining)

    new_result = idx_remain & idx_contract
    idx_removed = idx_contract - new_result
    remaining.append(new_result)

    return new_result, remaining, idx_removed, idx_contract


def flop_count(
    idx_contraction: Collection[str],
    inner: bool,
    num_terms: int,
    size_dictionary: Dict[str, int],
) -> int:
    """Computes the number of FLOPS in the contraction.

    Parameters
    ----------
    idx_contraction : iterable
        The indices involved in the contraction
    inner : bool
        Does this contraction require an inner product?
    num_terms : int
        The number of terms in a contraction
    size_dictionary : dict
        The size of each of the indices in idx_contraction

    Returns:
    -------
    flop_count : int
        The total number of FLOPS required for the contraction.

    Examples:
    --------
    >>> flop_count('abc', False, 1, {'a': 2, 'b':3, 'c':5})
    30

    >>> flop_count('abc', True, 2, {'a': 2, 'b':3, 'c':5})
    60

    """
    overall_size = compute_size_by_dict(idx_contraction, size_dictionary)
    op_factor = max(1, num_terms - 1)
    if inner:
        op_factor += 1

    return overall_size * op_factor


def convert_subscripts(old_sub: List[Any], symbol_map: Dict[Any, Any]) -> str:
    """Convert user custom subscripts list to subscript string according to `symbol_map`.

    Examples:
    --------
    >>>  oe.parser.convert_subscripts(['abc', 'def'], {'abc':'a', 'def':'b'})
    'ab'
    >>> oe.parser.convert_subscripts([Ellipsis, object], {object:'a'})
    '...a'
    """
    new_sub = ""
    for s in old_sub:
        if s is Ellipsis:
            new_sub += "..."
        else:
            # no need to try/except here because symbol_map has already been checked
            new_sub += symbol_map[s]
    return new_sub


def get_symbol(i: int) -> str:
    """Get the symbol corresponding to int ``i`` - runs through the usual 52
    letters before resorting to unicode characters, starting at ``chr(192)`` and skipping surrogates.

    **Examples:**

    ```python
    get_symbol(2)
    #> 'c'

    get_symbol(200)
    #> 'Ŕ'

    get_symbol(20000)
    #> '京'
    ```
    """
    if i < 52:
        return _einsum_symbols_base[i]
    elif i >= 55296:
        # Skip chr(57343) - chr(55296) as surrogates
        return chr(i + 2048)
    else:
        return chr(i + 140)


def find_output_shape(
    inputs: List[str], shapes: List[TensorShapeType], output: str
) -> TensorShapeType:
    """Find the output shape for given inputs, shapes and output string, taking
    into account broadcasting.

    Examples:
    --------
    >>> oe.parser.find_output_shape(["ab", "bc"], [(2, 3), (3, 4)], "ac")
    (2, 4)

    # Broadcasting is accounted for
    >>> oe.parser.find_output_shape(["a", "a"], [(4, ), (1, )], "a")
    (4,)
    """
    return tuple(
        max(
            shape[loc]
            for shape, loc in zip(shapes, [x.find(c) for x in inputs])
            if loc >= 0
        )
        for c in output
    )


def gen_unused_symbols(used: str, n: int) -> Iterator[str]:
    """Generate ``n`` symbols that are not already in ``used``.

    **Examples:**
    ```python
    list(oe.parser.gen_unused_symbols("abd", 2))
    #> ['c', 'e']
    ```
    """
    i = cnt = 0
    while cnt < n:
        s = get_symbol(i)
        i += 1
        if s in used:
            continue
        yield s
        cnt += 1


def find_output_str(subscripts: str) -> str:
    """Find the output string for the inputs ``subscripts`` under canonical einstein summation rules.
    That is, repeated indices are summed over by default.

    Examples:
    --------
    >>> oe.parser.find_output_str("ab,bc")
    'ac'

    >>> oe.parser.find_output_str("a,b")
    'ab'

    >>> oe.parser.find_output_str("a,a,b,b")
    ''
    """
    tmp_subscripts = subscripts.replace(",", "")
    return "".join(
        s for s in sorted(set(tmp_subscripts)) if tmp_subscripts.count(s) == 1
    )


def convert_interleaved_input(operands: Sequence[Any]) -> Tuple[str, Tuple[Any, ...]]:
    """Convert 'interleaved' input to standard einsum input."""
    tmp_operands = list(operands)
    operand_list = []
    subscript_list = []
    for _ in range(len(operands) // 2):
        operand_list.append(tmp_operands.pop(0))
        subscript_list.append(tmp_operands.pop(0))

    output_list = tmp_operands[-1] if len(tmp_operands) else None

    # build a map from user symbols to single-character symbols based on `get_symbol`
    # The map retains the intrinsic order of user symbols
    try:
        # collect all user symbols
        symbol_set = set(itertools.chain.from_iterable(subscript_list))

        # remove Ellipsis because it can not be compared with other objects
        symbol_set.discard(Ellipsis)

        # build the map based on sorted user symbols, retaining the order we lost in the `set`
        symbol_map = {
            symbol: get_symbol(idx) for idx, symbol in enumerate(sorted(symbol_set))
        }

    except TypeError:  # unhashable or uncomparable object
        raise TypeError(
            "For this input type lists must contain either Ellipsis "
            "or hashable and comparable object (e.g. int, str)."
        )

    subscripts = ",".join(convert_subscripts(sub, symbol_map) for sub in subscript_list)
    if output_list is not None:
        subscripts += "->"
        subscripts += convert_subscripts(output_list, symbol_map)

    return subscripts, tuple(operand_list)


def parse_einsum_input(
    operands: Any, shapes: bool = False
) -> Tuple[str, str, List[ArrayType]]:
    """A reproduction of einsum c side einsum parsing in python.

    Parameters:
        operands: Intakes the same inputs as `contract_path`, but NOT the keyword args. The only
            supported keyword argument is:
        shapes: Whether ``parse_einsum_input`` should assume arrays (the default) or
            array shapes have been supplied.

    Returns:
        input_strings: Parsed input strings
        output_string: Parsed output string
        operands: The operands to use in the numpy contraction

    Examples:
        The operand list is simplified to reduce printing:

        ```python
        >>> a = np.random.rand(4, 4)
        >>> b = np.random.rand(4, 4, 4)
        >>> parse_einsum_input(('...a,...a->...', a, b))
        ('za,xza', 'xz', [a, b])

        >>> parse_einsum_input((a, [Ellipsis, 0], b, [Ellipsis, 0]))
        ('za,xza', 'xz', [a, b])
        ```
    """
    if len(operands) == 0:
        raise ValueError("No input operands")

    if isinstance(operands[0], str):
        subscripts = operands[0].replace(" ", "")
        if shapes:
            if any(hasattr(o, "shape") for o in operands[1:]):
                raise ValueError(
                    "shapes is set to True but given at least one operand looks like an array"
                    " (at least one operand has a shape attribute). "
                )
        operands = operands[1:]
    else:
        subscripts, operands = convert_interleaved_input(operands)

    if shapes:
        operand_shapes = operands
    else:
        operand_shapes = [get_shape(o) for o in operands]

    # Check for proper "->"
    if ("-" in subscripts) or (">" in subscripts):
        invalid = (subscripts.count("-") > 1) or (subscripts.count(">") > 1)
        if invalid or (subscripts.count("->") != 1):
            raise ValueError("Subscripts can only contain one '->'.")

    # Parse ellipses
    if "." in subscripts:
        used = subscripts.replace(".", "").replace(",", "").replace("->", "")
        ellipse_inds = "".join(
            gen_unused_symbols(used, max(len(x) for x in operand_shapes))
        )
        longest = 0

        # Do we have an output to account for?
        if "->" in subscripts:
            input_tmp, output_sub = subscripts.split("->")
            split_subscripts = input_tmp.split(",")
            out_sub = True
        else:
            split_subscripts = subscripts.split(",")
            out_sub = False

        for num, sub in enumerate(split_subscripts):
            if "." in sub:
                if (sub.count(".") != 3) or (sub.count("...") != 1):
                    raise ValueError("Invalid Ellipses.")

                # Take into account numerical values
                if operand_shapes[num] == ():
                    ellipse_count = 0
                else:
                    ellipse_count = max(len(operand_shapes[num]), 1) - (len(sub) - 3)

                if ellipse_count > longest:
                    longest = ellipse_count

                if ellipse_count < 0:
                    raise ValueError("Ellipses lengths do not match.")
                elif ellipse_count == 0:
                    split_subscripts[num] = sub.replace("...", "")
                else:
                    split_subscripts[num] = sub.replace(
                        "...", ellipse_inds[-ellipse_count:]
                    )

        subscripts = ",".join(split_subscripts)

        # Figure out output ellipses
        if longest == 0:
            out_ellipse = ""
        else:
            out_ellipse = ellipse_inds[-longest:]

        if out_sub:
            subscripts += "->" + output_sub.replace("...", out_ellipse)
        else:
            # Special care for outputless ellipses
            output_subscript = find_output_str(subscripts)
            normal_inds = "".join(sorted(set(output_subscript) - set(out_ellipse)))

            subscripts += "->" + out_ellipse + normal_inds

    # Build output string if does not exist
    if "->" in subscripts:
        input_subscripts, output_subscript = subscripts.split("->")
    else:
        input_subscripts, output_subscript = subscripts, find_output_str(subscripts)

    # Make sure output subscripts are unique and in the input
    for char in output_subscript:
        if output_subscript.count(char) != 1:
            raise ValueError(
                f"Output character '{char}' appeared more than once in the output."
            )
        if char not in input_subscripts:
            raise ValueError(f"Output character '{char}' did not appear in the input")

    # Make sure number operands is equivalent to the number of terms
    if len(input_subscripts.split(",")) != len(operands):
        raise ValueError(
            f"Number of einsum subscripts, {len(input_subscripts.split(','))}, must be equal to the "
            f"number of operands, {len(operands)}."
        )

    return input_subscripts, output_subscript, operands


def get_shape(x: Any):
    """Get the shape of the array-like object `x`. If `x` is not array-like, raise an error.

    Array-like objects are those that have a `shape` attribute, are sequences of BaseTypes, or are BaseTypes.
    BaseTypes are defined as `bool`, `int`, `float`, `complex`, `str`, and `bytes`.
    """
    if hasattr(x, "shape"):
        return x.shape
    elif isinstance(x, _BaseTypes):
        return ()
    elif isinstance(x, Sequence):
        shape = []
        while isinstance(x, Sequence) and not isinstance(x, _BaseTypes):
            shape.append(len(x))
            x = x[0]
        return tuple(shape)
    else:
        raise ValueError(
            f"Cannot determine the shape of {x}, can only determine the shape of array-like objects."
        )


class PathInfo:
    """A printable object to contain information about a contraction path."""

    def __init__(
        self,
        contraction_list,
        input_subscripts: str,
        output_subscript: str,
        indices: ArrayIndexType,
        path: PathType,
        scale_list: Sequence[int],
        naive_cost: int,
        opt_cost: int,
        size_list: Sequence[int],
        size_dict: Dict[str, int],
    ):
        self.contraction_list = contraction_list
        self.input_subscripts = input_subscripts
        self.output_subscript = output_subscript
        self.path = path
        self.indices = indices
        self.scale_list = scale_list
        self.naive_cost = Decimal(naive_cost)
        self.opt_cost = Decimal(opt_cost)
        self.speedup = self.naive_cost / max(self.opt_cost, Decimal(1))
        self.size_list = size_list
        self.size_dict = size_dict

        self.shapes = [
            tuple(size_dict[k] for k in ks) for ks in input_subscripts.split(",")
        ]
        self.eq = f"{input_subscripts}->{output_subscript}"
        self.largest_intermediate = Decimal(max(size_list, default=1))

    def __repr__(self) -> str:
        # Return the path along with a nice string representation
        header = ("scaling", "BLAS", "current", "remaining")

        path_print = [
            f"  Complete contraction:  {self.eq}\n",
            f"         Naive scaling:  {len(self.indices)}\n",
            f"     Optimized scaling:  {max(self.scale_list, default=0)}\n",
            f"      Naive FLOP count:  {self.naive_cost:.3e}\n",
            f"  Optimized FLOP count:  {self.opt_cost:.3e}\n",
            f"   Theoretical speedup:  {self.speedup:.3e}\n",
            f"  Largest intermediate:  {self.largest_intermediate:.3e} elements\n",
            "-" * 80 + "\n",
            "{:>6} {:>11} {:>22} {:>37}\n".format(*header),
            "-" * 80,
        ]

        for n, contraction in enumerate(self.contraction_list):
            _, _, einsum_str, remaining, do_blas = contraction

            if remaining is not None:
                remaining_str = ",".join(remaining) + "->" + self.output_subscript
            else:
                remaining_str = "..."
            size_remaining = max(0, 56 - max(22, len(einsum_str)))

            path_run = (
                self.scale_list[n],
                do_blas,
                einsum_str,
                remaining_str,
                size_remaining,
            )
            path_print.append("\n{:>4} {:>14} {:>22}    {:>{}}".format(*path_run))

        return "".join(path_print)


def _choose_memory_arg(
    memory_limit: _MemoryLimit, size_list: List[int]
) -> Optional[int]:
    if memory_limit == "max_input":
        return max(size_list)

    if isinstance(memory_limit, str):
        raise ValueError(
            "memory_limit must be None, int, or the string Literal['max_input']."
        )

    if memory_limit is None:
        return None

    if memory_limit < 1:
        if memory_limit == -1:
            return None
        else:
            raise ValueError("Memory limit must be larger than 0, or -1")

    return int(memory_limit)


def can_blas(
    inputs: List[str],
    result: str,
    idx_removed: ArrayIndexType,
    shapes: Union[Sequence[Tuple[int]], None] = None,
) -> Union[str, bool]:
    """Checks if we can use a BLAS call.

    Parameters
    ----------
    inputs : list of str
        Specifies the subscripts for summation.
    result : str
        Resulting summation.
    idx_removed : set
        Indices that are removed in the summation
    shapes : sequence of tuple[int], optional
        If given, check also that none of the indices are broadcast dimensions.

    Returns:
    -------
    type : str or bool
        The type of BLAS call to be used or False if none.

    Notes:
    -----
    We assume several operations are not efficient such as a transposed
    DDOT, therefore 'ijk,jki->' should prefer einsum. These return the blas
    type appended with "/EINSUM" to differentiate when they can still be done
    with tensordot if required, e.g. when a backend has no einsum.

    Examples:
    --------
    >>> can_blas(['ij', 'jk'], 'ik', set('j'))
    'GEMM'

    >>> can_blas(['ijj', 'jk'], 'ik', set('j'))
    False

    >>> can_blas(['ab', 'cd'], 'abcd', set())
    'OUTER/EINSUM'

    >>> # looks like GEMM but actually 'j' is broadcast:
    >>> can_blas(['ij', 'jk'], 'ik', set('j'), shapes=[(4, 1), (5, 6)])
    False
    """
    # Can only do two
    if len(inputs) != 2:
        return False

    input_left, input_right = inputs

    for c in set(input_left + input_right):
        # can't deal with repeated indices on same input or more than 2 total
        nl, nr = input_left.count(c), input_right.count(c)
        if (nl > 1) or (nr > 1) or (nl + nr > 2):
            return False

        # can't do implicit summation or dimension collapse e.g.
        #     "ab,bc->c" (implicitly sum over 'a')
        #     "ab,ca->ca" (take diagonal of 'a')
        if nl + nr - 1 == int(c in result):
            return False

    # check for broadcast indices e.g:
    #     "ij,jk->ik" (but one of the 'j' dimensions is broadcast up)
    if shapes is not None:
        for c in idx_removed:
            if shapes[0][input_left.find(c)] != shapes[1][input_right.find(c)]:
                return False

    # Prefer einsum if not removing indices
    #     (N.B. tensordot outer faster for large arrays?)
    if len(idx_removed) == 0:
        return "OUTER/EINSUM"

    # Build a few temporaries
    sets = [set(x) for x in inputs]
    keep_left = sets[0] - idx_removed
    keep_right = sets[1] - idx_removed
    rs = len(idx_removed)

    # DDOT
    if inputs[0] == inputs[1]:
        return "DOT"

    # DDOT does not make sense if you have to transpose - prefer einsum
    elif sets[0] == sets[1]:
        return "DOT/EINSUM"

    # GEMM no transpose
    if input_left[-rs:] == input_right[:rs]:
        return "GEMM"

    # GEMM transpose both
    elif input_left[:rs] == input_right[-rs:]:
        return "GEMM"

    # GEMM transpose right
    elif input_left[-rs:] == input_right[-rs:]:
        return "GEMM"

    # GEMM transpose left
    elif input_left[:rs] == input_right[:rs]:
        return "GEMM"

    # Einsum is faster than vectordot if we have to copy
    elif (len(keep_left) == 0) or (len(keep_right) == 0):
        return "GEMV/EINSUM"

    # Conventional tensordot
    else:
        return "TDOT"


def contract_path(
    subscripts: Any,
    *operands: Any,
    use_blas: bool = True,
    optimize=True,
    memory_limit=None,
    shapes: bool = False,
    **kwargs: Any,
) -> Tuple[PathType, PathInfo]:
    if (optimize is True) or (optimize is None):
        optimize = "auto"

    # Hidden option, only einsum should call this
    einsum_call_arg = kwargs.pop("einsum_call", False)
    if len(kwargs):
        raise TypeError(f"Did not understand the following kwargs: {kwargs.keys()}")

    # Python side parsing
    operands_ = [subscripts] + list(operands)
    input_subscripts, output_subscript, operands_prepped = parse_einsum_input(
        operands_, shapes=shapes
    )

    # Build a few useful list and sets
    input_list = input_subscripts.split(",")
    input_sets = [frozenset(x) for x in input_list]
    if shapes:
        input_shapes = operands_prepped
    else:
        input_shapes = [get_shape(x) for x in operands_prepped]
    output_set = frozenset(output_subscript)
    indices = frozenset(input_subscripts.replace(",", ""))

    # Get length of each unique dimension and ensure all dimensions are correct
    size_dict: Dict[str, int] = {}
    for tnum, term in enumerate(input_list):
        sh = input_shapes[tnum]

        if len(sh) != len(term):
            raise ValueError(
                f"Einstein sum subscript '{input_list[tnum]}' does not contain the "
                f"correct number of indices for operand {tnum}."
            )
        for cnum, char in enumerate(term):
            dim = int(sh[cnum])

            if char in size_dict:
                # For broadcasting cases we always want the largest dim size
                if size_dict[char] == 1:
                    size_dict[char] = dim
                elif dim not in (1, size_dict[char]):
                    raise ValueError(
                        f"Size of label '{char}' for operand {tnum} ({size_dict[char]}) does not match previous "
                        f"terms ({dim})."
                    )
            else:
                size_dict[char] = dim

    # Compute size of each input array plus the output array
    size_list = [
        compute_size_by_dict(term, size_dict)
        for term in input_list + [output_subscript]
    ]
    memory_arg = _choose_memory_arg(memory_limit, size_list)

    num_ops = len(input_list)

    # Compute naive cost
    # This is not quite right, need to look into exactly how einsum does this
    # indices_in_input = input_subscripts.replace(',', '')
    inner_product = (sum(len(x) for x in input_sets) - len(indices)) > 0
    naive_cost = flop_count(indices, inner_product, num_ops, size_dict)

    # Compute the path
    if optimize is False:
        path_tuple: PathType = [tuple(range(num_ops))]
    elif not isinstance(optimize, (str, PathOptimizer)):
        # Custom path supplied
        path_tuple = optimize  # type: ignore
    elif num_ops <= 2:
        # Nothing to be optimized
        path_tuple = [tuple(range(num_ops))]
    elif isinstance(optimize, PathOptimizer):
        # Custom path optimizer supplied
        path_tuple = optimize(input_sets, output_set, size_dict, memory_arg)
    else:
        path_optimizer = get_path_fn(optimize)
        path_tuple = path_optimizer(input_sets, output_set, size_dict, memory_arg)

    cost_list = []
    scale_list = []
    size_list = []
    contraction_list = []

    # Build contraction tuple (positions, gemm, einsum_str, remaining)
    for cnum, contract_inds in enumerate(path_tuple):
        # Make sure we remove inds from right to left
        contract_inds = tuple(sorted(contract_inds, reverse=True))

        contract_tuple = find_contraction(contract_inds, input_sets, output_set)
        out_inds, input_sets, idx_removed, idx_contract = contract_tuple

        # Compute cost, scale, and size
        cost = flop_count(
            idx_contract, bool(idx_removed), len(contract_inds), size_dict
        )
        cost_list.append(cost)
        scale_list.append(len(idx_contract))
        size_list.append(compute_size_by_dict(out_inds, size_dict))

        tmp_inputs = [input_list.pop(x) for x in contract_inds]
        tmp_shapes = [input_shapes.pop(x) for x in contract_inds]

        if use_blas:
            do_blas = can_blas(tmp_inputs, "".join(out_inds), idx_removed, tmp_shapes)
        else:
            do_blas = False

        # Last contraction
        if (cnum - len(path_tuple)) == -1:
            idx_result = output_subscript
        else:
            # use tensordot order to minimize transpositions
            all_input_inds = "".join(tmp_inputs)
            idx_result = "".join(sorted(out_inds, key=all_input_inds.find))

        shp_result = find_output_shape(tmp_inputs, tmp_shapes, idx_result)

        input_list.append(idx_result)
        input_shapes.append(shp_result)

        einsum_str = ",".join(tmp_inputs) + "->" + idx_result

        # for large expressions saving the remaining terms at each step can
        # incur a large memory footprint - and also be messy to print
        if len(input_list) <= 20:
            remaining: Optional[Tuple[str, ...]] = tuple(input_list)
        else:
            remaining = None

        contraction = (contract_inds, idx_removed, einsum_str, remaining, do_blas)
        contraction_list.append(contraction)

    opt_cost = sum(cost_list)

    if einsum_call_arg:
        return operands_prepped, contraction_list  # type: ignore

    path_print = PathInfo(
        contraction_list,
        input_subscripts,
        output_subscript,
        indices,
        path_tuple,
        scale_list,
        naive_cost,
        opt_cost,
        size_list,
        size_dict,
    )

    return path_tuple, path_print


class MetaFactorizedTensor(type):
    """Meta class for tensor factorizations

    .. info::

        1. Calls __new__ normally.
        2. Removes the keyword argument 'factorization' if present
        3. Calls __init__ with the remaining *args and **kwargs

    Why are we using this?
    ----------------------

    Tensor Factorization does not create its own instances.
    Instead, it defers to children class which do not take factorization as a parameter.

    We want to be able to create (e.g. CP) tensors in two ways:
    1. Indirectly: ``FactorizedTensor('cp', shape, rank)``
    2. Directly:   ``CP(shape, rank)``

    Note that in the second case, we don't want users to have to specify the
    factorization, it would be redundant to ask them to create a CP as
    ``CP(shape, rank, factorization='CP')``.

    This means we need to intercept the call to __init__ and remove the factorization parameter
    when creating an instance from FactorizedTensor. Hence this metaclass.

    Current solution
    ----------------

    This metaclass customizes the object creation process.

    In the metaclass
    ++++++++++++++++

    First, we call __new__ with all the *args and **kwargs
    Then, if we are in FactorizedTensor, we remove the first argument.
    This is because FactorizedTensor never uses factorization in its own init.

    In __new__
    ++++++++++

    If `cls` is FactorizedTensor, we actually replace `cls` by one of the subclasses depending on
    the value of factorization and so create an instance of that subclass.
    If `cls` is already a subclass, we just create an instance of that.

    Creating a factorized tensor through `FactorizedTensor`
    ----------------------------------------------------------

    When creating a FactorizedTensor, the calls are as follow:
    1. __call__(FactorizedTensor, *args, **kwargs)
       where args = [factorization, *rest_of_args]

    2. __call__ first calls FactorizedTensor.__new__(FactorizedTensor, factorization, *args, **kwargs)

       In FactorizedTensor.__new__, instead of creating a new instance, we check for factorization's value
       against the internal _factorization dict that we maintain and return
       a new instance of FactorizedTensor._factorizations[factorization]

    3. We are now back in __call__ which now removes factorization from the argument list ``args``
       and calls instance.__init__ (now instance is CP, Tucker, **not** FactorizedTensor) with the
       remaining args and kwargs

    4. Since FactorizedTensor's signature is __init__(self, factorization, *args, **kwargs),
       the direct subclasses of FactorizedTensor call super().__init__(None, *args, **kwargs)

       This means that in practice FactorizedTensor always gets factorization=None.
       This does not matter as we only use factorization during the creation process.

       However, this forces users to specify factorization as a first argument when creating a tensor
       from Tensor Factorization.

    Creation through a subclass`FactorizedTensor`
    ------------------------------------------------
    Let's say now the user wants to directly create an instance of a subclass of `FactorizedTensor`,
    in this example, let's say `CP`.

    When creating a CPTensor, the calls are as follow:

    1. __call__(CPTensor, *args, **kwargs)
       __call__ just calls __new__, then __init__ with the given arguments and keyword arguments.

    2. __call__ first calls CPTensor.__new__(CPTensor, *args, **kwargs).
       In turn, this calls FactorizedTensor.__new__(CPTensor, *args, **kwargs)

       Since `cls` is now `CPTensor`, not `FactorizedTensor`, nothing special is done
       and ``super().__new__(cls, *args, **kwargs)`` is called to create an instance

    3. We are now back in __call__ again. Since `cls` is CPTensor and not FactorizedTensor,
       we just call instance.__init__

    4. Now, in CPTensor.__init__, we re-add the mendatory first arg `factorization` by calling super() as
       ``super().__init__(self, None, *args, **kwargs)``
    """

    def __call__(cls, *args, **kwargs):
        instance = cls.__new__(cls, *args, **kwargs)
        kwargs.pop("factorization", None)

        instance.__init__(*args, **kwargs)
        return instance


def _format_factorization(factorization):
    """Small utility function to make sure factorization names
    are dealt with the same whether using capital letters or not.

    factorization=None is remapped to 'Dense'.
    """
    if factorization is None:
        factorization = "Dense"
    return factorization.lower()


class FactorizedTensor(nn.Module, metaclass=MetaFactorizedTensor):
    """Tensor in Factorized form

    .. important::

       All tensor factorization must have an `order` parameter
    """

    _factorizations = dict()

    def __init_subclass__(cls, name, **kwargs):
        """When a subclass is created, register it in _factorizations"""
        super().__init_subclass__(**kwargs)

        if name != "":
            cls._factorizations[_format_factorization(name)] = cls
            cls._name = name
        else:
            if (
                cls.__name__ != "TensorizedTensor"
            ):  # Don't display warning when instantiating the TensorizedTensor class
                warnings.warn(
                    f"Creating a subclass of FactorizedTensor {cls.__name__} with no name."
                )

    def __new__(cls, *args, **kwargs):
        """Customize the creation of a factorized convolution

        Takes a parameter `factorization`, a string that specifies with subclass to use

        Returns
        -------
        FactorizedTensor._factorizations[_format_factorization(factorization)]
            subclass implementing the specified tensor factorization
        """
        if cls is FactorizedTensor:
            factorization = kwargs.get("factorization")
            try:
                cls = cls._factorizations[_format_factorization(factorization)]
            except KeyError:
                raise ValueError(
                    f"Got factorization={factorization} but expected"
                    f"one of {cls._factorizations.keys()}"
                )

        instance = super().__new__(cls)

        return instance

    def __getitem__(indices):
        """Returns raw indexed factorization, not class

        Parameters
        ----------
        indices : int or tuple
        """
        raise NotImplementedError

    @classmethod
    def new(cls, shape, rank="same", factorization="Tucker", **kwargs):
        """Main way to create a factorized tensor

        Parameters
        ----------
        shape : tuple[int]
            shape of the factorized tensor to create
        rank : int, 'same' or float, default is 'same'
            rank of the decomposition
        factorization : {'CP', 'TT', 'Tucker'}, optional
            Tensor factorization to use to decompose the tensor, by default 'Tucker'

        Returns
        -------
        TensorFactorization
            Tensor in Factorized form.

        Examples
        --------
        Create a Tucker tensor of shape `(3, 4, 2)`
        with half the parameters as a dense tensor would:

        >>> tucker_tensor = FactorizedTensor.new((3, 4, 2)), rank=0.5, factorization='tucker')

        Raises
        ------
        ValueError
            If the factorization given does not exist.
        """
        try:
            cls = cls._factorizations[_format_factorization(factorization)]
        except KeyError:
            raise ValueError(
                f"Got factorization={factorization} but expected"
                f"one of {cls._factorizations.keys()}"
            )

        return cls.new(shape, rank, **kwargs)

    @classmethod
    def from_tensor(cls, tensor, rank, factorization="CP", **kwargs):
        """Create a factorized tensor by decomposing a dense tensor

        Parameters
        ----------
        tensor : torch.tensor
            tensor to factorize
        rank : int, 'same' or float
            rank of the decomposition
        factorization : {'CP', 'TT', 'Tucker'}, optional
            Tensor factorization to use to decompose the tensor, by default 'CP'

        Returns
        -------
        TensorFactorization
            Tensor in Factorized form.

        Raises
        ------
        ValueError
            If the factorization given does not exist.
        """
        try:
            cls = cls._factorizations[_format_factorization(factorization)]
        except KeyError:
            raise ValueError(
                f"Got factorization={factorization} but expected"
                f"one of {cls._factorizations.keys()}"
            )

        return cls.from_tensor(tensor, rank, **kwargs)

    def forward(self, indices=None, **kwargs):
        """To use a tensor factorization within a network, use ``tensor.forward``, or, equivalently, ``tensor()``

        Parameters
        ----------
        indices : int or tuple[int], optional
            use to index the tensor during the forward pass, by default None

        Returns
        -------
        TensorFactorization
            tensor[indices]
        """
        if indices is None:
            return self
        else:
            return self[indices]

    @property
    def decomposition(self):
        """Returns the factors and parameters composing the tensor in factorized form"""
        raise NotImplementedError

    @property
    def _factorization(self, indices=None, **kwargs):
        """Returns the raw, unprocessed indexed tensor, same as `forward` but without forward hooks

        Parameters
        ----------
        indices : int, or tuple of int
            use to index the tensor

        Returns
        -------
        TensorFactorization
            tensor[indices] but without any forward hook applied
        """
        if indices is None:
            return self
        else:
            return self[indices]

    def to_tensor(self):
        """Reconstruct the full tensor from its factorized form"""
        raise NotImplementedError

    def dim(self):
        """Order of the tensor

        Notes
        -----
        fact_tensor.dim() == fact_tensor.ndim

        See Also
        --------
        ndim
        """
        return len(self.shape)

    def numel(self):
        return int(np.prod(self.shape))

    @property
    def ndim(self):
        """Order of the tensor

        Notes
        -----
        fact_tensor.dim() == fact_tensor.ndim

        See Also
        --------
        dim
        """
        return len(self.shape)

    def size(self, index=None):
        """shape of the tensor

        Parameters
        ----------
        index : int, or tuple, default is None
            if not None, returns tensor.shape[index]

        See Also
        --------
        shape
        """
        if index is None:
            return self.shape
        else:
            return self.shape[index]

    def normal_(self, mean=0, std=1):
        """Inialize the factors of the factorization such that the **reconstruction** follows a Gaussian distribution

        Parameters
        ----------
        mean : float, currently only 0 is supported
        std : float
            standard deviation

        Returns
        -------
        self
        """
        if mean != 0:
            raise ValueError(f"Currently only mean=0 is supported, but got mean={mean}")

    def __repr__(self):
        return f"{self.__class__.__name__}(shape={self.shape}, rank={self.rank})"

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        args = [t.to_tensor() if hasattr(t, "to_tensor") else t for t in args]
        # return super().__torch_function__(func, types, args, kwargs)
        return func(*args, **kwargs)

    @property
    def name(self):
        """Factorization name ('tucker', 'tt', 'cp', ...)"""
        return self._name

    @property
    def tensor_shape(self):
        return self.shape


def unfold(tensor, mode):
    """Returns the mode-`mode` unfolding of `tensor` with modes starting at `0`.

    Parameters
    ----------
    tensor : ndarray
    mode : int, default is 0
           indexing starts at 0, therefore mode is in ``range(0, tensor.ndim)``

    Returns
    -------
    ndarray
        unfolded_tensor of shape ``(tensor.shape[mode], -1)``
    """
    return torch.reshape(torch.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1))


import torch


def cp_normalize(cp_tensor):
    # Допустим, _validate_cp_tensor уже возвращает нужный ранг
    # В Torch можно просто взять форму второго измерения любого фактора
    weights, factors = cp_tensor
    rank = factors[0].shape[1]

    if weights is None:
        # Создаем веса сразу на том же устройстве и типе, что и первый фактор
        weights = torch.ones(rank, device=factors[0].device, dtype=factors[0].dtype)

    normalized_factors = []

    for i, factor in enumerate(factors):
        if i == 0:
            factor = factor * weights
            weights = torch.ones(rank, device=factor.device, dtype=factor.dtype)

        # T.norm(axis=0) -> torch.norm(dim=0)
        scales = torch.norm(factor, p=2, dim=0)

        # Вместо T.where для обработки нулей используем простой clamp или индексацию
        # Это предотвращает деление на ноль
        scales_non_zero = scales.clone()
        scales_non_zero[scales_non_zero == 0] = 1.0

        weights = weights * scales

        # В Torch деление матрицы на вектор по столбцам удобно делать через broadcasting
        # reshape(1, -1) аналогичен TensorLy
        normalized_factors.append(factor / scales_non_zero.view(1, -1))

    # Возвращаем в том формате, который вы используете (кортеж или класс)
    return (weights, normalized_factors)


def initialize_cp(
    tensor,
    rank,
    init="svd",
    svd="truncated_svd",
    non_negative=False,
    random_state=None,
    normalize_factors=False,
    mask=None,
    svd_mask_repeats=5,
):
    r"""Initialize factors used in `parafac`.

    The type of initialization is set using `init`. If `init == 'random'` then
    initialize factor matrices with uniform distribution using `random_state`. If `init == 'svd'` then
    initialize the `m`th factor matrix using the `rank` left singular vectors
    of the `m`th unfolding of the input tensor. If init is a previously initialized `cp tensor`, all
    the weights are pulled in the last factor and then the weights are set to "1" for the output tensor.

    Parameters
    ----------
    tensor : ndarray
    rank : int
    init : {'svd', 'random', cptensor}, optional
    svd : str, default is 'truncated_svd'
        function to use to compute the SVD, acceptable values in tensorly.SVD_FUNS
    non_negative : bool, default is False
        if True, non-negative factors are returned

    Returns
    -------
    factors : CPTensor
        An initial cp tensor.

    """
    if isinstance(random_state, int):
        rng = torch.Generator(device=tensor.device).manual_seed(random_state)
    elif isinstance(random_state, torch.Generator):
        rng = random_state
    else:
        rng = None  # Или torch.default_generator

    if init == "random":
        # random_cp — это обычно функция из TensorLy.
        # В чистом Torch вы либо вызываете свою аналогичную функцию,
        # либо инициализируете факторы вручную.
        kt = random_cp(
            tensor.shape,
            rank,
            normalise_factors=False,
            generator=rng,  # Передаем Generator вместо numpy RandomState
            device=tensor.device,
            dtype=tensor.dtype,
        )

    elif init == "svd":
        factors = []
        for mode in range(tensor.ndim):
            mask_unfold = None if mask is None else unfold(mask, mode)
            U, S, _ = svd_interface(
                unfold(tensor, mode),
                n_eigenvecs=rank,
                method=svd,
                non_negative=non_negative,
                mask=mask_unfold,
                n_iter_mask_imputation=svd_mask_repeats,
            )

            # Put SVD initialization on the same scaling as the tensor in case normalize_factors=False
            if mode == 0:
                idx = min(rank, S.shape[0])
                # tl.index_update в Torch — это обычное присваивание через слайс
                # ВАЖНО: U[:, :idx] * S[:idx] работает через broadcasting (умножение каждой колонки на скаляр из S)
                U[:, :idx] = U[:, :idx] * S[:idx]

            if tensor.shape[mode] < rank:
                # TODO: this is a hack but it seems to do the job for now
                diff = rank - tensor.shape[mode]

                # Вместо tl.tensor(rng.random_sample(...), **tl.context(tensor))
                # Используем встроенный генератор, который сразу создает тензор на нужном девайсе
                random_part = torch.rand(
                    (U.shape[0], diff), device=tensor.device, dtype=tensor.dtype
                )

                # tl.concatenate -> torch.cat
                U = torch.cat([U, random_part], dim=1)

            factors.append(U[:, :rank])

        kt = CPTensor((None, factors))

    elif isinstance(init, (tuple, list, CPTensor)):
        # TODO: Test this
        try:
            if normalize_factors is True:
                warnings.warn(
                    "It is not recommended to initialize a tensor with normalizing. Consider normalizing the tensor before using this function"
                )

            kt = CPTensor(init)
            weights, factors = kt

            if torch.all(weights == 1.0):
                # Если вы используете свой класс CPTensor или просто кортеж
                kt = (None, factors)
            else:
                # weights.shape[0] в Torch аналогичен
                # tl.prod(weights) -> weights.prod()
                weights_avg = weights.prod() ** (1.0 / weights.shape[0])

                # В Torch можно обновить список факторов через inplace или обычное умножение
                for i in range(len(factors)):
                    factors[i] = factors[i] * weights_avg

                kt = (None, factors)

            return kt
        except ValueError:
            raise ValueError(
                "If initialization method is a mapping, then it must "
                "be possible to convert it to a CPTensor instance"
            )
    else:
        raise ValueError(f'Initialization method "{init}" not recognized')

    if non_negative:
        # Make decomposition feasible by taking the absolute value of all factor matrices
        kt.factors = [torch.abs(f) for f in kt[1]]

    if normalize_factors:
        kt = cp_normalize(kt)

    return kt


def sparsify_tensor(tensor, card):
    """Zeros out all elements in the `tensor` except `card` elements with maximum absolute values.

    Parameters
    ----------
    tensor : ndarray
    card : int
        Desired number of non-zero elements in the `tensor`

    Returns
    -------
    ndarray of shape tensor.shape
    """
    if card >= np.prod(tensor.shape):
        return tensor
    bound = torch.sort(torch.abs(tensor), axis=None)[-card]

    return torch.where(torch.abs(tensor) < bound, torch.zeros_like(tensor), tensor)


def validate_cp_rank(tensor_shape, rank="same", rounding="round"):
    """Returns the rank of a CP Decomposition

    Parameters
    ----------
    tensor_shape : tupe
        shape of the tensor to decompose
    rank : {'same', float, int}, default is same
        way to determine the rank, by default 'same'
        if 'same': rank is computed to keep the number of parameters (at most) the same
        if float, computes a rank so as to keep rank percent of the original number of parameters
        if int, just returns rank
    rounding = {'round', 'floor', 'ceil'}

    Returns
    -------
    rank : int
        rank of the decomposition
    """
    if rounding == "ceil":
        rounding_fun = np.ceil
    elif rounding == "floor":
        rounding_fun = np.floor
    elif rounding == "round":
        rounding_fun = np.round
    else:
        raise ValueError(
            f"Rounding should be of round, floor or ceil, but got {rounding}"
        )

    if rank == "same":
        rank = float(1)

    if isinstance(rank, float):
        rank = int(rounding_fun(np.prod(tensor_shape) * rank / np.sum(tensor_shape)))
    return rank


import torch


def _validate_cp_tensor(cp_tensor):
    # Если вы еще не создали свой класс CPTensor, эту проверку можно убрать
    # или заменить на проверку вашего пользовательского класса
    if hasattr(cp_tensor, "shape") and hasattr(cp_tensor, "rank"):
        return cp_tensor.shape, cp_tensor.rank

    # Случай скаляра (0-order tensor)
    if isinstance(cp_tensor, (float, int)):
        return (), 0

    weights, factors = cp_tensor

    # В PyTorch используем .ndim вместо T.ndim
    first_factor_ndim = factors[0].ndim

    if first_factor_ndim == 2:
        rank = factors[0].shape[1]
    elif first_factor_ndim == 1:
        rank = 1
    else:
        raise ValueError(
            f"Got a factor with {first_factor_ndim} dimensions, but CP factors "
            "should be at most 2D (shape: size, rank)."
        )

    shape = []
    for i, factor in enumerate(factors):
        # .shape в Torch возвращает torch.Size, который ведет себя как кортеж
        s = factor.shape
        if len(s) == 2:
            current_mode_size, current_rank = s
        else:  # Случай вектора (rank 1)
            current_mode_size = s[0]
            current_rank = 1

        if current_rank != rank:
            raise ValueError(
                f"All factors must have the same number of columns (rank). "
                f"Factor 0 has {rank}, but factor {i} has {current_rank}."
            )
        shape.append(current_mode_size)

    # Проверка весов
    if weights is not None:
        # Убеждаемся, что weights — это 1D тензор нужной длины
        if weights.shape != (rank,):
            raise ValueError(
                f"Weight mismatch: rank is {rank}, but weights.shape is {weights.shape}."
            )

    return tuple(shape), rank


def cp_norm(cp_tensor):
    # Допустим, _validate_cp_tensor адаптирован или пропущен
    weights, factors = cp_tensor

    # Вместо T.ones(...) **T.context
    # Создаем матрицу Грама для первого фактора: (A.T @ A)
    # Инициализируем единичной матрицей или сразу результатом первого фактора
    rank = factors[0].shape[1]
    norm = torch.ones((rank, rank), device=factors[0].device, dtype=factors[0].dtype)

    for f in factors:
        # T.dot(T.transpose(f), T.conj(f)) -> f.T @ f.conj()
        # В Torch .T для 2D — это короткий транспоз.
        # Для вещественных чисел .conj() можно опустить.
        norm = norm * (f.mth.conj().T @ f)

    if weights is not None:
        # Эквивалентно weights.view(-1, 1) @ weights.view(1, -1) но через внешнее произведение
        # Это создает матрицу W_ij = w_i * w_j
        w_matrix = weights.view(-1, 1) * weights.view(1, -1)
        norm = norm * w_matrix

    # Суммируем все элементы матрицы и берем корень
    return torch.sqrt(torch.sum(norm))


def fold(unfolded_tensor, mode, shape):
    """Refolds the mode-`mode` unfolding into a tensor of shape `shape`

        In other words, refolds the n-mode unfolded tensor
        into the original tensor of the specified shape.

    Parameters
    ----------
    unfolded_tensor : ndarray
        unfolded tensor of shape ``(shape[mode], -1)``
    mode : int
        the mode of the unfolding
    shape : tuple
        shape of the original tensor before unfolding

    Returns
    -------
    ndarray
        folded_tensor of shape `shape`
    """
    full_shape = list(shape)
    mode_dim = full_shape.pop(mode)
    full_shape.insert(0, mode_dim)
    return torch.moveaxis(torch.reshape(unfolded_tensor, full_shape), 0, mode)


def khatri_rao(factors, skip_matrix=None, mask=None):
    """Упрощенная реализация Khatri-Rao продукта на PyTorch"""
    if skip_matrix is not None:
        factors = [f for i, f in enumerate(factors) if i != skip_matrix]

    res = factors[0]
    for i in range(1, len(factors)):
        # Эффективный способ KR через вещание (broadcasting)
        res = torch.reshape(
            factors[i].unsqueeze(1) * res.unsqueeze(0), (-1, res.shape[1])
        )

    if mask is not None:
        res = res * mask.reshape(-1, 1)

    return res


def cp_to_tensor(cp_tensor, mask=None):
    # Используем ваш ранее написанный валидатор
    shape, rank = _validate_cp_tensor(cp_tensor)

    if not shape:  # 0-order tensor
        return cp_tensor

    weights, factors = cp_tensor

    # Случай вектора (1-й порядок)
    if len(shape) == 1:
        # В Torch axis=1 заменяется на dim=1
        return torch.sum(weights * factors[0], dim=1)

    # Если веса не заданы, используем 1
    if weights is None:
        w_factors = factors[0]
    else:
        w_factors = factors[0] * weights

    if mask is None:
        # Основная логика: dot(factors[0], khatri_rao(others).T)
        # Нам понадобится реализация khatri_rao на Torch (см. ниже)
        kr_part = khatri_rao(factors, skip_matrix=0)
        full_tensor = w_factors @ kr_part.t()
    else:
        # Если есть маска, суммируем по рангу напрямую
        # Это более ресурсозатратно, но соответствует логике TensorLy с маской
        full_tensor = torch.sum(khatri_rao([w_factors] + factors[1:], mask=mask), dim=1)

    # fold в TensorLy для mode=0 — это просто reshape к исходной форме
    return full_tensor.reshape(shape)


import torch


def error_calc(tensor, norm_tensor, weights, factors, sparsity, mask, mttkrp=None):
    # Если есть маска или нет предвычисленного MTTKRP, строим полный тензор
    if (mask is not None) or (mttkrp is None):
        # Используем вашу версию cp_to_tensor на Torch
        low_rank_component = cp_to_tensor((weights, factors))

        if mask is not None:
            # Обновляем тензор: заполняем пропуски значениями из аппроксимации
            tensor = tensor * mask + low_rank_component * (1 - mask)
            norm_tensor = torch.linalg.norm(tensor)

        if sparsity:
            # Предполагается, что sparsify_tensor тоже переписана на Torch
            sparse_component = sparsify_tensor(tensor - low_rank_component, sparsity)
        else:
            sparse_component = 0.0

        unnorml_rec_error = torch.linalg.norm(
            tensor - low_rank_component - sparse_component
        )

    else:
        if sparsity:
            low_rank_component = cp_to_tensor((weights, factors))
            sparse_component = sparsify_tensor(tensor - low_rank_component, sparsity)
            unnorml_rec_error = torch.linalg.norm(
                tensor - low_rank_component - sparse_component
            )
        else:
            # Оптимизированный путь: ||A - B||^2 = ||A||^2 + ||B||^2 - 2<A, B>
            # Используем вашу функцию cp_norm
            factors_norm = cp_norm((weights, factors))

            # Скалярное произведение <tensor, rec> через MTTKRP и последний фактор
            # В Torch это просто сумма поэлементного произведения
            # .conj() нужен только для комплексных чисел
            iprod = torch.sum(mttkrp * factors[-1].conj())

            # В PyTorch возведение в квадрат и abs работают стандартно
            # Используем clamp_min(0), чтобы избежать отрицательных чисел из-за ошибок округления
            sq_error = norm_tensor**2 + factors_norm**2 - 2 * iprod
            unnorml_rec_error = torch.sqrt(torch.abs(sq_error))

    return unnorml_rec_error, tensor, norm_tensor


def unfolding_dot_khatri_rao(tensor, cp_tensor, mode):
    weights, factors = cp_tensor
    ndims = tensor.ndim

    # Подготавливаем символы для индексов (i, j, k, l...)
    all_indices = [chr(ord("a") + i) for i in range(ndims)]
    # Индекс для ранга
    rank_index = "R"

    # Индексы тензора: 'abcd'
    tensor_indices = "".join(all_indices)

    # Индексы факторов: 'aR', 'bR', 'dR' (пропуская целевую моду)
    factor_indices = []
    needed_factors = []
    for i in range(ndims):
        if i != mode:
            factor_indices.append(f"{all_indices[i]}{rank_index}")
            needed_factors.append(factors[i])

    # Результирующие индексы: 'cR' (где c - индекс текущей моды)
    target_index = f"{all_indices[mode]}{rank_index}"

    # Итоговая строка einsum: "abcd,bR,cR,dR->aR" (если mode=0)
    einsum_str = f"{tensor_indices},{','.join(factor_indices)}->{target_index}"

    # Вычисляем MTTKRP
    mttkrp = torch.einsum(einsum_str, tensor, *needed_factors)

    # Если есть веса, применяем их в конце
    if weights is not None:
        mttkrp = mttkrp * weights

    return mttkrp


def parafac(
    tensor,
    rank,
    n_iter_max=100,
    init="svd",
    svd="truncated_svd",
    normalize_factors=False,
    orthogonalise=False,
    tol=1e-8,
    random_state=None,
    verbose=0,
    return_errors=False,
    sparsity=None,
    l2_reg=0,
    mask=None,
    cvg_criterion="abs_rec_error",
    fixed_modes=None,
    svd_mask_repeats=5,
    linesearch=False,
    callback=None,
):
    """CANDECOMP/PARAFAC decomposition via alternating least squares (ALS)
    Computes a rank-`rank` decomposition of `tensor` [1]_ such that:

    ``tensor = [|weights; factors[0], ..., factors[-1] |]``.

    Parameters
    ----------
    tensor : ndarray
    rank  : int
        Number of components.
    n_iter_max : int
        Maximum number of iteration
    init : {'svd', 'random', CPTensor}, optional
        Type of factor matrix initialization.
        If a CPTensor is passed, this is directly used for initalization.
        See `initialize_factors`.
    svd : str, default is 'truncated_svd'
        function to use to compute the SVD, acceptable values in tensorly.SVD_FUNS
    normalize_factors : if True, aggregate the weights of each factor in a 1D-tensor
        of shape (rank, ), which will contain the norms of the factors
    tol : float, optional
        (Default: 1e-6) Relative reconstruction error tolerance. The
        algorithm is considered to have found the global minimum when the
        reconstruction error is less than `tol`.
    random_state : {None, int, np.random.RandomState}
    verbose : int, optional
        Level of verbosity
    return_errors : bool, optional
        Activate return of iteration errors
    mask : ndarray
        array of booleans with the same shape as ``tensor`` should be 0 where
        the values are missing and 1 everywhere else. Note:  if tensor is
        sparse, then mask should also be sparse with a fill value of 1 (or
        True). Allows for missing values [2]_
    cvg_criterion : {'abs_rec_error', 'rec_error'}, optional
       Stopping criterion for ALS, works if `tol` is not None.
       If 'rec_error',  ALS stops at current iteration if ``(previous rec_error - current rec_error) < tol``.
       If 'abs_rec_error', ALS terminates when `|previous rec_error - current rec_error| < tol`.
    sparsity : float or int
        If `sparsity` is not None, we approximate tensor as a sum of low_rank_component and sparse_component, where low_rank_component = cp_to_tensor((weights, factors)). `sparsity` denotes desired fraction or number of non-zero elements in the sparse_component of the `tensor`.
    fixed_modes : list, default is None
        A list of modes for which the initial value is not modified.
        The last mode cannot be fixed due to error computation.
    svd_mask_repeats: int
        If using a tensor with masked values, this initializes using SVD multiple times to
        remove the effect of these missing values on the initialization.
    linesearch : bool, default is False
        Whether to perform line search as proposed by Bro [3].

    Returns
    -------
    CPTensor : (weight, factors)
        * weights : 1D array of shape (rank, )

          * all ones if normalize_factors is False (default)
          * weights of the (normalized) factors otherwise

        * factors : List of factors of the CP decomposition element `i` is of shape ``(tensor.shape[i], rank)``
        * sparse_component : nD array of shape tensor.shape. Returns only if `sparsity` is not None.

    errors : list
        A list of reconstruction errors at each iteration of the algorithms.

    References
    ----------
    .. [1] T.G.Kolda and B.W.Bader, "Tensor Decompositions and Applications", SIAM
           REVIEW, vol. 51, n. 3, pp. 455-500, 2009.
    .. [2] Tomasi, Giorgio, and Rasmus Bro. "PARAFAC and missing values."
           Chemometrics and Intelligent Laboratory Systems 75.2 (2005): 163-180.
    .. [3] R. Bro, "Multi-Way Analysis in the Food Industry: Models, Algorithms, and
           Applications", PhD., University of Amsterdam, 1998
    """
    rank = validate_cp_rank(tensor.shape, rank=rank)

    if return_errors:
        DeprecationWarning(
            "return_errors argument will be removed in the next version of TensorLy. Please use a callback function instead."
        )

    if orthogonalise and not isinstance(orthogonalise, int):
        orthogonalise = n_iter_max

    if linesearch:
        acc_pow = 2.0  # Extrapolate to the iteration^(1/acc_pow) ahead
        acc_fail = 0  # How many times acceleration have failed
        max_fail = 4  # Increase acc_pow with one after max_fail failure

    weights, factors = initialize_cp(
        tensor,
        rank,
        init=init,
        svd=svd,
        random_state=random_state,
        normalize_factors=normalize_factors,
        mask=mask,
        svd_mask_repeats=svd_mask_repeats,
    )

    rec_errors = []
    # tl.norm(tensor, 2) -> torch.linalg.norm или .norm()
    norm_tensor = torch.linalg.norm(tensor)

    if l2_reg:
        # tl.eye(rank, **tl.context(tensor)) -> torch.eye(...)
        # Указываем устройство и тип данных напрямую из исходного тензора
        Id = torch.eye(rank, device=tensor.device, dtype=tensor.dtype) * l2_reg
    else:
        Id = 0

    if fixed_modes is None:
        fixed_modes = []

    if fixed_modes == list(range(tensor.ndim)):  # Check If all modes are fixed
        cp_tensor = CPTensor(
            (weights, factors)
        )  # No need to run optimization algorithm, just return the initialization
        return cp_tensor

    if tensor.ndim - 1 in fixed_modes:
        warnings.warn(
            "You asked for fixing the last mode, which is not supported.\n The last mode will not be fixed. Consider using torch.moveaxis()"
        )
        fixed_modes.remove(tensor.ndim - 1)
    modes_list = [mode for mode in range(tensor.ndim) if mode not in fixed_modes]

    if sparsity:
        sparse_component = torch.zeros_like(tensor)
        if isinstance(sparsity, float):
            sparsity = int(sparsity * np.prod(tensor.shape))
        else:
            sparsity = int(sparsity)

    if callback is not None:
        cp_tensor = CPTensor((weights, factors))
        unnorml_rec_error, _, norm_tensor = error_calc(
            tensor, norm_tensor, weights, factors, sparsity, mask
        )
        callback_error = unnorml_rec_error / norm_tensor

        if sparsity:
            sparse_component = sparsify_tensor(
                tensor - cp_to_tensor((weights, factors)), sparsity
            )
            callback((cp_tensor, sparse_component), callback_error)
        else:
            callback(cp_tensor, callback_error)

    for iteration in range(n_iter_max):
        if orthogonalise and iteration <= orthogonalise:
            # tl.qr(f)[0] -> torch.linalg.qr(f).Q
            factors = [
                torch.linalg.qr(f).Q if min(f.shape) >= rank else f
                for i, f in enumerate(factors)
            ]

        if linesearch and iteration % 2 == 0:
            # tl.copy(f) -> f.clone()
            factors_last = [f.clone() for f in factors]
            weights_last = weights.clone() if weights is not None else None

        if verbose > 1:
            print(f"Starting iteration {iteration + 1}")

        for mode in modes_list:
            if verbose > 1:
                print(f"Mode {mode} of {tensor.ndim}")

            # Создаем матрицу Грама (pseudo_inverse)
            # Инициализируем единицами на нужном устройстве
            pseudo_inverse = torch.ones(
                (rank, rank), device=tensor.device, dtype=tensor.dtype
            )
            for i, factor in enumerate(factors):
                if i != mode:
                    # tl.dot(T.conj(T.transpose(f)), f) -> f.conj().T @ f
                    pseudo_inverse = pseudo_inverse * (factor.conj().T @ factor)

            pseudo_inverse += Id

            # Модификация через веса (broadcasting)
            if weights is not None:
                # .view(-1, 1) быстрее и нагляднее reshape
                pseudo_inverse = (
                    weights.view(-1, 1) * pseudo_inverse * weights.view(1, -1)
                )

            # Эту функцию нужно реализовать на Torch (обычно через тензорное сжатие)
            mttkrp = unfolding_dot_khatri_rao(tensor, (weights, factors), mode)

            # Решение системы уравнений: pseudo_inverse * factor.T = mttkrp.T
            # tl.solve(A, B) -> torch.linalg.solve(A, B)
            # Используем .T (транспонирование), так как Torch ожидает (matrix, right_hand_side)
            factor = torch.linalg.solve(pseudo_inverse.conj().T, mttkrp.T).T

            factors[mode] = factor

        # Will we be performing a line search iteration
        if linesearch and iteration % 2 == 0 and iteration > 5:
            line_iter = True
        else:
            line_iter = False

        # Calculate the current unnormalized error if we need it
        if (tol or return_errors) and not line_iter:
            unnorml_rec_error, tensor, norm_tensor = error_calc(
                tensor, norm_tensor, weights, factors, sparsity, mask, mttkrp
            )
        else:
            if mask is not None:
                tensor = tensor * mask + cp_to_tensor((weights, factors), mask=1 - mask)

        # Start line search if requested.
        if line_iter:
            jump = iteration ** (1.0 / acc_pow)

            new_weights = weights_last + (weights - weights_last) * jump
            new_factors = [
                factors_last[ii] + (factors[ii] - factors_last[ii]) * jump
                for ii in range(tensor.ndim)
            ]

            new_rec_error, new_tensor, new_norm_tensor = error_calc(
                tensor, norm_tensor, new_weights, new_factors, sparsity, mask
            )

            if (new_rec_error / new_norm_tensor) < rec_errors[-1]:
                factors, weights = new_factors, new_weights
                tensor, norm_tensor = new_tensor, new_norm_tensor
                unnorml_rec_error = new_rec_error
                acc_fail = 0

                if verbose:
                    print(f"Accepted line search jump of {jump}.")
            else:
                unnorml_rec_error, tensor, norm_tensor = error_calc(
                    tensor, norm_tensor, weights, factors, sparsity, mask, mttkrp
                )
                acc_fail += 1

                if verbose:
                    print(f"Line search failed for jump of {jump}.")

                if acc_fail == max_fail:
                    acc_pow += 1.0
                    acc_fail = 0

                    if verbose:
                        print("Reducing acceleration.")

        if (tol or return_errors) and not line_iter:
            rec_error = unnorml_rec_error / norm_tensor
            rec_errors.append(rec_error)

        if callback is not None:
            cp_tensor = CPTensor((weights, factors))

            if sparsity:
                sparse_component = sparsify_tensor(
                    tensor - cp_to_tensor((weights, factors)), sparsity
                )
                retVal = callback((cp_tensor, sparse_component), rec_error)
            else:
                retVal = callback(cp_tensor, rec_error)

            if retVal is True:
                if verbose:
                    print("Received True from callback function. Exiting.")
                break

        if tol:
            if iteration >= 1:
                rec_error_decrease = rec_errors[-2] - rec_errors[-1]

                if verbose:
                    print(
                        f"iteration {iteration}, reconstruction error: {rec_error}, decrease = {rec_error_decrease}, unnormalized = {unnorml_rec_error}"
                    )

                if cvg_criterion == "abs_rec_error":
                    stop_flag = torch.abs(rec_error_decrease) < tol
                elif cvg_criterion == "rec_error":
                    stop_flag = rec_error_decrease < tol
                else:
                    raise TypeError("Unknown convergence criterion")

                if stop_flag:
                    if verbose:
                        print(f"PARAFAC converged after {iteration} iterations")
                    break

            else:
                if verbose:
                    print(f"reconstruction error={rec_errors[-1]}")
        if normalize_factors:
            weights, factors = cp_normalize((weights, factors))

    cp_tensor = CPTensor((weights, factors))

    if sparsity:
        sparse_component = sparsify_tensor(
            tensor - cp_to_tensor((weights, factors)), sparsity
        )
        cp_tensor = (cp_tensor, sparse_component)

    if return_errors:
        return cp_tensor, rec_errors
    else:
        return cp_tensor


def svd_flip(U, V, u_based_decision=True):
    if u_based_decision:
        # Находим индексы максимальных по модулю элементов в каждом столбце U
        max_abs_cols = torch.argmax(torch.abs(U), dim=0)

        # Векторный способ получить значения U[max_abs_cols, range]
        # Это заменяет конструкцию [U[i, j] for (i, j) in zip(...)]
        col_indices = torch.arange(U.shape[1], device=U.device)
        signs = torch.sign(U[max_abs_cols, col_indices])

        U = U * signs

        if V.shape[0] > U.shape[1]:
            # Создаем тензор единиц на том же устройстве
            extra_ones = torch.ones(
                V.shape[0] - U.shape[1], device=V.device, dtype=V.dtype
            )
            signs = torch.cat((signs, extra_ones))

        V = V * signs[: V.shape[0]].view(-1, 1)
    else:
        # Находим индексы максимальных элементов в каждой строке V
        max_abs_rows = torch.argmax(torch.abs(V), dim=1)

        row_indices = torch.arange(V.shape[0], device=V.device)
        signs = torch.sign(V[row_indices, max_abs_rows])

        V = V * signs.view(-1, 1)

        if U.shape[1] > V.shape[0]:
            extra_ones = torch.ones(
                U.shape[1] - V.shape[0], device=U.device, dtype=U.dtype
            )
            signs = torch.cat((signs, extra_ones))

        U = U * signs[: U.shape[1]]

    return U, V


def soft_thresholding(tensor, threshold):
    # F.relu(x) — это эффективный аналог torch.clamp(x, min=0)
    return torch.sign(tensor) * F.relu(torch.abs(tensor) - threshold)


def make_svd_non_negative(tensor, U, S, V, nntype=True):
    if nntype is True:
        nntype = "nndsvda"

    # Инициализация тензоров сразу на устройстве входного тензора
    W = torch.zeros_like(U)
    H = torch.zeros_like(V)

    # Ведущая компонента (S[0] в Torch — это скаляр, если S это вектор сингулярных чисел)
    W[:, 0] = torch.sqrt(S[0]) * torch.abs(U[:, 0])
    H[0, :] = torch.sqrt(S[0]) * torch.abs(V[0, :])

    # Цикл по рангам
    for j in range(1, min(U.shape[1], V.shape[0])):
        x, y = U[:, j], V[j, :]

        # Получаем положительные и отрицательные части
        x_p, y_p = F.relu(x), F.relu(y)
        x_n, y_n = F.relu(-x), F.relu(-y)  # Быстрее, чем abs(clamp(max=0))

        # Нормы
        x_p_nrm, y_p_nrm = torch.linalg.norm(x_p), torch.linalg.norm(y_p)
        x_n_nrm, y_n_nrm = torch.linalg.norm(x_n), torch.linalg.norm(y_n)

        m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm

        if m_p > m_n:
            u, v, sigma = x_p / x_p_nrm, y_p / y_p_nrm, m_p
        else:
            u, v, sigma = x_n / x_n_nrm, y_n / y_n_nrm, m_n

        lbd = torch.sqrt(S[j] * sigma)
        W[:, j] = lbd * u
        H[j, :] = lbd * v

    # Машинное эпсилон для точности
    eps = torch.finfo(tensor.dtype).eps

    if nntype == "nndsvd":
        W = soft_thresholding(W, eps)
        H = soft_thresholding(H, eps)
    elif nntype == "nndsvda":
        avg = torch.mean(tensor)
        # В Torch.where не нужно создавать тензор из единиц, можно просто передать скаляр avg
        W = torch.where(W < eps, avg, W)
        H = torch.where(H < eps, avg, H)
    else:
        raise ValueError(f"Invalid nntype: {nntype}")

    return W, H


def randomized_range_finder(A, n_dims, n_iter=2, random_state=None):
    # Управление случайным состоянием в стиле Torch
    if isinstance(random_state, int):
        generator = torch.Generator(device=A.device).manual_seed(random_state)
    else:
        generator = None  # Использует глобальный генератор torch

    dim_1, dim_2 = A.shape

    # Создаем случайную матрицу сразу в контексте A (device и dtype)
    # torch.randn эффективнее, чем rng.normal из NumPy
    Q = torch.randn(
        (dim_2, n_dims), device=A.device, dtype=A.dtype, generator=generator
    )

    # Первое приближение: QR(A @ Q)
    # В Torch матричное умножение — это @, QR — в модуле linalg
    Q, _ = torch.linalg.qr(A @ Q)

    # Power iterations (степенные итерации) для уточнения подпространства
    # Сопряженное транспонирование: .mth.conj().T или .H (в новых версиях)
    A_H = A.mth.conj().T

    for i in range(n_iter):
        # Оборот через сопряженную матрицу и обратно
        Q, _ = torch.linalg.qr(A_H @ Q)
        Q, _ = torch.linalg.qr(A @ Q)

    return Q


def svd_checks(matrix, n_eigenvecs=None):
    """Runs common checks to all of the SVD methods.

    Parameters
    ----------
    matrix : 2D-array
    n_eigenvecs : int, optional, default is None
        if specified, number of eigen[vectors-values] to return

    Returns
    -------
    n_eigenvecs : int
        the number of eigenvectors to solve for
    min_dim : int
        the minimum dimension of matrix
    max_dim : int
        the maximum dimension of matrix
    """
    # Check that matrix is... a matrix!
    if matrix.ndim != 2:
        raise ValueError(f"matrix be a matrix. matrix.ndim is {matrix.ndim} != 2")

    dim_1, dim_2 = matrix.shape
    min_dim, max_dim = min(dim_1, dim_2), max(dim_1, dim_2)

    if n_eigenvecs is None:
        n_eigenvecs = max_dim

    if n_eigenvecs > max_dim:
        warnings.warn(
            f"Trying to compute SVD with n_eigenvecs={n_eigenvecs}, which is larger "
            f"than max(matrix.shape)={max_dim}. Setting n_eigenvecs to {max_dim}."
        )
        n_eigenvecs = max_dim

    return n_eigenvecs, min_dim, max_dim


def truncated_svd(matrix, n_eigenvecs=None, **kwargs):
    """Усеченное SVD на чистом PyTorch"""
    # Предполагается, что svd_checks адаптирована под Torch (возвращает n_eigenvecs и min_dim)
    n_eigenvecs, min_dim, _ = svd_checks(matrix, n_eigenvecs=n_eigenvecs)

    # full_matrices=False в Torch эквивалентно 'reduced' SVD.
    # Это значит, что U будет формы (M, K), а Vh — (K, N), где K = min(M, N).
    U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)

    # Возвращаем срезы до нужного количества векторов
    # В Torch SVD возвращает Vh (V сопряженное транспонированное), что соответствует V из TensorLy
    return U[:, :n_eigenvecs], S[:n_eigenvecs], Vh[:n_eigenvecs, :]


def symeig_svd(matrix, n_eigenvecs=None, **kwargs):
    # Адаптированный svd_checks
    n_eigenvecs, _, _ = svd_checks(matrix, n_eigenvecs=n_eigenvecs)
    dim_1, dim_2 = matrix.shape
    eps = torch.finfo(matrix.dtype).eps

    if dim_1 > dim_2:
        # Эквивалент tl.eigh(tl.dot(matrix, tl.transpose(matrix)))
        # Но для случая dim_1 > dim_2 выгоднее считать Gram matrix по меньшей размерности
        # Однако следуем логике вашего кода:
        S, U = torch.linalg.eigh(matrix @ matrix.mth.conj().T)
        # Ограничиваем снизу эпсилоном, чтобы избежать корня из отрицательных чисел и деления на 0
        S = torch.sqrt(torch.clamp(S, min=eps))
        V = (matrix.mth.conj().T @ U) / S.view(1, -1)
    else:
        # Обычный случай: Gram matrix (M.T @ M)
        S, V = torch.linalg.eigh(matrix.mth.conj().T @ matrix)
        S = torch.sqrt(torch.clamp(S, min=eps))
        U = (matrix @ V) / S.view(1, -1)

    # eigh возвращает результат в порядке возрастания.
    # Для SVD нужно развернуть (flip), чтобы было по убыванию.
    U = torch.flip(U, dims=(1,))
    S = torch.flip(S, dims=(0,))
    V = torch.flip(V.mth.conj().T, dims=(0,))

    # Находим фактическое количество векторов для возврата
    k = min(dim_1, dim_2, n_eigenvecs)

    return (U[:, :k], S[:k], V[:k, :])


def randomized_svd(
    matrix,
    n_eigenvecs=None,
    n_oversamples=5,
    n_iter=2,
    random_state=None,
    **kwargs,
):
    # n_eigenvecs, min_dim, max_dim = svd_checks(matrix, n_eigenvecs=n_eigenvecs)
    # Предполагаем, что svd_checks возвращает актуальные значения
    n_eigenvecs, min_dim, max_dim = svd_checks(matrix, n_eigenvecs=n_eigenvecs)

    dim_1, dim_2 = matrix.shape
    n_dims = min(n_eigenvecs + n_oversamples, max_dim)

    # Логика выбора: транспонировать ли матрицу для минимизации размера уменьшенной матрицы
    if (
        dim_1 > dim_2
        and n_eigenvecs > min(min_dim, n_dims)
        or dim_1 < dim_2
        and n_eigenvecs < min(min_dim, n_dims)
    ):
        # Работаем с транспонированной матрицей
        matrix_T = matrix.mth.conj().T

        # Используем вашу функцию randomized_range_finder на Torch
        Q = randomized_range_finder(
            matrix_T, n_dims=n_dims, n_iter=n_iter, random_state=random_state
        )

        # matrix_reduced = (Q.H @ matrix_T).H
        # В Torch: (Q.T @ A.T).T == A @ Q
        matrix_reduced = (Q.mth.conj().T @ matrix_T).mth.conj().T

        U, S, V = truncated_svd(matrix_reduced, n_eigenvecs=n_eigenvecs)
        # V = V @ Q.T
        V = V @ Q.mth.conj().T
    else:
        # Прямой путь
        Q = randomized_range_finder(
            matrix, n_dims=n_dims, n_iter=n_iter, random_state=random_state
        )

        # Проекция матрицы на найденное подпространство
        matrix_reduced = Q.mth.conj().T @ matrix

        U, S, V = truncated_svd(matrix_reduced, n_eigenvecs=n_eigenvecs)

        # Восстановление размерности U: U = Q @ U_reduced
        U = Q @ U

    return U, S, V


SVD_FUNS = ["truncated_svd", "symeig_svd", "randomized_svd"]
SVD_TYPES = Literal["truncated_svd", "symeig_svd", "randomized_svd"]


def svd_interface(
    matrix,
    method="truncated_svd",
    n_eigenvecs=None,
    flip_sign=True,
    u_based_flip_sign=True,
    non_negative=None,
    mask=None,
    n_iter_mask_imputation=5,
    **kwargs,
):
    """Dispatching function to various SVD algorithms, alongside additional
    properties such as resolving sign invariance, imputation, and non-negativity.

    Parameters
    ----------
    matrix : tensor
        A 2D tensor.
    method : str, default is 'truncated_svd'
        Function to use to compute the SVD, acceptable values in tensorly.SVD_FUNS or a callable.
    n_eigenvecs : int, optional, default is None
        If specified, number of eigen[vectors-values] to return.
    flip_sign : bool, optional, default is True
        Whether to resolve the sign indeterminacy of SVD.
    u_based_flip_sign : bool, optional, default is True
        Whether the sign indeterminacy should be resolved using U (vs. V).
    non_negative : bool, optional, default is False
        Whether to make the SVD results non-negative.
    nn_type : str, default is 'nndsvd'
        Algorithm to use for converting U to be non-negative.
    mask : tensor, default is None.
        Array of booleans with the same shape as ``matrix``. Should be 0 where
        the values are missing and 1 everywhere else. None if nothing is missing.
        Imputation is done by iterative low rank approximation, so n_eigenvecs should be provided
        and be lower than the rank of the matrix.
    n_iter_mask_imputation : int, default is 5
        Number of repetitions to apply in missing value imputation.
    **kwargs : optional
        Arguments passed along to individual SVD algorithms.

    Returns
    -------
    U : 2-D tensor, shape (matrix.shape[0], n_eigenvecs)
        Contains the right singular vectors of `matrix`
    S : 1-D tensor, shape (n_eigenvecs, )
        Contains the singular values of `matrix`
    V : 2-D tensor, shape (n_eigenvecs, matrix.shape[1])
        Contains the left singular vectors of `matrix`
    """

    if method == "truncated_svd":
        svd_fun = truncated_svd
    elif method == "symeig_svd":
        svd_fun = symeig_svd
    elif method == "randomized_svd":
        svd_fun = randomized_svd
    elif callable(method):
        svd_fun = method
    else:
        raise ValueError(
            f"Got svd={method}. However, the possible choices are {SVD_FUNS} or to pass a callable."
        )

    U, S, V = svd_fun(matrix, n_eigenvecs=n_eigenvecs, **kwargs)

    if mask is not None and n_eigenvecs is not None:
        for _ in range(n_iter_mask_imputation):
            # Эквивалент (U @ St @ V), но без создания промежуточной матрицы St.
            # Мы просто масштабируем столбцы U значениями из S.
            # Это работает быстрее и тратит меньше памяти.
            reconstruction = (U * S) @ V

            # Обновляем значения в матрице только там, где маска равна 0
            matrix = matrix * mask + reconstruction * (1 - mask)

            # Пересчитываем SVD
            U, S, V = svd_fun(matrix, n_eigenvecs=n_eigenvecs, **kwargs)

    if flip_sign:
        U, V = svd_flip(U, V, u_based_decision=u_based_flip_sign)

    if non_negative is not False and non_negative is not None:
        U, V = make_svd_non_negative(matrix, U, S, V, non_negative)

    return U, S, V


import torch


def initialize_tucker(
    tensor,
    rank,
    modes,
    random_state,
    init="svd",
    svd="truncated_svd",
    non_negative=False,
    mask=None,
    svd_mask_repeats=5,
):
    if init == "svd":
        factors = []
        for index, mode in enumerate(modes):
            # tl.unfold(tensor, mode) -> tensor.moveaxis(mode, 0).reshape(tensor.shape[mode], -1)
            unfolded = tensor.moveaxis(mode, 0).reshape(tensor.shape[mode], -1)

            mask_unfold = None
            if mask is not None:
                mask_unfold = mask.moveaxis(mode, 0).reshape(mask.shape[mode], -1)

            # svd_interface должна возвращать U, S, V на Torch
            U, _, _ = svd_interface(
                unfolded,
                n_eigenvecs=rank[index],
                method=svd,
                non_negative=non_negative,
                mask=mask_unfold,
                n_iter_mask_imputation=svd_mask_repeats,
                random_state=random_state,
            )
            factors.append(U)

        # Начальное ядро (core) через multi_mode_dot
        # В Torch это последовательное умножение тензора на факторы (factors[i].T)
        core = tensor
        for i, mode in enumerate(modes):
            # Умножение тензора по моде: tucker_to_tensor в миниатюре
            core = tucker_mode_dot(core, factors[i].mth.conj().T, mode)

    elif init == "random":
        if isinstance(random_state, int):
            gen = torch.Generator(device=tensor.device).manual_seed(random_state)
        else:
            gen = None

        core_shape = [rank[i] for i in range(len(modes))]
        # Прямое создание на устройстве тензора
        core = (
            torch.rand(
                core_shape, device=tensor.device, dtype=tensor.dtype, generator=gen
            )
            + 0.01
        )

        factors = [
            torch.rand(
                (tensor.shape[mode], rank[index]),
                device=tensor.device,
                dtype=tensor.dtype,
                generator=gen,
            )
            for index, mode in enumerate(modes)
        ]
    else:
        core, factors = init

    if non_negative:
        # Используем in-place abs_() для экономии памяти
        core.abs_()
        for f in factors:
            f.abs_()

    return core, factors


def tucker_mode_dot(tensor, matrix, mode):
    """Аналог tl.mode_dot для PyTorch"""
    shape = list(tensor.shape)
    # Переносим целевую моду вперед, умножаем как матрицу, возвращаем обратно
    new_shape = shape[:]
    new_shape[mode] = matrix.shape[0]

    unfolded = tensor.moveaxis(mode, 0).reshape(shape[mode], -1)
    res = matrix @ unfolded
    return res.reshape(
        [matrix.shape[0]] + [s for i, s in enumerate(shape) if i != mode]
    ).moveaxis(0, mode)


def multi_mode_dot(tensor, factors, modes=None, transpose=False):
    if modes is None:
        modes = list(range(len(factors)))

    ndims = tensor.ndim
    # Символы для осей тензора: a, b, c, d...
    tensor_indices = [chr(ord("a") + i) for i in range(ndims)]

    # Подготавливаем индексы для факторов и итогового тензора
    factor_indices = []
    output_indices = list(tensor_indices)

    for i, mode in enumerate(modes):
        # Новый символ для размерности после умножения
        new_dim_char = chr(ord("z") - i)

        if transpose:
            # Матрица (R, I), тензор индекс I -> итог R
            # 'Ri'
            factor_indices.append(f"{new_dim_char}{tensor_indices[mode]}")
        else:
            # Матрица (I, R), тензор индекс I -> итог R
            # 'iR'
            factor_indices.append(f"{tensor_indices[mode]}{new_dim_char}")

        output_indices[mode] = new_dim_char

    # Формируем строку: "abcd,ia,jb->ijcd"
    einsum_str = f"{''.join(tensor_indices)},{','.join(factor_indices)}->{''.join(output_indices)}"

    return torch.einsum(einsum_str, tensor, *factors)


def partial_tucker(
    tensor,
    rank,
    modes=None,
    n_iter_max=100,
    init="svd",
    tol=10e-5,
    svd="truncated_svd",
    random_state=None,
    verbose=False,
    mask=None,
    svd_mask_repeats=5,
):
    if modes is None:
        modes = list(range(tensor.ndim))

    # Обработка рангов
    if rank is None:
        rank = [tensor.shape[mode] for mode in modes]
    elif isinstance(rank, int):
        rank = tuple(rank for _ in modes)
    else:
        rank = tuple(rank)

    # Инициализация (используем ранее написанную на Torch функцию)
    core, factors = initialize_tucker(
        tensor,
        rank,
        modes,
        init=init,
        svd=svd,
        random_state=random_state,
        mask=mask,
        svd_mask_repeats=svd_mask_repeats,
    )

    rec_errors = []
    # tl.norm(tensor, 2) -> torch.linalg.norm
    norm_tensor = torch.linalg.norm(tensor)

    for iteration in range(n_iter_max):
        if mask is not None:
            # Восстановление тензора для импутации маскированных значений
            reconstruction = multi_mode_dot(core, factors, modes=modes)
            tensor = tensor * mask + reconstruction * (1 - mask)

        for index, mode in enumerate(modes):
            # Проекция тензора на все факторы, кроме текущего (skip=index)
            # В Torch реализуем через последовательное применение mode_dot
            core_approximation = tensor
            for i, f_mode in enumerate(modes):
                if i != index:
                    core_approximation = tucker_mode_dot(
                        core_approximation, factors[i].mth.conj().T, f_mode
                    )

            # Развертка (unfold) и получение новых факторов через SVD
            unfolded = core_approximation.moveaxis(mode, 0).reshape(
                tensor.shape[mode], -1
            )

            eigenvecs, _, _ = svd_interface(
                unfolded,
                n_eigenvecs=rank[index],
                method=svd,
                random_state=random_state,
            )
            factors[index] = eigenvecs

        # Обновление ядра (core) проецированием на новые факторы
        core = tensor
        for i, mode in enumerate(modes):
            core = tucker_mode_dot(core, factors[i].mth.conj().T, mode)

        # Расчет ошибки: для ортонормированных факторов ||T - core x U|| = sqrt(||T||^2 - ||core||^2)
        norm_core = torch.linalg.norm(core)
        # clamp_min(0) для численной стабильности под корнем
        rec_error = (
            torch.sqrt(torch.clamp(norm_tensor**2 - norm_core**2, min=0)) / norm_tensor
        )
        rec_errors.append(rec_error.item())

        if iteration > 1:
            variation = rec_errors[-2] - rec_errors[-1]
            if verbose:
                print(
                    f"Iteration {iteration}: error={rec_errors[-1]:.6f}, variation={variation:.6f}"
                )

            if tol and abs(variation) < tol:
                if verbose:
                    print(f"Converged in {iteration} iterations.")
                break

    return (core, factors), rec_errors


def validate_tucker_rank(tensor_shape, rank="same", rounding="round", fixed_modes=None):
    r"""Returns the rank of a Tucker Decomposition

    Parameters
    ----------
    tensor_shape : tupe
        shape of the tensor to decompose
    rank : {'same', float, tuple, int}, default is same
        way to determine the rank, by default 'same'
        if 'same': rank is computed to keep the number of parameters (at most) the same
        if float, computes a rank so as to keep rank percent of the original number of parameters
        if int or tuple, just returns rank
    rounding = {'round', 'floor', 'ceil'}
    fixed_modes : int list or None, default is None
        if not None, a list of modes for which the rank will be the same as the original shape
        e.g. if i in fixed_modes, then rank[i] = tensor_shape[i]

    Returns
    -------
    rank : int tuple
        rank of the decomposition

    Notes
    -----
    For a fractional input rank, I want to find a Tucker rank such that:
    n_param_decomposition = rank*n_param_tensor

    In particular, for an input of size I_1, ..., I_N:
    I find a value c such that the rank will be (c I_1, ..., c I_N)

    We have sn_param_tensor = I_1 x ... x I_N

    We look for a Tucker decomposition of rank (c I_1, ..., c I_N )
    This decomposition will have the following n_params:
    For the core : \prod_k c I_k = c^N \prod I_k = c^N n_param_tensor
    For the factors : \sum_k c I_k^2

    In other words we want to solve:
    c^N n_param_tensor + \sum_k c I_k^2 = rank*n_param_tensor
    """
    if rounding == "ceil":
        rounding_fun = np.ceil
    elif rounding == "floor":
        rounding_fun = np.floor
    elif rounding == "round":
        rounding_fun = np.round
    else:
        raise ValueError(f"Rounding should be round, floor or ceil, but got {rounding}")

    # rank is 'same' or float: choose rank so as to preserve a fraction of the original #parameters
    if rank == "same":
        rank = float(1)

    if isinstance(rank, float):
        n_modes_compressed = len(tensor_shape)
        n_param_tensor = np.prod(tensor_shape)

        if fixed_modes is not None:
            tensor_shape = list(tensor_shape)

            # sorted to be careful with the order when popping and reinserting to not remove/add at wrong index.
            # list (mode, shape) that we removed as they will be kept the same, rank[i] =
            fixed_modes = [
                (mode, tensor_shape.pop(mode))
                for mode in sorted(fixed_modes, reverse=True)
            ][::-1]

            # number of parameters coming from the fixed modes (these don't have a variable size as a fun of fraction_param)
            n_fixed_params = np.sum(
                [s**2 for _, s in fixed_modes]
            )  # size of the factors
            n_modes_compressed -= len(fixed_modes)
        else:
            n_fixed_params = 0

        # Doesn't contain fixed_modes, those factors are accounted for in fixed_params
        squared_dims = np.sum([s**2 for s in tensor_shape])

        fun = (
            lambda x: n_param_tensor * x**n_modes_compressed
            + squared_dims * x
            + n_fixed_params * x
            - rank * n_param_tensor
        )
        fraction_param = brentq(fun, 0.0, max(rank, 1.0))
        rank = [max(int(rounding_fun(s * fraction_param)), 1) for s in tensor_shape]

        if fixed_modes is not None:
            for mode, size in fixed_modes:
                rank.insert(mode, size)

    elif isinstance(rank, int):
        n_modes = len(tensor_shape)
        message = f"Given only one int for 'rank' for decomposition a tensor of order {n_modes}. Using this rank for all modes."
        warnings.warn(message, RuntimeWarning)
        if fixed_modes is None:
            rank = [rank] * n_modes
        else:
            rank = [
                rank if i not in fixed_modes else s
                for (i, s) in enumerate(tensor_shape)
            ]  # *n_mode

    return rank


def tucker(
    tensor,
    rank,
    fixed_factors=None,
    n_iter_max=100,
    init="svd",
    return_errors=False,
    svd="truncated_svd",
    tol=10e-5,
    random_state=None,
    mask=None,
    verbose=False,
):
    """Tucker decomposition via Higher Order Orthogonal Iteration (HOI)

        Decomposes `tensor` into a Tucker decomposition:
        ``tensor = [| core; factors[0], ...factors[-1] |]`` [1]_

    Parameters
    ----------
    tensor : ndarray
    rank : None, int or int list
        size of the core tensor, ``(len(ranks) == tensor.ndim)``
        if int, the same rank is used for all modes
    fixed_factors : int list or None, default is None
        if not None, list of modes for which to keep the factors fixed.
        Only valid if a Tucker tensor is provided as init.
    n_iter_max : int
                 maximum number of iteration
    init : {'svd', 'random'}, optional
    return_errors : boolean
        Indicates whether the algorithm should return all reconstruction errors
        and computation time of each iteration or not
        Default: False
    svd : str, default is 'truncated_svd'
        function to use to compute the SVD,
        acceptable values in tensorly.SVD_FUNS
    tol : float, optional
          tolerance: the algorithm stops when the variation in
          the reconstruction error is less than the tolerance
    random_state : {None, int, np.random.RandomState}
    mask : ndarray
        array of booleans with the same shape as ``tensor`` should be 0 where
        the values are missing and 1 everywhere else. Note:  if tensor is
        sparse, then mask should also be sparse with a fill value of 1 (or
        True).
    verbose : int, optional
        level of verbosity

    Returns
    -------
    core : ndarray of size `ranks`
            core tensor of the Tucker decomposition
    factors : ndarray list
            list of factors of the Tucker decomposition.
            Its ``i``-th element is of shape ``(tensor.shape[i], ranks[i])``

    References
    ----------
    .. [1] tl.G.Kolda and B.W.Bader, "Tensor Decompositions and Applications",
       SIAM REVIEW, vol. 51, n. 3, pp. 455-500, 2009.
    """
    if fixed_factors:
        try:
            core, factors = init
        except:
            raise ValueError(
                f'Got fixed_factor={fixed_factors} but no appropriate Tucker tensor was passed for "init".'
            )

        fixed_factors = sorted(fixed_factors)
        modes_fixed, factors_fixed = zip(
            *[(i, f) for (i, f) in enumerate(factors) if i in fixed_factors]
        )
        core = multi_mode_dot(core, factors_fixed, modes=modes_fixed)
        modes, factors = zip(
            *[(i, f) for (i, f) in enumerate(factors) if i not in fixed_factors]
        )
        init = (core, list(factors))

        (core, new_factors), rec_errors = partial_tucker(
            tensor,
            rank=rank,
            modes=modes,
            n_iter_max=n_iter_max,
            init=init,
            svd=svd,
            tol=tol,
            random_state=random_state,
            mask=mask,
            verbose=verbose,
        )

        factors = list(new_factors)
        for i, e in enumerate(fixed_factors):
            factors.insert(e, factors_fixed[i])
        core = multi_mode_dot(core, factors_fixed, modes=modes_fixed, transpose=True)

        return TuckerTensor((core, factors))

    else:
        modes = list(range(tensor.ndim))
        # TO-DO validate rank for partial tucker as well
        rank = validate_tucker_rank(tensor.shape, rank=rank)

        (core, factors), rec_errors = partial_tucker(
            tensor,
            rank=rank,
            modes=modes,
            n_iter_max=n_iter_max,
            init=init,
            svd=svd,
            tol=tol,
            random_state=random_state,
            mask=mask,
            verbose=verbose,
        )
        tensor = TuckerTensor((core, factors))
        if return_errors:
            return tensor, rec_errors
        else:
            return tensor


class FactorList(nn.Module):
    def __init__(self, parameters=None):
        super().__init__()
        self.keys = []
        self.counter = 0
        if parameters is not None:
            self.extend(parameters)

    def _unique_key(self):
        """Creates a new unique key"""
        key = f"factor_{self.counter}"
        self.counter += 1
        return key

    def append(self, element):
        key = self._unique_key()
        if torch.is_tensor(element):
            if isinstance(element, nn.Parameter):
                self.register_parameter(key, element)
            else:
                self.register_buffer(key, element)
        else:
            setattr(self, key, self.__class__(element))
        self.keys.append(key)

    def insert(self, index, element):
        key = self._unique_key()
        setattr(self, key, element)
        self.keys.insert(index, key)

    def pop(self, index=-1):
        item = self[index]
        self.__delitem__(index)
        return item

    def __getitem__(self, index):
        keys = self.keys[index]
        if isinstance(keys, list):
            return self.__class__([getattr(self, key) for key in keys])
        return getattr(self, keys)

    def __setitem__(self, index, value):
        setattr(self, self.keys[index], value)

    def __delitem__(self, index):
        delattr(self, self.keys[index])
        self.keys.__delitem__(index)

    def __len__(self):
        return len(self.keys)

    def extend(self, parameters):
        for param in parameters:
            self.append(param)

    def __iadd__(self, parameters):
        return self.extend(parameters)

    def __add__(self, parameters):
        instance = self.__class__(self)
        instance.extend(parameters)
        return instance

    def __radd__(self, parameters):
        instance = self.__class__(parameters)
        instance.extend(self)
        return instance

    def extra_repr(self) -> str:
        child_lines = []
        for k, p in self._parameters.items():
            size_str = "x".join(str(size) for size in p.size())
            device_str = "" if not p.is_cuda else " (GPU {})".format(p.get_device())
            parastr = "Parameter containing: [{} of size {}{}]".format(
                torch.typename(p), size_str, device_str
            )
            child_lines.append("  (" + str(k) + "): " + parastr)
        tmpstr = "\n".join(child_lines)
        return tmpstr


class DenseTensor(FactorizedTensor, name="Dense"):
    """Dense tensor"""

    def __init__(self, tensor, shape=None, rank=None):
        super().__init__()
        if shape is not None and rank is not None:
            self.shape, self.rank = shape, rank
        else:
            self.shape = tensor.shape
            self.rank = None
        self.order = len(self.shape)

        if isinstance(tensor, nn.Parameter):
            self.register_parameter("tensor", tensor)
        else:
            self.register_buffer("tensor", tensor)

    @classmethod
    def new(cls, shape, rank=None, device=None, dtype=None, **kwargs):
        # Register the parameters
        tensor = nn.Parameter(torch.empty(shape, device=device, dtype=dtype))

        return cls(tensor)

    @classmethod
    def from_tensor(cls, tensor, rank="same", **kwargs):
        # В PyTorch принято клонировать тензор перед созданием параметра,
        # чтобы избежать нежелательных побочных эффектов.
        return cls(nn.Parameter(tensor.clone()))

    def init_from_tensor(self, tensor, l2_reg=1e-5, **kwargs):
        # torch.no_grad() гарантирует, что операция копирования
        # не будет отслеживаться градиентами.
        with torch.no_grad():
            # tensor.clone() заменяет tl.copy(tensor)
            self.tensor = nn.Parameter(tensor.clone())
        return self

    @property
    def decomposition(self):
        return self.tensor

    def to_tensor(self):
        return self.tensor

    def normal_(self, mean=0, std=1):
        with torch.no_grad():
            self.tensor.data.normal_(mean, std)
        return self

    def __getitem__(self, indices):
        return self.__class__(self.tensor[indices])


class CPTensor(FactorizedTensor, name="CP"):
    """CP Factorization

    Parameters
    ----------
    weights
    factors
    shape
    rank
    """

    def __init__(self, weights, factors, shape=None, rank=None):
        super().__init__()
        if shape is not None and rank is not None:
            self.shape, self.rank = shape, rank
        else:
            self.shape, self.rank = _validate_cp_tensor((weights, factors))
        self.order = len(self.shape)

        # self.weights = weights
        if isinstance(weights, nn.Parameter):
            self.register_parameter("weights", weights)
        else:
            self.register_buffer("weights", weights)

        self.factors = FactorList(factors)

    @classmethod
    def new(cls, shape, rank, device=None, dtype=None, **kwargs):
        rank = validate_cp_rank(shape, rank)

        # Register the parameters
        weights = nn.Parameter(torch.empty(rank, device=device, dtype=dtype))
        # Avoid the issues with ParameterList
        factors = [
            nn.Parameter(torch.empty((s, rank), device=device, dtype=dtype))
            for s in shape
        ]

        return cls(weights, factors)

    @classmethod
    def from_tensor(cls, tensor, rank="same", **kwargs):
        shape = tensor.shape
        rank = validate_cp_rank(shape, rank)
        dtype = tensor.dtype

        with torch.no_grad():
            weights, factors = parafac(tensor.to(torch.float64), rank, **kwargs)

        return cls(
            nn.Parameter(weights.to(dtype).contiguous()),
            [nn.Parameter(f.to(dtype).contiguous()) for f in factors],
        )

    def init_from_tensor(self, tensor, l2_reg=1e-5, **kwargs):
        with torch.no_grad():
            weights, factors = parafac(tensor, self.rank, l2_reg=l2_reg, **kwargs)

        self.weights = nn.Parameter(weights.contiguous())
        self.factors = FactorList([nn.Parameter(f.contiguous()) for f in factors])
        return self

    @property
    def decomposition(self):
        return self.weights, self.factors

    def to_tensor(self):
        return cp_to_tensor(self.decomposition)

    def normal_(self, mean=0, std=1):
        super().normal_(mean, std)
        std_factors = (std / math.sqrt(self.rank)) ** (1 / self.order)

        with torch.no_grad():
            self.weights.fill_(1)
            for factor in self.factors:
                factor.data.normal_(0, std_factors)
        return self

    def __getitem__(self, indices):
        if isinstance(indices, int):
            # Select one dimension of one mode
            mixing_factor, *factors = self.factors
            weights = self.weights * mixing_factor[indices, :]
            return self.__class__(weights, factors)

        elif isinstance(indices, slice):
            # Index part of a factor
            mixing_factor, *factors = self.factors
            factors = [mixing_factor[indices, :], *factors]
            weights = self.weights
            return self.__class__(weights, factors)

        else:
            # Index multiple dimensions
            factors = self.factors
            index_factors = []
            weights = self.weights
            for index in indices:
                if index is Ellipsis:
                    raise ValueError(
                        f"Ellipsis is not yet supported, yet got indices={indices} which contains one."
                    )

                mixing_factor, *factors = factors
                if isinstance(index, (np.integer, int)):
                    if factors or index_factors:
                        weights = weights * mixing_factor[index, :]
                    else:
                        # No factors left
                        return torch.sum(weights * mixing_factor[index, :])
                else:
                    index_factors.append(mixing_factor[index, :])

            return self.__class__(weights, index_factors + factors)
        # return self.__class__(*tl.cp_indexing(self.weights, self.factors, indices))

    def transduct(self, new_dim, mode=0, new_factor=None):
        """Transduction adds a new dimension to the existing factorization

        Parameters
        ----------
        new_dim : int
            dimension of the new mode to add
        mode : where to insert the new dimension, after the channels, default is 0
            by default, insert the new dimensions before the existing ones
            (e.g. add time before height and width)

        Returns
        -------
        self
        """
        factors = self.factors
        # Important: don't increment the order before accessing factors which uses order!
        self.order += 1
        self.shape = self.shape[:mode] + (new_dim,) + self.shape[mode:]

        if new_factor is None:
            new_factor = torch.ones(new_dim, self.rank)  # /new_dim

        factors.insert(
            mode, nn.Parameter(new_factor.to(factors[0].device).contiguous())
        )
        self.factors = FactorList(factors)

        return self


def random_cp(shape, rank, device=None, dtype=None, generator=None, **kwargs):
    # Создаем факторы (матрицы) для каждой моды тензора
    factors = [
        torch.randn((s, rank), device=device, dtype=dtype, generator=generator)
        for s in shape
    ]
    # В CP-разложении также обычно есть веса (weights)
    weights = torch.ones(rank, device=device, dtype=dtype)
    return weights, factors


def _validate_tucker_tensor(tucker_tensor):
    core, factors = tucker_tensor

    if len(factors) < 2:
        raise ValueError(
            f"A Tucker tensor should be composed of at least two factors and a core. "
            f"However, {len(factors)} factor(s) were given."
        )

    if len(factors) != core.ndim:
        raise ValueError(
            f"Tucker decompositions should have one factor per mode of the core. "
            f"However, core has {core.ndim} modes but {len(factors)} factors provided."
        )

    # ПРОВЕРКА УСТРОЙСТВА И ТИПА ДАННЫХ
    # Берем эталонные значения у ядра
    device = core.device
    dtype = core.dtype

    shape = []
    rank = []
    for i, factor in enumerate(factors):
        # Проверка девайса
        if factor.device != device:
            raise ValueError(
                f"Device mismatch: core is on {device}, but factors[{i}] is on {factor.device}. "
                "All tensors must be on the same device."
            )

        # Проверка типа данных (опционально, но полезно)
        if factor.dtype != dtype:
            raise ValueError(
                f"Dtype mismatch: core is {dtype}, but factors[{i}] is {factor.dtype}. "
                "All tensors must have the same dtype."
            )

        current_shape, current_rank = factor.shape

        if current_rank != core.shape[i]:
            raise ValueError(
                f"Factor {i} rank mismatch: factors[{i}].shape[1]={current_rank} "
                f"but core.shape[{i}]={core.shape[i]}."
            )

        shape.append(current_shape)
        rank.append(current_rank)

    return tuple(shape), tuple(rank)


def tucker_to_tensor(tucker_tensor, skip_factor=None, transpose_factors=False):
    core, factors = tucker_tensor

    # Определяем, какие моды и факторы мы используем
    if skip_factor is not None:
        # Индексы мод, которые НЕ пропускаем
        modes = [i for i in range(len(factors)) if i != skip_factor]
        # Сами факторы, которые НЕ пропускаем
        active_factors = [f for i, f in enumerate(factors) if i != skip_factor]
    else:
        modes = list(range(len(factors)))
        active_factors = factors

    # Вызываем нашу multi_mode_dot на базе einsum
    return multi_mode_dot(
        core, active_factors, modes=modes, transpose=transpose_factors
    )


class TuckerTensor(FactorizedTensor, name="Tucker"):
    """Tucker Factorization

    Parameters
    ----------
    core
    factors
    shape
    rank
    """

    def __init__(self, core, factors, shape=None, rank=None):
        super().__init__()
        if shape is not None and rank is not None:
            self.shape, self.rank = shape, rank
        else:
            self.shape, self.rank = _validate_tucker_tensor((core, factors))

        self.order = len(self.shape)
        # self.core = core
        if isinstance(core, nn.Parameter):
            self.register_parameter("core", core)
        else:
            self.register_buffer("core", core)

        self.factors = FactorList(factors)

    @classmethod
    def new(cls, shape, rank, fixed_rank_modes=None, device=None, dtype=None, **kwargs):
        rank = validate_tucker_rank(shape, rank, fixed_modes=fixed_rank_modes)

        # Register the parameters
        core = nn.Parameter(torch.empty(rank, device=device, dtype=dtype))
        # Avoid the issues with ParameterList
        factors = [
            nn.Parameter(torch.empty((s, r), device=device, dtype=dtype))
            for (s, r) in zip(shape, rank)
        ]

        return cls(core, factors)

    @classmethod
    def from_tensor(cls, tensor, rank="same", fixed_rank_modes=None, **kwargs):
        shape = tensor.shape
        rank = validate_tucker_rank(shape, rank, fixed_modes=fixed_rank_modes)

        with torch.no_grad():
            core, factors = tucker(tensor, rank, **kwargs)

        return cls(
            nn.Parameter(core.contiguous()),
            [nn.Parameter(f.contiguous()) for f in factors],
        )

    def init_from_tensor(
        self, tensor, unsqueezed_modes=None, unsqueezed_init="average", **kwargs
    ):
        """Initialize the tensor factorization from a tensor

        Parameters
        ----------
        tensor : torch.Tensor
            full tensor to decompose
        unsqueezed_modes : int list
            list of modes for which the rank is 1 that don't correspond to a mode in the full tensor
            essentially we are adding a new dimension for which the core has dim 1,
            and that is not initialized through decomposition.
            Instead first `tensor` is decomposed into the other factors.
            The `unsqueezed factors` are then added and  initialized e.g. with 1/dim[i]
        unsqueezed_init : 'average' or float
            if unsqueezed_modes, this is how the added "unsqueezed" factors will be initialized
            if 'average', then unsqueezed_factor[i] will have value 1/tensor.shape[i]
        """
        if unsqueezed_modes is not None:
            unsqueezed_modes = sorted(unsqueezed_modes)
            for mode in unsqueezed_modes[::-1]:
                if self.rank[mode] != 1:
                    msg = "It is only possible to initialize by averagig over mode for which rank=1."
                    msg += f"However, got unsqueezed_modes={unsqueezed_modes} but rank[{mode}]={self.rank[mode]} != 1."
                    raise ValueError(msg)

            rank = tuple(
                r for (i, r) in enumerate(self.rank) if i not in unsqueezed_modes
            )
        else:
            rank = self.rank

        with torch.no_grad():
            core, factors = tucker(tensor, rank, **kwargs)

            if unsqueezed_modes is not None:
                # Initialise with 1/shape[mode] or given value
                for mode in unsqueezed_modes:
                    size = self.shape[mode]
                    factor = torch.ones(size, 1)
                    if unsqueezed_init == "average":
                        factor /= size
                    else:
                        factor *= unsqueezed_init
                    factors.insert(mode, factor)
                    core = core.unsqueeze(mode)

        self.core = nn.Parameter(core.contiguous())
        self.factors = FactorList([nn.Parameter(f.contiguous()) for f in factors])
        return self

    @property
    def decomposition(self):
        return self.core, self.factors

    def to_tensor(self):
        return tucker_to_tensor(self.decomposition)

    def normal_(self, mean=0, std=1):
        if mean != 0:
            raise ValueError(f"Currently only mean=0 is supported, but got mean={mean}")

        r = np.prod([math.sqrt(r) for r in self.rank])
        std_factors = (std / r) ** (1 / (self.order + 1))

        with torch.no_grad():
            self.core.data.normal_(0, std_factors)
            for factor in self.factors:
                factor.data.normal_(0, std_factors)
        return self

    def __getitem__(self, indices):
        if isinstance(indices, int):
            # Выбор одного измерения первой моды
            mixing_factor, *factors = self.factors
            # Вместо tenalg.mode_dot используем нашу tucker_mode_dot
            # Берем строку mixing_factor[indices, :] и сворачиваем с ядром по 0-й моде
            core = tucker_mode_dot(self.core, mixing_factor[indices, :].unsqueeze(0), 0)
            # Убираем лишнюю размерность, так как это выбор индекса (int)
            core = core.squeeze(0)
            return self.__class__(core, factors)

        elif isinstance(indices, slice):
            # Срез первой моды
            mixing_factor, *factors = self.factors
            factors = [mixing_factor[indices, :], *factors]
            return self.__class__(self.core, factors)

        else:
            # Индексация по нескольким модам
            factors_remaining = []
            factors_contract = []
            modes_to_contract = []

            # Разделяем индексы на те, что схлопывают размерность (int),
            # и те, что сохраняют (slice)
            for i, index in enumerate(indices):
                if index is Ellipsis:
                    raise ValueError("Ellipsis is not supported.")

                current_factor = self.factors[i]

                if isinstance(index, int):
                    modes_to_contract.append(i)
                    # Извлекаем вектор и превращаем в строку (1, rank) для умножения
                    factors_contract.append(current_factor[index, :].unsqueeze(0))
                else:
                    # Оставляем срез фактора
                    factors_remaining.append(current_factor[index, :])

            if modes_to_contract:
                # Используем нашу multi_mode_dot (через einsum),
                # чтобы применить факторы-векторы к ядру
                core = multi_mode_dot(
                    self.core, factors_contract, modes=modes_to_contract
                )
                # Убираем размерности, которые схлопнулись из-за int-индексов
                # squeeze(modes) удалит только те оси, по которым прошли int
                core = core.squeeze(tuple(modes_to_contract))
            else:
                core = self.core

            # Добавляем факторы от мод, которые не были затронуты индексами
            factors_remaining = factors_remaining + self.factors[len(indices) :]

            if factors_remaining:
                return self.__class__(core, factors_remaining)

            # Если все моды схлопнуты, возвращаем ядро (теперь это скаляр или вектор)
            return core


def validate_tt_rank(
    tensor_shape,
    rank="same",
    constant_rank=False,
    rounding="round",
    allow_overparametrization=True,
):
    # Заменяем np функции на стандартные или torch
    if rounding == "ceil":
        rounding_fun = math.ceil
    elif rounding == "floor":
        rounding_fun = math.floor
    elif rounding == "round":
        rounding_fun = round
    else:
        raise ValueError(f"Rounding should be round, floor or ceil, but got {rounding}")

    if rank == "same":
        rank = float(1)

    if isinstance(rank, float) and constant_rank:
        # np.prod -> math.prod (доступен в Python 3.8+)
        n_param_tensor = math.prod(tensor_shape) * rank
        order = len(tensor_shape)

        if order == 2:
            rank = (1, int(n_param_tensor / (tensor_shape[0] + tensor_shape[1])), 1)
            warnings.warn(f"Determining tt-rank for a matrix: {tensor_shape}")

        a = sum(tensor_shape[1:-1])
        b = sum([tensor_shape[0], tensor_shape[-1]])
        c = -n_param_tensor
        delta = math.sqrt(b**2 - 4 * a * c)

        solution = int(rounding_fun((-b + delta) / (2 * a)))
        rank = (1,) + (solution,) * (order - 1) + (1,)

    elif isinstance(rank, float):
        order = len(tensor_shape)
        avg_dim = [
            (tensor_shape[i] + tensor_shape[i + 1]) / 2 for i in range(order - 1)
        ]

        if len(avg_dim) > 1:
            a = sum(
                avg_dim[i - 1] * tensor_shape[i] * avg_dim[i]
                for i in range(1, order - 1)
            )
        else:
            warnings.warn(f"Determining tt-rank for a matrix: {tensor_shape}")
            a = avg_dim[0] ** 2 * tensor_shape[0]

        b = tensor_shape[0] * avg_dim[0] + tensor_shape[-1] * avg_dim[-1]
        c = -math.prod(tensor_shape) * rank
        delta = math.sqrt(b**2 - 4 * a * c)

        fraction_param = (-b + delta) / (2 * a)
        rank = tuple([max(int(rounding_fun(d * fraction_param)), 1) for d in avg_dim])
        rank = (1,) + rank + (1,)

    else:
        n_dim = len(tensor_shape)
        if isinstance(rank, int):
            rank = [1] + [rank] * (n_dim - 1) + [1]
        elif n_dim + 1 != len(rank):
            raise ValueError(
                f"Incorrect rank length. Expected {n_dim+1}, got {len(rank)}"
            )

        if rank[0] != 1 or rank[-1] != 1:
            raise ValueError("Boundary conditions dictate rank[0] == rank[-1] == 1.")

    if allow_overparametrization:
        return list(rank)
    else:
        validated_rank = [1]
        for i, s in enumerate(tensor_shape[:-1]):
            n_row = int(rank[i] * s)
            # Заменяем np.prod на math.prod для среза формы
            n_column = math.prod(tensor_shape[(i + 1) :])
            validated_rank.append(min(n_row, n_column, rank[i + 1]))
        validated_rank.append(1)

        return validated_rank


def tt_to_tensor(factors):
    if isinstance(factors, (float, int)):  # случай скаляра
        return factors

    # Извлекаем размерности исходного тензора (средняя ось каждого ядра)
    full_shape = [f.shape[1] for f in factors]

    # Первая развертка: (1, I_1, R_1) -> (I_1, R_1)
    full_tensor = factors[0].reshape(full_shape[0], -1)

    for factor in factors[1:]:
        rank_prev, mode_size, rank_next = factor.shape

        # Развертка текущего ядра в матрицу (R_{k-1}, I_k * R_k)
        factor_matrix = factor.reshape(rank_prev, -1)

        # Матричное умножение: (I_1*...*I_{k-1}, R_{k-1}) @ (R_{k-1}, I_k * R_k)
        full_tensor = full_tensor @ factor_matrix

        # Перегруппировка для следующей итерации: выносим R_k в конец
        full_tensor = full_tensor.reshape(-1, rank_next)

    # Итоговый reshape к многомерному тензору
    return full_tensor.reshape(full_shape)


def tensor_train(input_tensor, rank, svd="truncated_svd", verbose=False):
    # Используем вашу адаптированную функцию validate_tt_rank
    rank = validate_tt_rank(input_tensor.shape, rank=rank)
    tensor_size = input_tensor.shape
    n_dim = len(tensor_size)

    unfolding = input_tensor
    factors = [None] * n_dim

    # Основной цикл рекурсивного SVD
    for k in range(n_dim - 1):
        # Формируем матрицу для разложения
        n_row = int(rank[k] * tensor_size[k])
        unfolding = unfolding.reshape(n_row, -1)

        # Вычисляем SVD через наш интерфейс
        n_row, n_column = unfolding.shape
        current_rank = min(n_row, n_column, rank[k + 1])

        # svd_interface на Torch должен возвращать (U, S, V)
        U, S, V = svd_interface(unfolding, n_eigenvecs=current_rank, method=svd)

        # Обновляем ранг (на случай, если SVD нашел меньше компонент)
        rank[k + 1] = S.shape[0]

        # Формируем k-ое ядро TT-поезда (TT-core)
        factors[k] = U.reshape(rank[k], tensor_size[k], rank[k + 1])

        if verbose:
            print(f"TT factor {k} computed with shape {factors[k].shape}")

        # Подготовка матрицы для следующего шага
        # В Torch: масштабируем строки V значениями S через broadcasting
        # S.view(-1, 1) превращает вектор в столбец
        unfolding = S.view(-1, 1) * V

    # Формируем последнее ядро (всегда имеет ранг 1 на выходе)
    prev_rank, last_dim = unfolding.shape
    factors[-1] = unfolding.reshape(prev_rank, last_dim, 1)

    if verbose:
        print(f"TT factor {n_dim - 1} computed with shape {factors[-1].shape}")

    # Возвращаем ваш класс TTTensor (передаем список тензоров Torch)
    return TTTensor(factors)


def validate_tt_tensor(tt_tensor):
    # Если это уже объект вашего класса TTTensor
    if hasattr(tt_tensor, "factors"):
        return tt_tensor.shape, tt_tensor.rank

    # Случай скаляра (0-й порядок)
    if isinstance(tt_tensor, (float, int)):
        return (), (1, 1)

    factors = tt_tensor
    n_factors = len(factors)

    # Берем эталонное устройство и тип данных у первого ядра
    device = factors[0].device
    dtype = factors[0].dtype

    rank = []
    shape = []

    for index, factor in enumerate(factors):
        # Проверка девайса и типа
        if factor.device != device or factor.dtype != dtype:
            raise ValueError(
                f"Context mismatch: factors[{index}] is on {factor.device}/{factor.dtype}, "
                f"but factors[0] is on {device}/{dtype}."
            )

        # В Torch .shape возвращает кортеж-подобный объект
        if factor.ndim != 3:
            raise ValueError(
                f"TT-cores must be 3rd order tensors. "
                f"However, factors[{index}].ndim = {factor.ndim}."
            )

        current_rank, current_shape, next_rank = factor.shape

        # Проверка согласованности рангов (R_k)
        if index > 0:
            prev_next_rank = factors[index - 1].shape[2]
            if prev_next_rank != current_rank:
                raise ValueError(
                    f"Rank mismatch: factors[{index-1}].shape[2] ({prev_next_rank}) "
                    f"must equal factors[{index}].shape[0] ({current_rank})."
                )

        # Граничные условия (R_0 = 1 и R_N = 1)
        if index == 0 and current_rank != 1:
            raise ValueError(f"First rank must be 1, but got {current_rank}.")

        if index == n_factors - 1 and next_rank != 1:
            raise ValueError(f"Last rank must be 1, but got {next_rank}.")

        shape.append(current_shape)
        rank.append(current_rank)

    # Добавляем финальный ранг (1)
    rank.append(next_rank)

    return tuple(shape), tuple(rank)


class TTTensor(FactorizedTensor, name="TT"):
    """Tensor-Train (Matrix-Product-State) Factorization

    Parameters
    ----------
    factors
    shape
    rank
    """

    def __init__(self, factors, shape=None, rank=None):
        super().__init__()
        if shape is None or rank is None:
            self.shape, self.rank = validate_tt_tensor(factors)
        else:
            self.shape, self.rank = shape, rank

        self.order = len(self.shape)
        self.factors = FactorList(factors)

    @classmethod
    def new(cls, shape, rank, device=None, dtype=None, **kwargs):
        rank = validate_tt_rank(shape, rank)

        # Avoid the issues with ParameterList
        factors = [
            nn.Parameter(
                torch.empty((rank[i], s, rank[i + 1]), device=device, dtype=dtype)
            )
            for i, s in enumerate(shape)
        ]

        return cls(factors)

    @classmethod
    def from_tensor(cls, tensor, rank="same", **kwargs):
        shape = tensor.shape
        rank = validate_tt_rank(shape, rank)

        with torch.no_grad():
            # TODO: deal properly with wrong kwargs
            factors = tensor_train(tensor, rank)

        return cls([nn.Parameter(f.contiguous()) for f in factors])

    def init_from_tensor(self, tensor, **kwargs):
        with torch.no_grad():
            # TODO: deal properly with wrong kwargs
            factors = tensor_train(tensor, self.rank)

        self.factors = FactorList([nn.Parameter(f.contiguous()) for f in factors])
        self.rank = tuple([f.shape[0] for f in factors] + [1])
        return self

    @property
    def decomposition(self):
        return self.factors

    def to_tensor(self):
        return tt_to_tensor(self.decomposition)

    def normal_(self, mean=0, std=1):
        if mean != 0:
            raise ValueError(f"Currently only mean=0 is supported, but got mean={mean}")

        r = np.prod(self.rank)
        std_factors = (std / r) ** (1 / self.order)
        with torch.no_grad():
            for factor in self.factors:
                factor.data.normal_(0, std_factors)
        return self

    def __getitem__(self, indices):
        if isinstance(indices, int):
            # Выбор одного измерения первой моды
            factor, next_factor, *factors = self.factors
            # factor[:, indices, :] -> срез. squeeze(1) убирает выбранную ось.
            # tenalg.mode_dot(next_factor, ..., 0) -> tucker_mode_dot
            contracted = (
                factor[:, indices, :].squeeze(0)
                if factor.ndim == 2
                else factor[indices, :]
            )
            # В ТТ-формате это обычно перемножение матриц узлов
            next_factor = torch.matmul(
                contracted, next_factor.moveaxis(0, 0)
            )  # Уточните логику ТТ
            return self.__class__([next_factor, *factors])

        elif isinstance(indices, slice):
            mixing_factor, *factors = self.factors
            # Срез первой моды
            factors = [mixing_factor[indices, :], *factors]
            return self.__class__(factors)

        else:
            # Сложная индексация по нескольким модам
            factors_list = []
            all_contracted = True

            for i, index in enumerate(indices):
                if index is Ellipsis:
                    raise ValueError(
                        "Ellipsis is not yet supported in this implementation."
                    )

                current_factor = self.factors[i]

                if isinstance(index, int):
                    # Если индекс - число, мы "схлопываем" (contract) эту моду
                    # current_factor[index] даст матрицу (rank_in, rank_out)
                    if i == 0:
                        running_factor = current_factor[index]
                    else:
                        # Матричное умножение накопленного результата на текущий срез
                        running_factor = running_factor @ current_factor[index]
                else:
                    # Если индекс - срез, мода сохраняется
                    sliced = current_factor[index, :]
                    if i == 0:
                        running_factor = sliced
                    else:
                        if all_contracted:
                            # Если до этого были только int, перемножаем накопленную матрицу на срез
                            # (rank_prev) @ (rank_prev, new_dim, rank_next)
                            running_factor = torch.matmul(
                                running_factor, sliced.moveaxis(0, 0)
                            )
                        else:
                            factors_list.append(running_factor)
                            running_factor = sliced
                    all_contracted = False

            # Финальная сборка
            remaining = self.factors[i + 1 :]
            if (
                running_factor.ndim == 2
            ):  # Результат - матрица (все индексы были int или последний)
                if not remaining:
                    return running_factor.squeeze()
                else:
                    next_f = remaining[0]
                    # Сворачиваем накопленную матрицу со следующим узлом
                    combined = torch.matmul(running_factor, next_f.moveaxis(0, 0))
                    return self.__class__([combined, *remaining[1:]])
            else:
                return self.__class__([*factors_list, running_factor, *remaining])

    def transduct(self, new_dim, mode=0, new_factor=None):
        """Transduction adds a new dimension to the existing factorization

        Parameters
        ----------
        new_dim : int
            dimension of the new mode to add
        mode : where to insert the new dimension, after the channels, default is 0
            by default, insert the new dimensions before the existing ones
            (e.g. add time before height and width)

        Returns
        -------
        self
        """
        factors = self.factors

        # Important: don't increment the order before accessing factors which uses order!
        self.order += 1
        new_rank = self.rank[mode]
        self.rank = self.rank[:mode] + (new_rank,) + self.rank[mode:]
        self.shape = self.shape[:mode] + (new_dim,) + self.shape[mode:]

        # Init so the reconstruction is equivalent to concatenating the previous self new_dim times
        if new_factor is None:
            new_factor = torch.zeros(new_rank, new_dim, new_rank)
            for i in range(new_dim):
                new_factor[:, i, :] = torch.eye(new_rank)  # /new_dim
            # Below: <=> static prediciton
            # new_factor[:, new_dim//2, :] = torch.eye(new_rank)

        factors.insert(
            mode, nn.Parameter(new_factor.to(factors[0].device).contiguous())
        )
        self.factors = FactorList(factors)

        return self


def resample(x, res_scale, axis, output_shape=None):
    """
    A module for generic n-dimentional interpolation (Fourier resampling).

    Parameters
    ----------
    x : torch.Tensor
            input activation of size (batch_size, channels, d1, ..., dN)
    res_scale: int or tuple
            Scaling factor along each of the dimensions in 'axis' parameter. If res_scale is scaler, then isotropic
            scaling is performed
    axis: axis or dimensions along which interpolation will be performed.
    output_shape : None or tuple[int]
    """

    if isinstance(res_scale, (float, int)):
        if axis is None:
            axis = list(range(2, x.ndim))
            res_scale = [res_scale] * len(axis)
        elif isinstance(axis, int):
            axis = [axis]
            res_scale = [res_scale]
        else:
            res_scale = [res_scale] * len(axis)
    else:
        assert len(res_scale) == len(axis), "leght of res_scale and axis are not same"

    old_size = x.shape[-len(axis) :]
    if output_shape is None:
        new_size = tuple([int(round(s * r)) for (s, r) in zip(old_size, res_scale)])
    else:
        new_size = output_shape

    if len(axis) == 1:
        return F.interpolate(x, size=new_size[0], mode="linear", align_corners=True)
    if len(axis) == 2:
        return F.interpolate(x, size=new_size, mode="bicubic", align_corners=True)

    X = torch.fft.rfftn(x.float(), norm="forward", dim=axis)

    new_fft_size = list(new_size)
    new_fft_size[-1] = new_fft_size[-1] // 2 + 1  # Redundant last coefficient
    new_fft_size_c = [min(i, j) for (i, j) in zip(new_fft_size, X.shape[-len(axis) :])]
    out_fft = torch.zeros(
        [x.shape[0], x.shape[1], *new_fft_size], device=x.device, dtype=torch.cfloat
    )

    mode_indexing = [((None, m // 2), (-m // 2, None)) for m in new_fft_size_c[:-1]] + [
        ((None, new_fft_size_c[-1]),)
    ]
    for i, boundaries in enumerate(itertools.product(*mode_indexing)):

        idx_tuple = [slice(None), slice(None)] + [slice(*b) for b in boundaries]

        out_fft[idx_tuple] = X[idx_tuple]
    y = torch.fft.irfftn(out_fft, s=new_size, norm="forward", dim=axis)

    return y


einsum_symbols = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def einsum_complexhalf_two_input(eq, a, b):
    """
    Compute (two-input) einsum for complexhalf tensors.
    Because torch.einsum currently does not support complex32 (complexhalf) types.
    The inputs and outputs are the same as in torch.einsum
    """
    assert len(eq.split(",")) == 2, "Equation must have two inputs."

    # cast both tensors to "view as real" form, and half precision
    a = torch.view_as_real(a)
    b = torch.view_as_real(b)
    a = a.half()
    b = b.half()

    # create a new einsum equation that takes into account "view as real" form
    input_output = eq.split("->")
    new_output = "xy" + input_output[1]
    input_terms = input_output[0].split(",")
    new_inputs = [input_terms[0] + "x", input_terms[1] + "y"]
    new_eqn = new_inputs[0] + "," + new_inputs[1] + "->" + new_output

    # convert back to complex form
    tmp = torch.einsum(new_eqn, a, b)
    res = torch.stack(
        [tmp[0, 0, ...] - tmp[1, 1, ...], tmp[1, 0, ...] + tmp[0, 1, ...]], dim=-1
    )
    return torch.view_as_complex(res)


def einsum_complexhalf(eq, *args):
    """
    Compute einsum for complexhalf tensors.
    Because torch.einsum currently does not support complex32 (complexhalf) types.
    The inputs and outputs are the same as in torch.einsum
    """
    if len(args) == 2:
        # if there are two inputs, it is faster to call this method
        return einsum_complexhalf_two_input(eq, *args)

    # find the optimal path
    _, path_info = contract_path(eq, *args)
    partial_eqns = [
        contraction_info[2] for contraction_info in path_info.contraction_list
    ]

    # create a dict of the input tensors by their label in the einsum equation
    tensors = {}
    input_labels = eq.split("->")[0].split(",")
    output_label = eq.split("->")[1]
    tensors = dict(zip(input_labels, args))

    # convert all tensors to half precision and "view as real" form
    for key, tensor in tensors.items():
        tensor = torch.view_as_real(tensor)
        tensor = tensor.half()
        tensors[key] = tensor

    for partial_eq in partial_eqns:
        # get the input tensors to partial_eq
        in_labels, out_label = partial_eq.split("->")
        in_labels = in_labels.split(",")
        in_tensors = [tensors[label] for label in in_labels]

        # create new einsum equation that takes into account "view as real" form
        input_output = partial_eq.split("->")
        new_output = "xy" + input_output[1]
        input_terms = input_output[0].split(",")
        new_inputs = [input_terms[0] + "x", input_terms[1] + "y"]
        new_eqn = new_inputs[0] + "," + new_inputs[1] + "->" + new_output

        # perform the einsum, and convert to "view as real" form
        tmp = torch.einsum(new_eqn, *in_tensors)
        result = torch.stack(
            [tmp[0, 0, ...] - tmp[1, 1, ...], tmp[1, 0, ...] + tmp[0, 1, ...]], dim=-1
        )
        tensors[out_label] = result

    return torch.view_as_complex(tensors[output_label])


class AdaIN(nn.Module):
    def __init__(self, embed_dim, in_channels, mlp=None, eps=1e-5):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.eps = eps

        if mlp is None:
            mlp = nn.Sequential(
                nn.Linear(embed_dim, 512), nn.GELU(), nn.Linear(512, 2 * in_channels)
            )
        self.mlp = mlp

        self.embedding = None

    def set_embedding(self, x):
        self.embedding = x.reshape(
            self.embed_dim,
        )

    def forward(self, x):
        assert (
            self.embedding is not None
        ), "AdaIN: update embeddding before running forward"

        weight, bias = torch.split(self.mlp(self.embedding), self.in_channels, dim=0)

        return nn.functional.group_norm(x, self.in_channels, weight, bias, eps=self.eps)


class InstanceNorm(nn.Module):
    def __init__(self, **kwargs):
        """InstanceNorm applies dim-agnostic instance normalization
        to data as an nn.Module.

        kwargs: additional parameters to pass to instance_norm() for use as a module
        e.g. eps, affine
        """
        super().__init__()
        self.kwargs = kwargs

    def forward(self, x):
        size = x.shape
        x = torch.nn.functional.instance_norm(x, **self.kwargs)
        assert x.shape == size
        return x


class Flattened1dConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, bias=False):
        """Flattened3dConv is a Conv-based skip layer for
        input tensors of ndim > 3 (batch, channels, d1, ...) that flattens all dimensions
        past the batch and channel dims into one dimension, applies the Conv,
        and un-flattens.

        Parameters
        ----------
        in_channels : int
            in_channels of Conv1d
        out_channels : int
            out_channels of Conv1d
        kernel_size : int
            kernel_size of Conv1d
        bias : bool, optional
            bias of Conv3d, by default False
        """
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=bias,
        )

    def forward(self, x):
        # x.shape: b, c, x1, ..., xn x_ndim > 1
        size = list(x.shape)
        # flatten everything past 1st data dim
        x = x.view(*size[:2], -1)
        x = self.conv(x)
        # reshape x into an Nd tensor b, c, x1, x2, ...
        x = x.view(size[0], self.conv.out_channels, *size[2:])
        return x


def skip_connection(
    in_features, out_features, n_dim=2, bias=False, skip_type="soft-gating"
):
    """A wrapper for several types of skip connections.
    Returns an nn.Module skip connections, one of  {'identity', 'linear', soft-gating'}

    Parameters
    ----------
    in_features : int
        number of input features
    out_features : int
        number of output features
    n_dim : int, default is 2
        Dimensionality of the input (excluding batch-size and channels).
        ``n_dim=2`` corresponds to having Module2D.
    bias : bool, optional
        whether to use a bias, by default False
    skip_type : {'identity', 'linear', soft-gating'}
        kind of skip connection to use, by default "soft-gating"

    Returns
    -------
    nn.Module
        module that takes in x and returns skip(x)
    """
    if skip_type.lower() == "soft-gating":
        return SoftGating(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            n_dim=n_dim,
        )
    elif skip_type.lower() == "linear":
        return Flattened1dConv(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            bias=bias,
        )
    elif skip_type.lower() == "identity":
        return nn.Identity()
    else:
        raise ValueError(
            f"Got skip-connection type={skip_type}, expected one of"
            f" {'soft-gating', 'linear', 'id'}."
        )


class SoftGating(nn.Module):
    """Applies soft-gating by weighting the channels of the given input

    Given an input x of size `(batch-size, channels, height, width)`,
    this returns `x * w `
    where w is of shape `(1, channels, 1, 1)`

    Parameters
    ----------
    in_features : int
    out_features : None
        this is provided for API compatibility with nn.Linear only
    n_dim : int, default is 2
        Dimensionality of the input (excluding batch-size and channels).
        ``n_dim=2`` corresponds to having Module2D.
    bias : bool, default is False
    """

    def __init__(self, in_features, out_features=None, n_dim=2, bias=False):
        super().__init__()
        if out_features is not None and in_features != out_features:
            raise ValueError(
                f"Got in_features={in_features} and out_features={out_features}, "
                "but these two must be the same for soft-gating"
            )
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.ones(1, self.in_features, *(1,) * n_dim))
        if bias:
            self.bias = nn.Parameter(torch.ones(1, self.in_features, *(1,) * n_dim))
        else:
            self.bias = None

    def forward(self, x):
        """Applies soft-gating to a batch of activations"""
        if self.bias is not None:
            return self.weight * x + self.bias
        else:
            return self.weight * x


class SubModule(nn.Module):
    """Class representing one of the sub_module from the mother joint module

    Notes
    -----
    This relies on the fact that nn.Parameters are not duplicated:
    if the same nn.Parameter is assigned to multiple modules,
    they all point to the same data, which is shared.
    """

    def __init__(self, main_module, indices):
        super().__init__()
        self.main_module = main_module
        self.indices = indices

    def forward(self, x):
        return self.main_module.forward(x, self.indices)


class BaseSpectralConv(nn.Module):
    def __init__(self, device=None, dtype=None):
        """Base Class for Spectral Convolutions

        Use it when you want to build your own FNO-type Neural Operators
        """
        super().__init__()

        self.dtype = dtype
        self.device = device

    def transform(self, x):
        """Transforms an input x for a skip connection, by default just an identity map

        If your function transforms the input then you should also implement this transform method
        so the skip connection can also work.

        Typical usecases are:

        * Your upsample or downsample the input in the Spectral conv: the skip connection has to be similarly scaled.
          This allows you to deal with it however you want (e.g. avoid aliasing)
        * You perform a change of basis in your Spectral Conv, again, this needs to be applied to the skip connection too.
        """
        return x


def _contract_dense(x, weight, separable=False):
    order = x.ndim
    # batch-size, in_channels, x, y...
    x_syms = list(einsum_symbols[:order])

    # in_channels, out_channels, x, y...
    weight_syms = list(x_syms[1:])  # no batch-size

    # batch-size, out_channels, x, y...
    if separable:
        out_syms = [x_syms[0]] + list(weight_syms)
    else:
        weight_syms.insert(1, einsum_symbols[order])  # outputs
        out_syms = list(weight_syms)
        out_syms[0] = x_syms[0]

    eq = f'{"".join(x_syms)},{"".join(weight_syms)}->{"".join(out_syms)}'

    if not torch.is_tensor(weight):
        weight = weight.to_tensor()

    if x.dtype == torch.complex32:
        # if x is half precision, run a specialized einsum
        return einsum_complexhalf(eq, x, weight)
    else:
        return torch.einsum(eq, x, weight)


def _contract_dense_separable(x, weight, separable):
    if not torch.is_tensor(weight):
        weight = weight.to_tensor()
    return x * weight


def _contract_cp(x, cp_weight, separable=False):
    order = x.ndim

    x_syms = str(einsum_symbols[:order])
    rank_sym = einsum_symbols[order]
    out_sym = einsum_symbols[order + 1]
    out_syms = list(x_syms)
    if separable:
        factor_syms = [einsum_symbols[1] + rank_sym]  # in only
    else:
        out_syms[1] = out_sym
        factor_syms = [einsum_symbols[1] + rank_sym, out_sym + rank_sym]  # in, out
    factor_syms += [xs + rank_sym for xs in x_syms[2:]]  # x, y, ...
    eq = f'{x_syms},{rank_sym},{",".join(factor_syms)}->{"".join(out_syms)}'

    if x.dtype == torch.complex32:
        return einsum_complexhalf(eq, x, cp_weight.weights, *cp_weight.factors)
    else:
        return torch.einsum(eq, x, cp_weight.weights, *cp_weight.factors)


def _contract_tucker(x, tucker_weight, separable=False):
    order = x.ndim

    x_syms = str(einsum_symbols[:order])
    out_sym = einsum_symbols[order]
    out_syms = list(x_syms)
    if separable:
        core_syms = einsum_symbols[order + 1 : 2 * order]
        # factor_syms = [einsum_symbols[1]+core_syms[0]] #in only
        # x, y, ...
        factor_syms = [xs + rs for (xs, rs) in zip(x_syms[1:], core_syms)]

    else:
        core_syms = einsum_symbols[order + 1 : 2 * order + 1]
        out_syms[1] = out_sym
        factor_syms = [
            einsum_symbols[1] + core_syms[0],
            out_sym + core_syms[1],
        ]  # out, in
        # x, y, ...
        factor_syms += [xs + rs for (xs, rs) in zip(x_syms[2:], core_syms[2:])]

    eq = f'{x_syms},{core_syms},{",".join(factor_syms)}->{"".join(out_syms)}'

    if x.dtype == torch.complex32:
        return einsum_complexhalf(eq, x, tucker_weight.core, *tucker_weight.factors)
    else:
        return torch.einsum(eq, x, tucker_weight.core, *tucker_weight.factors)


def _contract_tt(x, tt_weight, separable=False):
    order = x.ndim

    x_syms = list(einsum_symbols[:order])
    weight_syms = list(x_syms[1:])  # no batch-size
    if not separable:
        weight_syms.insert(1, einsum_symbols[order])  # outputs
        out_syms = list(weight_syms)
        out_syms[0] = x_syms[0]
    else:
        out_syms = list(x_syms)
    rank_syms = list(einsum_symbols[order + 1 :])
    tt_syms = []
    for i, s in enumerate(weight_syms):
        tt_syms.append([rank_syms[i], s, rank_syms[i + 1]])
    eq = (
        "".join(x_syms)
        + ","
        + ",".join("".join(f) for f in tt_syms)
        + "->"
        + "".join(out_syms)
    )

    if x.dtype == torch.complex32:
        return einsum_complexhalf(eq, x, *tt_weight.factors)
    else:
        return torch.einsum(eq, x, *tt_weight.factors)


def get_contract_fun(weight, implementation="reconstructed", separable=False):
    """Generic ND implementation of Fourier Spectral Conv contraction

    Parameters
    ----------
    weight : tensorly-torch's FactorizedTensor
    implementation : {'reconstructed', 'factorized'}, default is 'reconstructed'
        whether to reconstruct the weight and do a forward pass (reconstructed)
        or contract directly the factors of the factorized weight with the input (factorized)
    separable: bool
        if True, performs contraction with individual tensor factors.
        if False,
    Returns
    -------
    function : (x, weight) -> x * weight in Fourier space
    """
    if implementation == "reconstructed":
        if separable:
            return _contract_dense_separable
        else:
            return _contract_dense
    elif implementation == "factorized":
        if torch.is_tensor(weight):
            return _contract_dense
        elif isinstance(weight, FactorizedTensor):
            if weight.name.lower().endswith("dense"):
                return _contract_dense
            elif weight.name.lower().endswith("tucker"):
                return _contract_tucker
            elif weight.name.lower().endswith("tt"):
                return _contract_tt
            elif weight.name.lower().endswith("cp"):
                return _contract_cp
            else:
                raise ValueError(f"Got unexpected factorized weight type {weight.name}")
        else:
            raise ValueError(
                f"Got unexpected weight type of class {weight.__class__.__name__}"
            )
    else:
        raise ValueError(
            f'Got implementation={implementation}, expected "reconstructed" or "factorized"'
        )


class SpectralConv(BaseSpectralConv):
    """SpectralConv implements the Spectral Convolution component of a Fourier layer
    described in [1]_ and [2]_.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    n_modes : int or int tuple
        Number of modes to use for contraction in Fourier domain during training.

        .. warning::

            We take care of the redundancy in the Fourier modes, therefore, for an input
            of size I_1, ..., I_N, please provide modes M_K that are I_1 < M_K <= I_N
            We will automatically keep the right amount of modes: specifically, for the
            last mode only, if you specify M_N modes we will use M_N // 2 + 1 modes
            as the real FFT is redundant along that last dimension. For more information on
            mode truncation, refer to :ref:`fourier_layer_impl`


        .. note::

            Provided modes should be even integers. odd numbers will be rounded to the closest even number.

        This can be updated dynamically during training.

    max_n_modes : int tuple or None, default is None
        * If not None, **maximum** number of modes to keep in Fourier Layer, along each dim
            The number of modes (`n_modes`) cannot be increased beyond that.
        * If None, all the n_modes are used.

    separable : bool, default is True
        whether to use separable implementation of contraction
        if True, contracts factors of factorized
        tensor weight individually
    init_std : float or 'auto', default is 'auto'
        std to use for the init
    factorization : str or None, {'tucker', 'cp', 'tt'}, default is None
        If None, a single dense weight is learned for the FNO.
        Otherwise, that weight, used for the contraction in the Fourier domain
        is learned in factorized form. In that case, `factorization` is the
        tensor factorization of the parameters weight used.
    rank : float or rank, optional
        Rank of the tensor factorization of the Fourier weights, by default 1.0
        Ignored if ``factorization is None``
    fixed_rank_modes : bool, optional
        Modes to not factorize, by default False
        Ignored if ``factorization is None``
    fft_norm : str, optional
        fft normalization parameter, by default 'forward'
    implementation : {'factorized', 'reconstructed'}, optional, default is 'factorized'
        If factorization is not None, forward mode to use::
        * `reconstructed` : the full weight tensor is reconstructed from the
          factorization and used for the forward pass
        * `factorized` : the input is directly contracted with the factors of
          the decomposition
        Ignored if ``factorization is None``
    decomposition_kwargs : dict, optional, default is {}
        Optionaly additional parameters to pass to the tensor decomposition
        Ignored if ``factorization is None``
    complex_data: bool, optional
        whether data takes on complex values in the spatial domain, by default False
        if True, uses different logic for FFT contraction and uses full FFT instead of real-valued

    References
    -----------
    .. [1] :

    Li, Z. et al. "Fourier Neural Operator for Parametric Partial Differential
        Equations" (2021). ICLR 2021, https://arxiv.org/pdf/2010.08895.

    .. [2] :

    Kossaifi, J., Kovachki, N., Azizzadenesheli, K., Anandkumar, A. "Multi-Grid
        Tensorized Fourier Neural Operator for High-Resolution PDEs" (2024).
        TMLR 2024, https://openreview.net/pdf?id=AWiDlO63bH.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        n_modes,
        complex_data=False,
        max_n_modes=None,
        bias=True,
        separable=False,
        resolution_scaling_factor: Optional[Union[Number, List[Number]]] = None,
        fno_block_precision="full",
        rank=0.5,
        factorization=None,
        implementation="reconstructed",
        fixed_rank_modes=False,
        decomposition_kwargs: Optional[dict] = None,
        init_std="auto",
        fft_norm="forward",
        device=None,
    ):
        super().__init__(device=device)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.complex_data = complex_data

        # n_modes is the total number of modes kept along each dimension
        self.n_modes = n_modes
        self.order = len(self.n_modes)

        if max_n_modes is None:
            max_n_modes = self.n_modes
        elif isinstance(max_n_modes, int):
            max_n_modes = [max_n_modes]
        self.max_n_modes = max_n_modes

        self.fno_block_precision = fno_block_precision
        self.rank = rank
        self.factorization = factorization
        self.implementation = implementation

        self.resolution_scaling_factor: Union[None, List[List[float]]] = (
            validate_scaling_factor(resolution_scaling_factor, self.order)
        )

        if init_std == "auto":
            init_std = (2 / (in_channels + out_channels)) ** 0.5
        else:
            init_std = init_std

        if isinstance(fixed_rank_modes, bool):
            if fixed_rank_modes:
                # If bool, keep the number of layers fixed
                fixed_rank_modes = [0]
            else:
                fixed_rank_modes = None
        self.fft_norm = fft_norm

        if factorization is None:
            factorization = "Dense"  # No factorization

        if separable:
            if in_channels != out_channels:
                raise ValueError(
                    "To use separable Fourier Conv, in_channels must be equal "
                    f"to out_channels, but got in_channels={in_channels} and "
                    f"out_channels={out_channels}",
                )
            weight_shape = (in_channels, *max_n_modes)
        else:
            weight_shape = (in_channels, out_channels, *max_n_modes)
        self.separable = separable

        tensor_kwargs = decomposition_kwargs if decomposition_kwargs is not None else {}

        # Create/init spectral weight tensor

        if factorization is None:
            self.weight = torch.tensor(weight_shape, dtype=torch.cfloat)
        else:
            self.weight = FactorizedTensor.new(
                weight_shape,
                rank=self.rank,
                factorization=factorization,
                fixed_rank_modes=fixed_rank_modes,
                **tensor_kwargs,
                dtype=torch.cfloat,
            )
        self.weight.normal_(0, init_std)

        self._contract = get_contract_fun(
            self.weight, implementation=implementation, separable=separable
        )

        if bias:
            self.bias = nn.Parameter(
                init_std
                * torch.randn(*(tuple([self.out_channels]) + (1,) * self.order))
            )
        else:
            self.bias = None

    def transform(self, x, output_shape=None):
        in_shape = list(x.shape[2:])

        if self.resolution_scaling_factor is not None and output_shape is None:
            out_shape = tuple(
                [
                    round(s * r)
                    for (s, r) in zip(in_shape, self.resolution_scaling_factor)
                ]
            )
        elif output_shape is not None:
            out_shape = output_shape
        else:
            out_shape = in_shape

        if in_shape == out_shape:
            return x
        else:
            return resample(x, 1.0, list(range(2, x.ndim)), output_shape=out_shape)

    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        if isinstance(n_modes, int):  # Should happen for 1D FNO only
            n_modes = [n_modes]
        else:
            n_modes = list(n_modes)
        # the real FFT is skew-symmetric, so the last mode has a redundacy if our data is real in space
        # As a design choice we do the operation here to avoid users dealing with the +1
        # if we use the full FFT we cannot cut off informtion from the last mode
        if not self.complex_data:
            n_modes[-1] = n_modes[-1] // 2 + 1
        self._n_modes = n_modes

    def forward(self, x: torch.Tensor, output_shape: Optional[Tuple[int]] = None):
        """Generic forward pass for the Factorized Spectral Conv

        Parameters
        ----------
        x : torch.Tensor
            input activation of size (batch_size, channels, d1, ..., dN)

        Returns
        -------
        tensorized_spectral_conv(x)
        """
        batchsize, channels, *mode_sizes = x.shape

        fft_size = list(mode_sizes)
        if not self.complex_data:
            fft_size[-1] = (
                fft_size[-1] // 2 + 1
            )  # Redundant last coefficient in real spatial data
        fft_dims = list(range(-self.order, 0))

        if self.fno_block_precision == "half":
            x = x.half()

        if self.complex_data:
            x = torch.fft.fftn(x, norm=self.fft_norm, dim=fft_dims)
        else:
            x = torch.fft.rfftn(x, norm=self.fft_norm, dim=fft_dims)
        if self.order > 1:
            x = torch.fft.fftshift(x, dim=fft_dims[:-1])

        if self.fno_block_precision == "mixed":
            # if 'mixed', the above fft runs in full precision, but the
            # following operations run at half precision
            x = x.chalf()

        if self.fno_block_precision in ["half", "mixed"]:
            out_dtype = torch.chalf
        else:
            out_dtype = torch.cfloat
        out_fft = torch.zeros(
            [batchsize, self.out_channels, *fft_size], device=x.device, dtype=out_dtype
        )

        # if current modes are less than max, start indexing modes closer to the center of the weight tensor
        starts = [
            (max_modes - min(size, n_mode))
            for (size, n_mode, max_modes) in zip(
                fft_size, self.n_modes, self.max_n_modes
            )
        ]

        # if contraction is separable, weights have shape (channels, modes_x, ...)
        # otherwise they have shape (in_channels, out_channels, modes_x, ...)
        if self.separable:
            slices_w = [slice(None)]  # channels
        else:
            slices_w = [slice(None), slice(None)]  # in_channels, out_channels
        if self.complex_data:
            slices_w += [
                slice(start // 2, -start // 2) if start else slice(start, None)
                for start in starts
            ]
        else:
            # The last mode already has redundant half removed in real FFT
            slices_w += [
                slice(start // 2, -start // 2) if start else slice(start, None)
                for start in starts[:-1]
            ]
            slices_w += [slice(None, -starts[-1]) if starts[-1] else slice(None)]

        weight = self.weight[slices_w]

        # if separable conv, weight tensor only has one channel dim
        if self.separable:
            weight_start_idx = 1
        # otherwise drop first two dims (in_channels, out_channels)
        else:
            weight_start_idx = 2
        starts = [
            (size - min(size, n_mode))
            for (size, n_mode) in zip(
                list(x.shape[2:]), list(weight.shape[weight_start_idx:])
            )
        ]
        slices_x = [slice(None), slice(None)]  # Batch_size, channels

        if self.complex_data:
            slices_x += [
                slice(start // 2, -start // 2) if start else slice(start, None)
                for start in starts
            ]
        else:
            slices_x += [
                slice(start // 2, -start // 2) if start else slice(start, None)
                for start in starts[:-1]
            ]
            slices_x += [
                slice(None, -starts[-1]) if starts[-1] else slice(None)
            ]  # The last mode already has redundant half removed
        out_fft[slices_x] = self._contract(
            x[slices_x], weight, separable=self.separable
        )

        if self.resolution_scaling_factor is not None and output_shape is None:
            mode_sizes = tuple(
                [
                    round(s * r)
                    for (s, r) in zip(mode_sizes, self.resolution_scaling_factor)
                ]
            )

        if output_shape is not None:
            mode_sizes = output_shape

        if self.order > 1:
            out_fft = torch.fft.fftshift(out_fft, dim=fft_dims[:-1])

        if self.complex_data:
            x = torch.fft.ifftn(out_fft, s=mode_sizes, dim=fft_dims, norm=self.fft_norm)
        else:
            x = torch.fft.irfftn(
                out_fft, s=mode_sizes, dim=fft_dims, norm=self.fft_norm
            )

        if self.bias is not None:
            x = x + self.bias

        return x


def validate_scaling_factor(
    scaling_factor: Union[None, Number, List[Number], List[List[Number]]],
    n_dim: int,
    n_layers: Optional[int] = None,
) -> Union[None, List[float], List[List[float]]]:
    """
    Parameters
    ----------
    scaling_factor : None OR float OR list[float] Or list[list[float]]
    n_dim : int
    n_layers : int or None; defaults to None
        If None, return a single list (rather than a list of lists)
        with `factor` repeated `dim` times.
    """
    if scaling_factor is None:
        return None
    if isinstance(scaling_factor, (float, int)):
        if n_layers is None:
            return [float(scaling_factor)] * n_dim

        return [[float(scaling_factor)] * n_dim] * n_layers

    if (
        isinstance(scaling_factor, list)
        and len(scaling_factor) > 0
        and all([isinstance(s, (float, int)) for s in scaling_factor])
    ):
        if n_layers is None and len(scaling_factor) == n_dim:
            # this is a dim-wise scaling
            return [float(s) for s in scaling_factor]
        return [[float(s)] * n_dim for s in scaling_factor]

    if (
        isinstance(scaling_factor, list)
        and len(scaling_factor) > 0
        and all([isinstance(s, (list)) for s in scaling_factor])
    ):
        s_sub_pass = True
        for s in scaling_factor:
            if all([isinstance(s_sub, (int, float)) for s_sub in s]):
                pass
            else:
                s_sub_pass = False
            if s_sub_pass:
                return scaling_factor

    return None


class FNOBlocks(nn.Module):
    """FNOBlocks implements a sequence of Fourier layers, the operations of which
    are first described in [1]_. The exact implementation details of the Fourier
    layer architecture are discussed in [2]_.

    Parameters
    ----------
    in_channels : int
        input channels to Fourier layers
    out_channels : int
        output channels after Fourier layers
    n_modes : int, List[int]
        number of modes to keep along each dimension
        in frequency space. Can either be specified as
        an int (for all dimensions) or an iterable with one
        number per dimension
    resolution_scaling_factor : Optional[Union[Number, List[Number]]], optional
        factor by which to scale outputs for super-resolution, by default None
    n_layers : int, optional
        number of Fourier layers to apply in sequence, by default 1
    max_n_modes : int, List[int], optional
        maximum number of modes to keep along each dimension, by default None
    fno_block_precision : str, optional
        floating point precision to use for computations, by default "full"
    channel_mlp_dropout : int, optional
        dropout parameter for self.channel_mlp, by default 0
    channel_mlp_expansion : float, optional
        expansion parameter for self.channel_mlp, by default 0.5
    non_linearity : torch.nn.F module, optional
        nonlinear activation function to use between layers, by default F.gelu
    stabilizer : Literal["tanh"], optional
        stabilizing module to use between certain layers, by default None
        if "tanh", use tanh
    norm : Literal["ada_in", "group_norm", "instance_norm"], optional
        Normalization layer to use, by default None
    ada_in_features : int, optional
        number of features for adaptive instance norm above, by default None
    preactivation : bool, optional
        whether to call forward pass with pre-activation, by default False
        if True, call nonlinear activation and norm before Fourier convolution
        if False, call activation and norms after Fourier convolutions
    fno_skip : str, optional
        module to use for FNO skip connections, by default "linear"
        see layers.skip_connections for more details
    channel_mlp_skip : str, optional
        module to use for ChannelMLP skip connections, by default "soft-gating"
        see layers.skip_connections for more details

    Other Parameters
    -------------------
    complex_data : bool, optional
        whether the FNO's data takes on complex values in space, by default False
    separable : bool, optional
        separable parameter for SpectralConv, by default False
    factorization : str, optional
        factorization parameter for SpectralConv, by default None
    rank : float, optional
        rank parameter for SpectralConv, by default 1.0
    conv_module : BaseConv, optional
        module to use for convolutions in FNO block, by default SpectralConv
    joint_factorization : bool, optional
        whether to factorize all spectralConv weights as one tensor, by default False
    fixed_rank_modes : bool, optional
        fixed_rank_modes parameter for SpectralConv, by default False
    implementation : str, optional
        implementation parameter for SpectralConv, by default "factorized"
    decomposition_kwargs : _type_, optional
        kwargs for tensor decomposition in SpectralConv, by default dict()

    References
    -----------
    .. [1] Li, Z. et al. "Fourier Neural Operator for Parametric Partial Differential
           Equations" (2021). ICLR 2021, https://arxiv.org/pdf/2010.08895.
    .. [2] Kossaifi, J., Kovachki, N., Azizzadenesheli, K., Anandkumar, A. "Multi-Grid
           Tensorized Fourier Neural Operator for High-Resolution PDEs" (2024).
           TMLR 2024, https://openreview.net/pdf?id=AWiDlO63bH.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        n_modes,
        resolution_scaling_factor=None,
        n_layers=1,
        max_n_modes=None,
        fno_block_precision="full",
        channel_mlp_dropout=0,
        channel_mlp_expansion=0.5,
        non_linearity=F.gelu,
        stabilizer=None,
        norm=None,
        ada_in_features=None,
        preactivation=False,
        fno_skip="linear",
        channel_mlp_skip="soft-gating",
        complex_data=False,
        separable=False,
        factorization=None,
        rank=1.0,
        conv_module=SpectralConv,
        fixed_rank_modes=False,  # undoc
        implementation="factorized",  # undoc
        decomposition_kwargs=dict(),
        **kwargs,
    ):
        super().__init__()
        if isinstance(n_modes, int):
            n_modes = [n_modes]
        self._n_modes = n_modes
        self.n_dim = len(n_modes)

        self.resolution_scaling_factor: Union[None, List[List[float]]] = (
            validate_scaling_factor(resolution_scaling_factor, self.n_dim, n_layers)
        )

        self.max_n_modes = max_n_modes
        self.fno_block_precision = fno_block_precision
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.stabilizer = stabilizer
        self.rank = rank
        self.factorization = factorization
        self.fixed_rank_modes = fixed_rank_modes
        self.decomposition_kwargs = decomposition_kwargs
        self.fno_skip = fno_skip
        self.channel_mlp_skip = channel_mlp_skip
        self.complex_data = complex_data

        self.channel_mlp_expansion = channel_mlp_expansion
        self.channel_mlp_dropout = channel_mlp_dropout
        self.implementation = implementation
        self.separable = separable
        self.preactivation = preactivation
        self.ada_in_features = ada_in_features

        # apply real nonlin if data is real, otherwise CGELU
        if self.complex_data:
            self.non_linearity = CGELU
        else:
            self.non_linearity = non_linearity

        self.convs = nn.ModuleList(
            [
                conv_module(
                    self.in_channels,
                    self.out_channels,
                    self.n_modes,
                    resolution_scaling_factor=(
                        None
                        if resolution_scaling_factor is None
                        else self.resolution_scaling_factor[i]
                    ),
                    max_n_modes=max_n_modes,
                    rank=rank,
                    fixed_rank_modes=fixed_rank_modes,
                    implementation=implementation,
                    separable=separable,
                    factorization=factorization,
                    fno_block_precision=fno_block_precision,
                    decomposition_kwargs=decomposition_kwargs,
                    complex_data=complex_data,
                )
                for i in range(n_layers)
            ]
        )

        self.fno_skips = nn.ModuleList(
            [
                skip_connection(
                    self.in_channels,
                    self.out_channels,
                    skip_type=fno_skip,
                    n_dim=self.n_dim,
                )
                for _ in range(n_layers)
            ]
        )
        if self.complex_data:
            self.fno_skips = nn.ModuleList([ComplexValued(x) for x in self.fno_skips])

        self.channel_mlp = nn.ModuleList(
            [
                ChannelMLP(
                    in_channels=self.out_channels,
                    hidden_channels=round(self.out_channels * channel_mlp_expansion),
                    dropout=channel_mlp_dropout,
                    n_dim=self.n_dim,
                )
                for _ in range(n_layers)
            ]
        )
        if self.complex_data:
            self.channel_mlp = nn.ModuleList(
                [ComplexValued(x) for x in self.channel_mlp]
            )
        self.channel_mlp_skips = nn.ModuleList(
            [
                skip_connection(
                    self.in_channels,
                    self.out_channels,
                    skip_type=channel_mlp_skip,
                    n_dim=self.n_dim,
                )
                for _ in range(n_layers)
            ]
        )
        if self.complex_data:
            self.channel_mlp_skips = nn.ModuleList(
                [ComplexValued(x) for x in self.channel_mlp_skips]
            )

        # Each block will have 2 norms if we also use a ChannelMLP
        self.n_norms = 2
        if norm is None:
            self.norm = None
        elif norm == "instance_norm":
            self.norm = nn.ModuleList(
                [InstanceNorm() for _ in range(n_layers * self.n_norms)]
            )
        elif norm == "group_norm":
            self.norm = nn.ModuleList(
                [
                    nn.GroupNorm(num_groups=1, num_channels=self.out_channels)
                    for _ in range(n_layers * self.n_norms)
                ]
            )

        elif norm == "ada_in":
            self.norm = nn.ModuleList(
                [
                    AdaIN(ada_in_features, out_channels)
                    for _ in range(n_layers * self.n_norms)
                ]
            )
        else:
            raise ValueError(
                f"Got norm={norm} but expected None or one of "
                "[instance_norm, group_norm, ada_in]"
            )

    def set_ada_in_embeddings(self, *embeddings):
        """Sets the embeddings of each Ada-IN norm layers

        Parameters
        ----------
        embeddings : tensor or list of tensor
            if a single embedding is given, it will be used for each norm layer
            otherwise, each embedding will be used for the corresponding norm layer
        """
        if len(embeddings) == 1:
            for norm in self.norm:
                norm.set_embedding(embeddings[0])
        else:
            for norm, embedding in zip(self.norm, embeddings):
                norm.set_embedding(embedding)

    def forward(self, x, index=0, output_shape=None):
        if self.preactivation:
            return self.forward_with_preactivation(x, index, output_shape)
        else:
            return self.forward_with_postactivation(x, index, output_shape)

    def forward_with_postactivation(self, x, index=0, output_shape=None):
        x_skip_fno = self.fno_skips[index](x)
        x_skip_fno = self.convs[index].transform(x_skip_fno, output_shape=output_shape)

        x_skip_channel_mlp = self.channel_mlp_skips[index](x)
        x_skip_channel_mlp = self.convs[index].transform(
            x_skip_channel_mlp, output_shape=output_shape
        )

        if self.stabilizer == "tanh":
            if self.complex_data:
                x = ctanh(x)
            else:
                x = torch.tanh(x)

        x_fno = self.convs[index](x, output_shape=output_shape)
        # self.convs(x, index, output_shape=output_shape)

        if self.norm is not None:
            x_fno = self.norm[self.n_norms * index](x_fno)

        x = x_fno + x_skip_fno

        if index < (self.n_layers - 1):
            x = self.non_linearity(x)

        x = self.channel_mlp[index](x) + x_skip_channel_mlp

        if self.norm is not None:
            x = self.norm[self.n_norms * index + 1](x)

        if index < (self.n_layers - 1):
            x = self.non_linearity(x)

        return x

    def forward_with_preactivation(self, x, index=0, output_shape=None):
        # Apply non-linear activation (and norm)
        # before this block's convolution/forward pass:
        x = self.non_linearity(x)

        if self.norm is not None:
            x = self.norm[self.n_norms * index](x)

        x_skip_fno = self.fno_skips[index](x)
        x_skip_fno = self.convs[index].transform(x_skip_fno, output_shape=output_shape)

        x_skip_channel_mlp = self.channel_mlp_skips[index](x)
        x_skip_channel_mlp = self.convs[index].transform(
            x_skip_channel_mlp, output_shape=output_shape
        )

        if self.stabilizer == "tanh":
            if self.complex_data:
                x = ctanh(x)
            else:
                x = torch.tanh(x)

        x_fno = self.convs[index](x, output_shape=output_shape)

        x = x_fno + x_skip_fno

        if index < (self.n_layers - 1):
            x = self.non_linearity(x)

        if self.norm is not None:
            x = self.norm[self.n_norms * index + 1](x)

        x = self.channel_mlp[index](x) + x_skip_channel_mlp

        return x

    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        for i in range(self.n_layers):
            self.convs[i].n_modes = n_modes
        self._n_modes = n_modes

    def get_block(self, indices):
        """Returns a sub-FNO Block layer from the jointly parametrized main block

        The parametrization of an FNOBlock layer is shared with the main one.
        """
        if self.n_layers == 1:
            raise ValueError(
                "A single layer is parametrized, directly use the main class."
            )

        return SubModule(self, indices)

    def __getitem__(self, indices):
        return self.get_block(indices)


class DomainPadding(nn.Module):
    """Applies domain padding scaled automatically to the input's resolution

    Parameters
    ----------
    domain_padding : float or list
        typically, between zero and one, percentage of padding to use
        if a list, make sure if matches the dim of (d1, ..., dN)
    padding_mode : {'symmetric', 'one-sided'}, optional
        whether to pad on both sides, by default 'one-sided'
    resolution_scaling_factor : int ; default is 1

    Notes
    -----
    This class works for any input resolution, as long as it is in the form
    `(batch-size, channels, d1, ...., dN)`
    """

    def __init__(
        self,
        domain_padding,
        padding_mode="one-sided",
        resolution_scaling_factor: Union[int, List[int]] = 1,
    ):
        super().__init__()
        self.domain_padding = domain_padding
        self.padding_mode = padding_mode.lower()
        if resolution_scaling_factor is None:
            resolution_scaling_factor = 1
        self.resolution_scaling_factor: Union[int, List[int]] = (
            resolution_scaling_factor
        )

        # dict(f'{resolution}'=padding) such that padded = F.pad(x, indices)
        self._padding = dict()

        # dict(f'{resolution}'=indices_to_unpad) such that unpadded = x[indices]
        self._unpad_indices = dict()

    def forward(self, x):
        """forward pass: pad the input"""
        self.pad(x)

    def pad(self, x, verbose=False):
        """Take an input and pad it by the desired fraction

        The amount of padding will be automatically scaled with the resolution
        """
        resolution = x.shape[2:]

        # if domain_padding is list, then to pass on
        if isinstance(self.domain_padding, (float, int)):
            self.domain_padding = [float(self.domain_padding)] * len(resolution)

        assert len(self.domain_padding) == len(resolution), (
            "domain_padding length must match the number of spatial/time dimensions "
            "(excluding batch, ch)"
        )

        resolution_scaling_factor = self.resolution_scaling_factor
        if not isinstance(self.resolution_scaling_factor, list):
            # if unset by the user, scaling_factor will be 1 be default,
            # so `resolution_scaling_factor` should never be None.
            resolution_scaling_factor: List[float] = validate_scaling_factor(
                self.resolution_scaling_factor, len(resolution), n_layers=None
            )

        try:
            padding = self._padding[f"{resolution}"]
            return F.pad(x, padding, mode="constant")

        except KeyError:
            padding = [round(p * r) for (p, r) in zip(self.domain_padding, resolution)]

            if verbose:
                print(
                    f"Padding inputs of resolution={resolution} with "
                    f"padding={padding}, {self.padding_mode}"
                )

            output_pad = padding

            output_pad = [
                round(i * j) for (i, j) in zip(resolution_scaling_factor, output_pad)
            ]

            # padding is being applied in reverse order
            # (so we must reverse the padding list)
            padding = padding[::-1]

            # the F.pad(x, padding) funtion pads the tensor 'x' in reverse order
            # of the "padding" list i.e. the last axis of tensor 'x' will be
            # padded by the amount mention at the first position of the
            # 'padding' vector. The details about F.pad can be found here:
            # https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html

            if self.padding_mode == "symmetric":
                # Pad both sides
                unpad_list = list()
                for p in output_pad:
                    if p == 0:
                        padding_end = None
                        padding_start = None
                    else:
                        padding_end = p
                        padding_start = -p
                    unpad_list.append(slice(padding_end, padding_start, None))
                unpad_indices = (Ellipsis,) + tuple(unpad_list)

                padding = [i for p in padding for i in (p, p)]

            elif self.padding_mode == "one-sided":
                # One-side padding
                unpad_list = list()
                for p in output_pad:
                    if p == 0:
                        padding_start = None
                    else:
                        padding_start = -p
                    unpad_list.append(slice(None, padding_start, None))
                unpad_indices = (Ellipsis,) + tuple(unpad_list)
                padding = [i for p in padding for i in (0, p)]
            else:
                raise ValueError(f"Got padding_mode={self.padding_mode}")

            self._padding[f"{resolution}"] = padding

            padded = F.pad(x, padding, mode="constant")

            output_shape = padded.shape[2:]

            output_shape = [
                round(i * j) for (i, j) in zip(resolution_scaling_factor, output_shape)
            ]

            self._unpad_indices[f"{[i for i in output_shape]}"] = unpad_indices

            return padded

    def unpad(self, x):
        """Remove the padding from padding inputs"""
        unpad_indices = self._unpad_indices[f"{list(x.shape[2:])}"]
        return x[unpad_indices]


class ChannelMLP(nn.Module):
    """ChannelMLP applies an arbitrary number of layers of
    1d convolution and nonlinearity to the channels of input
    and is invariant to spatial resolution.

    Parameters
    ----------
    in_channels : int
    out_channels : int, default is None
        if None, same is in_channels
    hidden_channels : int, default is None
        if None, same is in_channels
    n_layers : int, default is 2
        number of linear layers in the MLP
    non_linearity : default is F.gelu
    dropout : float, default is 0
        if > 0, dropout probability
    """

    def __init__(
        self,
        in_channels,
        out_channels=None,
        hidden_channels=None,
        n_layers=2,
        n_dim=2,
        non_linearity=F.gelu,
        dropout=0.0,
        **kwargs,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.hidden_channels = (
            in_channels if hidden_channels is None else hidden_channels
        )
        self.non_linearity = non_linearity
        self.dropout = (
            nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layers)])
            if dropout > 0.0
            else None
        )

        # we use nn.Conv1d for everything and roll data along the 1st data dim
        self.fcs = nn.ModuleList()
        for i in range(n_layers):
            if i == 0 and i == (n_layers - 1):
                self.fcs.append(nn.Conv1d(self.in_channels, self.out_channels, 1))
            elif i == 0:
                self.fcs.append(nn.Conv1d(self.in_channels, self.hidden_channels, 1))
            elif i == (n_layers - 1):
                self.fcs.append(nn.Conv1d(self.hidden_channels, self.out_channels, 1))
            else:
                self.fcs.append(
                    nn.Conv1d(self.hidden_channels, self.hidden_channels, 1)
                )

    def forward(self, x):
        reshaped = False
        size = list(x.shape)
        if x.ndim > 3:
            # batch, channels, x1, x2... extra dims
            # .reshape() is preferable but .view()
            # cannot be called on non-contiguous tensors
            x = x.reshape((*size[:2], -1))
            reshaped = True

        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if i < self.n_layers - 1:
                x = self.non_linearity(x)
            if self.dropout is not None:
                x = self.dropout[i](x)

        # if x was an N-d tensor reshaped into 1d, undo the reshaping
        # same logic as above: .reshape() handles contiguous tensors as well
        if reshaped:
            x = x.reshape((size[0], self.out_channels, *size[2:]))

        return x


def CGELU(x: torch.Tensor):
    """Complex GELU activation function
    Follows the formulation of CReLU from [1]_.
    Applies GELU to real and imaginary parts of the input
    separately, then combine as complex number


    Parameters
    -----------
    x : torch.tensor (dtype=complex)
        pre-activation inputs

    References
    ----------
    .. [1] :

    Trabelsi, C., et al. (2018). "Deep Complex Networks".
        ICLR 2018, https://openreview.net/pdf?id=H1T2hmZAb.
    """

    return F.gelu(x.real).type(torch.cfloat) + 1j * F.gelu(x.imag).type(torch.cfloat)


def ctanh(x: torch.Tensor):
    """Complex-valued tanh stabilizer
    Apply ctanh is real and imag part of the input separately, then combine as complex number
    Args:
        x: complex tensor
    """
    return torch.tanh(x.real).type(torch.cfloat) + 1j * torch.tanh(x.imag).type(
        torch.cfloat
    )


def apply_complex(real_func, imag_func, x, dtype=torch.cfloat):
    """
    fr: a function (e.g., conv) to be applied on real part of x
    fi: a function (e.g., conv) to be applied on imag part of x
    x: complex input.
    """
    return (real_func(x.real) - imag_func(x.imag)).type(dtype) + 1j * (
        real_func(x.imag) + imag_func(x.real)
    ).type(dtype)


class ComplexValued(nn.Module):
    """
    Wrapper class that converts a standard nn.Module that operates on real data
    into a module that operates on complex-valued spatial data.
    """

    def __init__(self, module):
        super(ComplexValued, self).__init__()
        self.fr = deepcopy(module)
        self.fi = deepcopy(module)

    def forward(self, x):
        return apply_complex(self.fr, self.fi, x)


class BaseModel(nn.Module):
    """Based class for all Models

    This class has two main functionalities:
    * It monitors the creation of subclass, that are automatically registered
      for users to use by name using the library's config system
    * When a new instance of this class is created, the init call is intercepted
      so we can store the parameters used to create the instance.
      This makes it possible to save trained models along with their init parameters,
      and therefore load saved modes easily.

    Notes
    -----
    Model can be versioned using the _version class attribute.
    This can be used for sanity check when loading models from checkpoints to verify the
    model hasn't been updated since.
    """

    _models = dict()
    _version = "0.1.0"

    def __init_subclass__(cls, name=None, **kwargs):
        """When a subclass is created, register it in _models
        We look for an existing name attribute.
        If not give, then we use the class' name.
        """
        super().__init_subclass__(**kwargs)
        if name is not None:
            BaseModel._models[name.lower()] = cls
            cls._name = name
        else:
            # warnings.warn(f'Creating a subclass of BaseModel {cls.__name__} with no name, initializing with {cls.__name__}.')
            BaseModel._models[cls.__name__.lower()] = cls
            cls._name = cls.__name__

    def __new__(cls, *args, **kwargs):
        """Verify arguments and save init kwargs for loading/saving

        We inspect the class' signature and check for unused parameters, or
        parameters not passed.

        We store all the args and kwargs given so we can duplicate the instance transparently.
        """
        sig = inspect.signature(cls)
        model_name = cls.__name__

        verbose = kwargs.get("verbose", False)
        # Verify that given parameters are actually arguments of the model
        for key in kwargs:
            if key not in sig.parameters:
                if verbose:
                    print(
                        f"Given argument key={key} "
                        f"that is not in {model_name}'s signature."
                    )

        # Check for model arguments not specified in the configuration
        for key, value in sig.parameters.items():
            if (value.default is not inspect._empty) and (key not in kwargs):
                if verbose:
                    print(
                        f"Keyword argument {key} not specified for model {model_name}, "
                        f"using default={value.default}."
                    )
                kwargs[key] = value.default

        if hasattr(cls, "_version"):
            kwargs["_version"] = cls._version
        kwargs["args"] = args
        kwargs["_name"] = cls._name
        instance = super().__new__(cls)
        instance._init_kwargs = kwargs

        return instance

    def state_dict(
        self, destination: dict = None, prefix: str = "", keep_vars: bool = False
    ):
        """
        state_dict subclasses nn.Module.state_dict() and adds a metadata field
        to track the model version and ensure only compatible saves are loaded.

        Parameters
        ----------
        destination : dict, optional
            If provided, the state of module will
            be updated into the dict and the same object is returned.
            Otherwise, an OrderedDict will be created and returned, by default None
        prefix : str, optional
            a prefix added to parameter and buffer
            names to compose the keys in state_dict, by default ``''``
        keep_vars (bool, optional): by default the torch.Tensors
            returned in the state dict are detached from autograd.
            If True, detaching will not be performed, by default False

        """
        state_dict = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        if state_dict.get("_metadata") == None:
            state_dict["_metadata"] = self._init_kwargs
        else:
            warnings.warn(
                "Attempting to update metadata for a module with metadata already in self.state_dict()"
            )
        return state_dict

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """load_state_dict subclasses nn.Module.load_state_dict() and adds a metadata field
        to track the model version and ensure only compatible saves are loaded.

        Parameters
        ----------
        state_dict : dict
            state dictionary generated by ``nn.Module.state_dict()``
        strict : bool, optional
            whether to strictly enforce that the keys in ``state_dict``
            match the keys returned by this module's, by default True.
        assign : bool, optional
            whether to assign items in the state dict to their corresponding keys
            in the module instead of copying them inplace into the module's current
            parameters and buffers. When False, the properties of the tensors in the
            current module are preserved while when True, the properties of the Tensors
            in the state dict are preserved, by default False

        Returns
        -------
        _type_
            _description_
        """
        metadata = state_dict.pop("_metadata", None)

        if metadata is not None:
            saved_version = metadata.get("_version", None)
            if saved_version is None:
                warnings.warn(
                    f"Saved instance of {self.__class__} has no stored version attribute."
                )
            if saved_version != self._version:
                warnings.warn(
                    f"Attempting to load a {self.__class__} of version {saved_version},"
                    f"But current version of {self.__class__} is {saved_version}"
                )
            # remove state dict metadata at the end to ensure proper loading with PyTorch module
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def save_checkpoint(self, save_folder, save_name):
        """Saves the model state and init param in the given folder under the given name"""
        save_folder = Path(save_folder)
        if not save_folder.exists():
            save_folder.mkdir(parents=True)

        state_dict_filepath = save_folder.joinpath(
            f"{save_name}_state_dict.pt"
        ).as_posix()
        torch.save(self.state_dict(), state_dict_filepath)
        metadata_filepath = save_folder.joinpath(f"{save_name}_metadata.pkl").as_posix()
        # Objects (e.g. GeLU) are not serializable by json - find a better solution in the future
        torch.save(self._init_kwargs, metadata_filepath)
        # with open(metadata_filepath, 'w') as f:
        #     json.dump(self._init_kwargs, f)

    def load_checkpoint(self, save_folder, save_name, map_location=None):
        save_folder = Path(save_folder)
        state_dict_filepath = save_folder.joinpath(
            f"{save_name}_state_dict.pt"
        ).as_posix()
        self.load_state_dict(torch.load(state_dict_filepath, map_location=map_location))

    @classmethod
    def from_checkpoint(cls, save_folder, save_name, map_location=None):
        save_folder = Path(save_folder)

        metadata_filepath = save_folder.joinpath(f"{save_name}_metadata.pkl").as_posix()
        init_kwargs = torch.load(metadata_filepath)
        # with open(metadata_filepath, 'r') as f:
        #     init_kwargs = json.load(f)

        version = init_kwargs.pop("_version")
        if hasattr(cls, "_version") and version != cls._version:
            print(version)
            warnings.warn(
                f"Checkpoint saved for version {version} of model {cls._name} but current code is version {cls._version}"
            )

        if "args" in init_kwargs:
            init_args = init_kwargs.pop("args")
        else:
            init_args = []
        instance = cls(*init_args, **init_kwargs)

        instance.load_checkpoint(save_folder, save_name, map_location=map_location)
        return instance


def regular_grid_2d(spatial_dims, grid_boundaries=[[0, 1], [0, 1]]):
    """
    Creates a 2 x height x width stack of positional encodings A, where
    A[:,i,j] = [[x,y]] at coordinate (i,j) on a (height, width) grid.
    """
    height, width = spatial_dims

    xt = torch.linspace(grid_boundaries[0][0], grid_boundaries[0][1], height + 1)[:-1]
    yt = torch.linspace(grid_boundaries[1][0], grid_boundaries[1][1], width + 1)[:-1]

    grid_x, grid_y = torch.meshgrid(xt, yt, indexing="ij")

    grid_x = grid_x.repeat(1, 1)
    grid_y = grid_y.repeat(1, 1)

    return grid_x, grid_y


def regular_grid_nd(
    resolutions: List[int], grid_boundaries: List[List[int]] = [[0, 1]] * 2
):
    """regular_grid_nd generates a tensor of coordinate points that
    describe a bounded regular grid.

    Creates a dim x res_d1 x ... x res_dn stack of positional encodings A, where
    A[:,c1,c2,...] = [[d1,d2,...dn]] at coordinate (c1,c2,...cn) on a (res_d1, ...res_dn) grid.

    Parameters
    ----------
    resolutions : List[int]
        resolution of the output grid along each dimension
    grid_boundaries : List[List[int]], optional
        List of pairs [start, end] of the boundaries of the
        regular grid. Must correspond 1-to-1 with resolutions default [[0,1], [0,1]]

    Returns
    -------
    grid: tuple(Tensor)
    list of tensors describing positional encoding
    """
    assert len(resolutions) == len(
        grid_boundaries
    ), "Error: inputs must have same number of dimensions"
    dim = len(resolutions)

    meshgrid_inputs = list()
    for res, (start, stop) in zip(resolutions, grid_boundaries):
        meshgrid_inputs.append(torch.linspace(start, stop, res + 1)[:-1])
    grid = torch.meshgrid(*meshgrid_inputs, indexing="ij")
    grid = tuple([x.repeat([1] * dim) for x in grid])
    return grid


class Embedding(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def out_channels(self):
        pass


class GridEmbedding2D(Embedding):
    """A simple positional embedding as a regular 2D grid"""

    def __init__(self, in_channels: int, grid_boundaries=[[0, 1], [0, 1]]):
        """GridEmbedding2D applies a simple positional
        embedding as a regular 2D grid

        Parameters
        ----------
        in_channels : int
            number of channels in input. Fixed for output channel interface
        grid_boundaries : list, optional
            coordinate boundaries of input grid, by default [[0, 1], [0, 1]]
        """
        super().__init__()
        self.in_channels = in_channels
        self.grid_boundaries = grid_boundaries
        self._grid = None
        self._res = None

    @property
    def out_channels(self):
        return self.in_channels + 2

    def grid(self, spatial_dims, device, dtype):
        """grid generates 2D grid needed for pos encoding
        and caches the grid associated with MRU resolution

        Parameters
        ----------
        spatial_dims : torch.size
             sizes of spatial resolution
        device : literal 'cpu' or 'cuda:*'
            where to load data
        dtype : str
            dtype to encode data

        Returns
        -------
        torch.tensor
            output grids to concatenate
        """
        # handle case of multiple train resolutions
        if self._grid is None or self._res != spatial_dims:
            grid_x, grid_y = regular_grid_2d(
                spatial_dims, grid_boundaries=self.grid_boundaries
            )
            grid_x = grid_x.to(device).to(dtype).unsqueeze(0).unsqueeze(0)
            grid_y = grid_y.to(device).to(dtype).unsqueeze(0).unsqueeze(0)
            self._grid = grid_x, grid_y
            self._res = spatial_dims

        return self._grid

    def forward(self, data, batched=True):
        if not batched:
            if data.ndim == 3:
                data = data.unsqueeze(0)
        batch_size = data.shape[0]
        x, y = self.grid(data.shape[-2:], data.device, data.dtype)
        out = torch.cat(
            (data, x.expand(batch_size, -1, -1, -1), y.expand(batch_size, -1, -1, -1)),
            dim=1,
        )
        # in the unbatched case, the dataloader will stack N
        # examples with no batch dim to create one
        if not batched and batch_size == 1:
            return out.squeeze(0)
        else:
            return out


class GridEmbeddingND(nn.Module):
    """A positional embedding as a regular ND grid"""

    def __init__(
        self, in_channels: int, dim: int = 2, grid_boundaries=[[0, 1], [0, 1]]
    ):
        """GridEmbeddingND applies a simple positional
        embedding as a regular ND grid

        Parameters
        ----------
        in_channels : int
            number of channels in input
        dim : int
            dimensions of positional encoding to apply
        grid_boundaries : list, optional
            coordinate boundaries of input grid along each dim, by default [[0, 1], [0, 1]]
        """
        super().__init__()
        self.in_channels = in_channels
        self.dim = dim
        assert self.dim == len(
            grid_boundaries
        ), f"Error: expected grid_boundaries to be\
            an iterable of length {self.dim}, received {grid_boundaries}"
        self.grid_boundaries = grid_boundaries
        self._grid = None
        self._res = None

    @property
    def out_channels(self):
        return self.in_channels + self.dim

    def grid(self, spatial_dims: torch.Size, device: str, dtype: torch.dtype):
        """grid generates ND grid needed for pos encoding
        and caches the grid associated with MRU resolution

        Parameters
        ----------
        spatial_dims : torch.Size
             sizes of spatial resolution
        device : literal 'cpu' or 'cuda:*'
            where to load data
        dtype : str
            dtype to encode data

        Returns
        -------
        torch.tensor
            output grids to concatenate
        """
        # handle case of multiple train resolutions
        if self._grid is None or self._res != spatial_dims:
            grids_by_dim = regular_grid_nd(
                spatial_dims, grid_boundaries=self.grid_boundaries
            )
            # add batch, channel dims
            grids_by_dim = [
                x.to(device).to(dtype).unsqueeze(0).unsqueeze(0) for x in grids_by_dim
            ]
            self._grid = grids_by_dim
            self._res = spatial_dims

        return self._grid

    def forward(self, data, batched=True):
        """
        Params
        --------
        data: torch.Tensor
            assumes shape batch (optional), channels, x_1, x_2, ...x_n
        batched: bool
            whether data has a batch dim
        """
        # add batch dim if it doesn't exist
        if not batched:
            if data.ndim == self.dim + 1:
                data = data.unsqueeze(0)
        batch_size = data.shape[0]
        grids = self.grid(
            spatial_dims=data.shape[2:], device=data.device, dtype=data.dtype
        )
        grids = [x.repeat(batch_size, *[1] * (self.dim + 1)) for x in grids]
        out = torch.cat((data, *grids), dim=1)
        return out


class FNO(BaseModel, name="FNO"):
    """N-Dimensional Fourier Neural Operator. The FNO learns a mapping between
    spaces of functions discretized over regular grids using Fourier convolutions,
    as described in [1]_.

    The key component of an FNO is its SpectralConv layer (see
    ``neuralop.layers.spectral_convolution``), which is similar to a standard CNN
    conv layer but operates in the frequency domain.

    For a deeper dive into the FNO architecture, refer to :ref:`fno_intro`.

    Parameters
    ----------
    n_modes : Tuple[int]
        number of modes to keep in Fourier Layer, along each dimension
        The dimensionality of the FNO is inferred from ``len(n_modes)``
    in_channels : int
        Number of channels in input function
    out_channels : int
        Number of channels in output function
    hidden_channels : int
        width of the FNO (i.e. number of channels), by default 256
    n_layers : int, optional
        Number of Fourier Layers, by default 4

    Documentation for more advanced parameters is below.

    Other parameters
    ------------------
    lifting_channel_ratio : int, optional
        ratio of lifting channels to hidden_channels, by default 2
        The number of liting channels in the lifting block of the FNO is
        lifting_channel_ratio * hidden_channels (e.g. default 512)
    projection_channel_ratio : int, optional
        ratio of projection channels to hidden_channels, by default 2
        The number of projection channels in the projection block of the FNO is
        projection_channel_ratio * hidden_channels (e.g. default 512)
    positional_embedding : Union[str, nn.Module], optional
        Positional embedding to apply to last channels of raw input
        before being passed through the FNO. Defaults to "grid"

        * If "grid", appends a grid positional embedding with default settings to
        the last channels of raw input. Assumes the inputs are discretized
        over a grid with entry [0,0,...] at the origin and side lengths of 1.

        * If an initialized GridEmbedding module, uses this module directly
        See :mod:`neuralop.embeddings.GridEmbeddingND` for details.

        * If None, does nothing

    non_linearity : nn.Module, optional
        Non-Linear activation function module to use, by default F.gelu
    norm : str {"ada_in", "group_norm", "instance_norm"}, optional
        Normalization layer to use, by default None
    complex_data : bool, optional
        Whether data is complex-valued (default False)
        if True, initializes complex-valued modules.
    channel_mlp_dropout : float, optional
        dropout parameter for ChannelMLP in FNO Block, by default 0
    channel_mlp_expansion : float, optional
        expansion parameter for ChannelMLP in FNO Block, by default 0.5
    channel_mlp_skip : str {'linear', 'identity', 'soft-gating'}, optional
        Type of skip connection to use in channel-mixing mlp, by default 'soft-gating'
    fno_skip : str {'linear', 'identity', 'soft-gating'}, optional
        Type of skip connection to use in FNO layers, by default 'linear'
    resolution_scaling_factor : Union[Number, List[Number]], optional
        layer-wise factor by which to scale the domain resolution of function, by default None

        * If a single number n, scales resolution by n at each layer

        * if a list of numbers [n_0, n_1,...] scales layer i's resolution by n_i.
    domain_padding : Union[Number, List[Number]], optional
        If not None, percentage of padding to use, by default None
        To vary the percentage of padding used along each input dimension,
        pass in a list of percentages e.g. [p1, p2, ..., pN] such that
        p1 corresponds to the percentage of padding along dim 1, etc.
    domain_padding_mode : str {'symmetric', 'one-sided'}, optional
        How to perform domain padding, by default 'one-sided'
    fno_block_precision : str {'full', 'half', 'mixed'}, optional
        precision mode in which to perform spectral convolution, by default "full"
    stabilizer : str {'tanh'} | None, optional
        whether to use a tanh stabilizer in FNO block, by default None

        Note: stabilizer greatly improves performance in the case
        `fno_block_precision='mixed'`.

    max_n_modes : Tuple[int] | None, optional

        * If not None, this allows to incrementally increase the number of
        modes in Fourier domain during training. Has to verify n <= N
        for (n, m) in zip(max_n_modes, n_modes).

        * If None, all the n_modes are used.

        This can be updated dynamically during training.
    factorization : str, optional
        Tensor factorization of the FNO layer weights to use, by default None.

        * If None, a dense tensor parametrizes the Spectral convolutions

        * Otherwise, the specified tensor factorization is used.
    rank : float, optional
        tensor rank to use in above factorization, by default 1.0
    fixed_rank_modes : bool, optional
        Modes to not factorize, by default False
    implementation : str {'factorized', 'reconstructed'}, optional

        * If 'factorized', implements tensor contraction with the individual factors of the decomposition

        * If 'reconstructed', implements with the reconstructed full tensorized weight.
    decomposition_kwargs : dict, optional
        extra kwargs for tensor decomposition (see `tltorch.FactorizedTensor`), by default dict()
    separable : bool, optional (**DEACTIVATED**)
        if True, use a depthwise separable spectral convolution, by default False
    preactivation : bool, optional (**DEACTIVATED**)
        whether to compute FNO forward pass with resnet-style preactivation, by default False
    conv_module : nn.Module, optional
        module to use for FNOBlock's convolutions, by default SpectralConv

    Examples
    ---------

    >>> from neuralop.models import FNO
    >>> model = FNO(n_modes=(12,12), in_channels=1, out_channels=1, hidden_channels=64)
    >>> model
    FNO(
    (positional_embedding): GridEmbeddingND()
    (fno_blocks): FNOBlocks(
        (convs): SpectralConv(
        (weight): ModuleList(
            (0-3): 4 x DenseTensor(shape=torch.Size([64, 64, 12, 7]), rank=None)
        )
        )
            ... torch.nn.Module printout truncated ...

    References
    -----------
    .. [1] :

    Li, Z. et al. "Fourier Neural Operator for Parametric Partial Differential
        Equations" (2021). ICLR 2021, https://arxiv.org/pdf/2010.08895.

    """

    def __init__(
        self,
        n_modes: Tuple[int],
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_layers: int = 4,
        lifting_channel_ratio: int = 2,
        projection_channel_ratio: int = 2,
        positional_embedding: Union[str, nn.Module] = "grid",
        non_linearity: nn.Module = F.gelu,
        norm: str = None,
        complex_data: bool = False,
        channel_mlp_dropout: float = 0,
        channel_mlp_expansion: float = 0.5,
        channel_mlp_skip: str = "soft-gating",
        fno_skip: str = "linear",
        resolution_scaling_factor: Union[Number, List[Number]] = None,
        domain_padding: Union[Number, List[Number]] = None,
        domain_padding_mode: str = "one-sided",
        fno_block_precision: str = "full",
        stabilizer: str = None,
        max_n_modes: Tuple[int] = None,
        factorization: str = None,
        rank: float = 1.0,
        fixed_rank_modes: bool = False,
        implementation: str = "factorized",
        decomposition_kwargs: dict = dict(),
        separable: bool = False,
        preactivation: bool = False,
        conv_module: nn.Module = SpectralConv,
        **kwargs,
    ):

        super().__init__()
        self.n_dim = len(n_modes)

        # n_modes is a special property - see the class' property for underlying mechanism
        # When updated, change should be reflected in fno blocks
        self._n_modes = n_modes

        self.hidden_channels = hidden_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers

        # init lifting and projection channels using ratios w.r.t hidden channels
        self.lifting_channel_ratio = lifting_channel_ratio
        self.lifting_channels = lifting_channel_ratio * self.hidden_channels

        self.projection_channel_ratio = projection_channel_ratio
        self.projection_channels = projection_channel_ratio * self.hidden_channels

        self.non_linearity = non_linearity
        self.rank = rank
        self.factorization = factorization
        self.fixed_rank_modes = fixed_rank_modes
        self.decomposition_kwargs = decomposition_kwargs
        self.fno_skip = (fno_skip,)
        self.channel_mlp_skip = (channel_mlp_skip,)
        self.implementation = implementation
        self.separable = separable
        self.preactivation = preactivation
        self.complex_data = complex_data
        self.fno_block_precision = fno_block_precision

        if positional_embedding == "grid":
            spatial_grid_boundaries = [[0.0, 1.0]] * self.n_dim
            self.positional_embedding = GridEmbeddingND(
                in_channels=self.in_channels,
                dim=self.n_dim,
                grid_boundaries=spatial_grid_boundaries,
            )
        elif isinstance(positional_embedding, GridEmbedding2D):
            if self.n_dim == 2:
                self.positional_embedding = positional_embedding
            else:
                raise ValueError(
                    f"Error: expected {self.n_dim}-d positional embeddings, got {positional_embedding}"
                )
        elif isinstance(positional_embedding, GridEmbeddingND):
            self.positional_embedding = positional_embedding
        elif positional_embedding == None:
            self.positional_embedding = None
        else:
            raise ValueError(
                f"Error: tried to instantiate FNO positional embedding with {positional_embedding},\
                              expected one of 'grid', GridEmbeddingND"
            )

        if domain_padding is not None and (
            (isinstance(domain_padding, list) and sum(domain_padding) > 0)
            or (isinstance(domain_padding, (float, int)) and domain_padding > 0)
        ):
            self.domain_padding = DomainPadding(
                domain_padding=domain_padding,
                padding_mode=domain_padding_mode,
                resolution_scaling_factor=resolution_scaling_factor,
            )
        else:
            self.domain_padding = None

        self.domain_padding_mode = domain_padding_mode
        self.complex_data = self.complex_data

        if resolution_scaling_factor is not None:
            if isinstance(resolution_scaling_factor, (float, int)):
                resolution_scaling_factor = [resolution_scaling_factor] * self.n_layers
        self.resolution_scaling_factor = resolution_scaling_factor

        self.fno_blocks = FNOBlocks(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            n_modes=self.n_modes,
            resolution_scaling_factor=resolution_scaling_factor,
            channel_mlp_dropout=channel_mlp_dropout,
            channel_mlp_expansion=channel_mlp_expansion,
            non_linearity=non_linearity,
            stabilizer=stabilizer,
            norm=norm,
            preactivation=preactivation,
            fno_skip=fno_skip,
            channel_mlp_skip=channel_mlp_skip,
            complex_data=complex_data,
            max_n_modes=max_n_modes,
            fno_block_precision=fno_block_precision,
            rank=rank,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            separable=separable,
            factorization=factorization,
            decomposition_kwargs=decomposition_kwargs,
            conv_module=conv_module,
            n_layers=n_layers,
            **kwargs,
        )

        # if adding a positional embedding, add those channels to lifting
        lifting_in_channels = self.in_channels
        if self.positional_embedding is not None:
            lifting_in_channels += self.n_dim
        # if lifting_channels is passed, make lifting a Channel-Mixing MLP
        # with a hidden layer of size lifting_channels
        if self.lifting_channels:
            self.lifting = ChannelMLP(
                in_channels=lifting_in_channels,
                out_channels=self.hidden_channels,
                hidden_channels=self.lifting_channels,
                n_layers=2,
                n_dim=self.n_dim,
                non_linearity=non_linearity,
            )
        # otherwise, make it a linear layer
        else:
            self.lifting = ChannelMLP(
                in_channels=lifting_in_channels,
                hidden_channels=self.hidden_channels,
                out_channels=self.hidden_channels,
                n_layers=1,
                n_dim=self.n_dim,
                non_linearity=non_linearity,
            )
        # Convert lifting to a complex ChannelMLP if self.complex_data==True
        if self.complex_data:
            self.lifting = ComplexValued(self.lifting)

        self.projection = ChannelMLP(
            in_channels=self.hidden_channels,
            out_channels=out_channels,
            hidden_channels=self.projection_channels,
            n_layers=2,
            n_dim=self.n_dim,
            non_linearity=non_linearity,
        )
        if self.complex_data:
            self.projection = ComplexValued(self.projection)

    def forward(self, x, output_shape=None, **kwargs):
        """FNO's forward pass

        1. Applies optional positional encoding

        2. Sends inputs through a lifting layer to a high-dimensional latent space

        3. Applies optional domain padding to high-dimensional intermediate function representation

        4. Applies `n_layers` Fourier/FNO layers in sequence (SpectralConvolution + skip connections, nonlinearity)

        5. If domain padding was applied, domain padding is removed

        6. Projection of intermediate function representation to the output channels

        Parameters
        ----------
        x : tensor
            input tensor

        output_shape : {tuple, tuple list, None}, default is None
            Gives the option of specifying the exact output shape for odd shaped inputs.

            * If None, don't specify an output shape

            * If tuple, specifies the output-shape of the **last** FNO Block

            * If tuple list, specifies the exact output-shape of each FNO Block
        """

        if output_shape is None:
            output_shape = [None] * self.n_layers
        elif isinstance(output_shape, tuple):
            output_shape = [None] * (self.n_layers - 1) + [output_shape]

        # append spatial pos embedding if set
        if self.positional_embedding is not None:
            x = self.positional_embedding(x)

        x = self.lifting(x)

        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)

        for layer_idx in range(self.n_layers):
            x = self.fno_blocks(x, layer_idx, output_shape=output_shape[layer_idx])

        if self.domain_padding is not None:
            x = self.domain_padding.unpad(x)

        x = self.projection(x)

        return x

    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        self.fno_blocks.n_modes = n_modes
        self._n_modes = n_modes


class FNO1d(FNO):
    """1D Fourier Neural Operator

    For the full list of parameters, see :class:`neuralop.models.FNO`.

    Parameters
    ----------
    modes_height : int
        number of Fourier modes to keep along the height
    """

    def __init__(
        self,
        n_modes_height,
        hidden_channels,
        in_channels=3,
        out_channels=1,
        lifting_channels=256,
        projection_channels=256,
        max_n_modes=None,
        n_layers=4,
        resolution_scaling_factor=None,
        non_linearity=F.gelu,
        stabilizer=None,
        complex_data=False,
        fno_block_precision="full",
        channel_mlp_dropout=0,
        channel_mlp_expansion=0.5,
        norm=None,
        skip="soft-gating",
        separable=False,
        preactivation=False,
        factorization=None,
        rank=1.0,
        fixed_rank_modes=False,
        implementation="factorized",
        decomposition_kwargs=dict(),
        domain_padding=None,
        domain_padding_mode="one-sided",
        **kwargs,
    ):
        super().__init__(
            n_modes=(n_modes_height,),
            hidden_channels=hidden_channels,
            in_channels=in_channels,
            out_channels=out_channels,
            lifting_channels=lifting_channels,
            projection_channels=projection_channels,
            n_layers=n_layers,
            resolution_scaling_factor=resolution_scaling_factor,
            non_linearity=non_linearity,
            stabilizer=stabilizer,
            complex_data=complex_data,
            fno_block_precision=fno_block_precision,
            channel_mlp_dropout=channel_mlp_dropout,
            channel_mlp_expansion=channel_mlp_expansion,
            max_n_modes=max_n_modes,
            norm=norm,
            skip=skip,
            separable=separable,
            preactivation=preactivation,
            factorization=factorization,
            rank=rank,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            decomposition_kwargs=decomposition_kwargs,
            domain_padding=domain_padding,
            domain_padding_mode=domain_padding_mode,
        )
        self.n_modes_height = n_modes_height
