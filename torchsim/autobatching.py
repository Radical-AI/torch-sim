"""Utilities for batching and memory management in torchsim."""

from collections.abc import Iterator
from itertools import chain
from typing import Literal

import binpacking
import torch
from ase.build import bulk

from torchsim.models.interface import ModelInterface
from torchsim.runners import atoms_to_state
from torchsim.state import BaseState, concatenate_states, pop_states, split_state


def measure_model_memory_forward(model: ModelInterface, state: BaseState) -> float:
    """Measure peak GPU memory usage during model forward pass.

    Args:
        model: The model to measure memory usage for.
        state: The input state to pass to the model.

    Returns:
        Peak memory usage in GB.
    """
    # Clear GPU memory

    # gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.reset_peak_memory_stats()

    model(
        positions=state.positions,
        cell=state.cell,
        batch=state.batch,
        atomic_numbers=state.atomic_numbers,
    )

    return torch.cuda.max_memory_allocated() / 1024**3  # Convert to GB


def determine_max_batch_size(
    model: ModelInterface, state: BaseState, max_atoms: int = 20000
) -> int:
    """Determine maximum batch size that fits in GPU memory.

    Args:
        model: The model to test with.
        state: The base state to replicate.
        max_atoms: Maximum number of atoms to try.

    Returns:
        Maximum number of batches that fit in GPU memory.
    """
    # create a list of integers following the fibonacci sequence
    fib = [1, 2]
    while fib[-1] < max_atoms:
        fib.append(fib[-1] + fib[-2])

    for i in range(len(fib)):
        n_batches = fib[i]
        concat_state = concatenate_states([state] * n_batches)

        try:
            measure_model_memory_forward(model, concat_state)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                return fib[i - 2]
            raise

    return fib[-2]


def calculate_baseline_memory(model: ModelInterface) -> float:
    """Calculate baseline memory usage of the model.

    Args:
        model: The model to measure baseline memory for.

    Returns:
        Baseline memory usage in GB.
    """
    # Create baseline atoms with different sizes
    baseline_atoms = [bulk("Al", "fcc").repeat((i, 1, 1)) for i in range(1, 9, 2)]
    baseline_states = [
        atoms_to_state(atoms, model.device, model.dtype) for atoms in baseline_atoms
    ]

    # Measure memory usage for each state
    memory_list = [
        measure_model_memory_forward(model, state) for state in baseline_states
    ]

    # Calculate number of atoms in each baseline state
    n_atoms_list = [state.n_atoms for state in baseline_states]

    # Convert to tensors
    n_atoms_tensor = torch.tensor(n_atoms_list, dtype=torch.float)
    memory_tensor = torch.tensor(memory_list, dtype=torch.float)

    # Prepare design matrix (with column of ones for intercept)
    X = torch.stack([torch.ones_like(n_atoms_tensor), n_atoms_tensor], dim=1)

    # Solve normal equations
    beta = torch.linalg.lstsq(X, memory_tensor.unsqueeze(1)).solution.squeeze()

    # Extract intercept (b) and slope (m)
    intercept, _ = beta[0].item(), beta[1].item()

    return intercept


def calculate_scaling_metric(
    state_slice: BaseState,
    metric: Literal["n_atoms_x_density", "n_atoms"] = "n_atoms_x_density",
) -> float:
    """Calculate scaling metric for a state.

    Args:
        state_slice: The state to calculate metric for.
        metric: The type of metric to calculate.

    Returns:
        The calculated metric value.
    """
    if metric == "n_atoms":
        return state_slice.n_atoms
    if metric == "n_atoms_x_density":
        volume = torch.abs(torch.linalg.det(state_slice.cell[0])) / 1000
        number_density = state_slice.n_atoms / volume.item()
        return state_slice.n_atoms * number_density
    raise ValueError(f"Invalid metric: {metric}")


def estimate_max_metric(
    model: ModelInterface,
    state_list: list[BaseState],
    metric_values: list[float],
    max_atoms: int = 20000,
) -> float:
    """Estimate maximum metric value that fits in GPU memory.

    Args:
        model: The model to test with.
        state_list: List of states to test.
        metric_values: Corresponding metric values for each state.
        max_atoms: Maximum number of atoms to try.

    Returns:
        Maximum metric value that fits in GPU memory.
    """
    metric_values = torch.tensor(metric_values)

    # select one state with the min n_atoms
    min_metric = metric_values.min()
    max_metric = metric_values.max()

    min_state = state_list[metric_values.argmin()]
    max_state = state_list[metric_values.argmax()]

    min_state_max_batches = determine_max_batch_size(model, min_state, max_atoms)
    max_state_max_batches = determine_max_batch_size(model, max_state, max_atoms)

    return min(min_state_max_batches * min_metric, max_state_max_batches * max_metric)


class ChunkingAutoBatcher:
    """Batcher that chunks states into bins of similar computational cost."""

    def __init__(
        self,
        model: ModelInterface,
        states: list[BaseState] | BaseState,
        metric: Literal["n_atoms", "n_atoms_x_density"] = "n_atoms_x_density",
        max_metric: float | None = None,
        max_atoms_to_try: int = 1_000_000,
    ) -> None:
        """Initialize the batcher.

        Args:
            model: The model to batch for.
            states: States to batch.
            metric: Metric to use for batching.
            max_metric: Maximum metric value per batch.
                max_atoms_to_try: Maximum number of atoms to try when estimating max_metric.
        """
        self.state_slices = (
            split_state(states) if isinstance(states, BaseState) else states
        )
        self.metrics = [
            calculate_scaling_metric(state_slice, metric)
            for state_slice in self.state_slices
        ]
        if not max_metric:
            self.max_metric = estimate_max_metric(
                model, self.state_slices, self.metrics, max_atoms_to_try
            )
            print(f"Max metric calculated: {self.max_metric}")
        else:
            self.max_metric = max_metric

        # verify that no systems are too large
        max_metric_value = max(self.metrics)
        max_metric_idx = self.metrics.index(max_metric_value)
        if max_metric_value > self.max_metric:
            raise ValueError(
                f"Max metric of system with index {max_metric_idx} in states: "
                f"{max(self.metrics)} is greater than max_metric {self.max_metric}, "
                f"please set a larger max_metric or run smaller systems metric."
            )

        self.index_to_metric = dict(enumerate(self.metrics))
        self.index_bins = binpacking.to_constant_volume(
            self.index_to_metric, V_max=self.max_metric
        )
        self.batched_states = []
        for index_bin in self.index_bins:
            self.batched_states.append([self.state_slices[i] for i in index_bin])
        self.current_state_bin = 0

    def next_batch(
        self, *, return_indices: bool = False
    ) -> BaseState | tuple[list[BaseState], list[int]] | None:
        """Get the next batch of states.

        Args:
            return_indices: Whether to return indices along with the batch.

        Returns:
            The next batch of states, optionally with indices, or None if no more batches.
        """
        # TODO: we need to refactor this to operate on the full states rather
        # than the state slices, to be aligned with how the hotswapping batcher
        # works.

        # TODO: need to think about how this intersects with reporting too
        # TODO: definitely a clever treatment to be done with iterators here
        if self.current_state_bin < len(self.batched_states):
            state_bin = self.batched_states[self.current_state_bin]
            state = concatenate_states(state_bin)
            self.current_state_bin += 1
            if return_indices:
                return state, self.index_bins[self.current_state_bin - 1]
            return state
        return None

    def __iter__(self):
        return self

    def __next__(self):
        next_batch = self.next_batch()
        if next_batch is None:
            raise StopIteration
        return next_batch

    def restore_original_order(
        self, batched_states: list[BaseState]
    ) -> list[BaseState]:
        """Take the state bins and reorder them into a list.

        Args:
            batched_states: List of state batches to reorder.

        Returns:
            States in their original order.
        """
        state_bins = [split_state(state) for state in batched_states]

        # Flatten lists
        all_states = list(chain.from_iterable(state_bins))
        original_indices = list(chain.from_iterable(self.index_bins))

        if len(all_states) != len(original_indices):
            raise ValueError(
                f"Number of states ({len(all_states)}) does not match "
                f"number of original indices ({len(original_indices)})"
            )

        # sort states by original indices
        indexed_states = list(zip(original_indices, all_states, strict=False))
        return [state for _, state in sorted(indexed_states, key=lambda x: x[0])]


class HotswappingAutoBatcher:
    """Batcher that dynamically swaps states in and out based on convergence."""

    def __init__(
        self,
        model: ModelInterface,
        states: list[BaseState] | Iterator[BaseState] | BaseState,
        metric: Literal["n_atoms", "n_atoms_x_density"] = "n_atoms_x_density",
        max_metric: float | None = None,
        max_atoms_to_try: int = 1_000_000,
    ) -> None:
        """Initialize the batcher.

        Args:
            model: The model to batch for.
            states: States to batch.
            metric: Metric to use for batching.
            max_metric: Maximum metric value per batch.
            max_atoms_to_try: Maximum number of atoms to try when estimating max_metric.
        """
        if isinstance(states, BaseState):
            states = split_state(states)
        if isinstance(states, list):
            states = iter(states)

        self.model = model
        self.states_iterator = states
        self.metric = metric
        self.max_metric = max_metric or None
        self.max_atoms_to_try = max_atoms_to_try

        self.current_metrics = []
        self.current_idx = []
        self.iterator_idx = 0

        self.completed_idx_og_order = []

    def _get_next_states(self) -> None:
        """Insert states from the iterator until max_metric is reached."""
        new_metrics = []
        new_idx = []
        new_states = []
        for state in self.states_iterator:
            metric = calculate_scaling_metric(state, self.metric)
            if metric > self.max_metric:
                raise ValueError(
                    f"State metric {metric} is greater than max_metric "
                    f"{self.max_metric}, please set a larger max_metric "
                    f"or run smaller systems metric."
                )
            # new_metric += sum(new_metrics)
            if sum(self.current_metrics) + sum(new_metrics) + metric > self.max_metric:
                # put the state back in the iterator
                self.states_iterator = chain([state], self.states_iterator)
                break

            new_metrics.append(metric)
            new_idx.append(self.iterator_idx)
            new_states.append(state)
            self.iterator_idx += 1

        self.current_metrics.extend(new_metrics)
        self.current_idx.extend(new_idx)

        return new_states

    def _delete_old_states(self, completed_idx: list[int]) -> None:
        # Sort in descending order to avoid index shifting problems
        completed_idx.sort(reverse=True)

        # update state tracking lists
        for idx in completed_idx:
            og_idx = self.current_idx.pop(idx)
            self.current_metrics.pop(idx)
            self.completed_idx_og_order.append(og_idx)

    def _first_batch(self) -> BaseState:
        """Get the first batch of states.

        Returns:
            The first batch of states.
        """
        # we need to sample a state and use it to estimate the max metric
        # for the first batch
        first_state = next(self.states_iterator)
        first_metric = calculate_scaling_metric(first_state, self.metric)
        self.current_metrics += [first_metric]
        self.current_idx += [0]
        self.iterator_idx += 1
        # self.total_metric += first_metric

        # if max_metric is not set, estimate it
        has_max_metric = bool(self.max_metric)
        if not has_max_metric:
            self.max_metric = estimate_max_metric(
                self.model,
                [first_state],
                [first_metric],
                max_atoms=self.max_atoms_to_try,
            )
            self.max_metric *= 0.8

        states = self._get_next_states()

        if not has_max_metric:
            self.max_metric = estimate_max_metric(
                self.model,
                [first_state] + states,
                self.current_metrics,
                max_atoms=self.max_atoms_to_try,
            )
            print(f"Max metric calculated: {self.max_metric}")
        return concatenate_states([first_state] + states), []

    def next_batch(
        self,
        updated_state: BaseState,
        convergence_tensor: torch.Tensor | None = None,
        *,
        return_indices: bool = False,
    ) -> (
        tuple[BaseState, list[BaseState]] | tuple[BaseState, list[BaseState], list[int]]
    ):
        """Get the next batch of states based on convergence.

        Args:
            updated_state: The updated state.
            convergence_tensor: Boolean tensor indicating which states have converged.
            return_indices: Whether to return indices along with the batch.

        Returns:
            The next batch of states.
        """
        # TODO: this needs to be refactored to avoid so
        # many split and concatenate operations, we should
        # take the updated_concat_state and pop off
        # the states that have converged. with the pop_states function

        if convergence_tensor is None:
            if self.iterator_idx > 0:
                raise ValueError(
                    "A convergence tensor must be provided after the "
                    "first batch has been run."
                )
            return self._first_batch()

        # assert statements helpful for debugging, should be moved to validate fn
        # the first two are most important
        assert len(convergence_tensor) == updated_state.n_batches
        assert len(self.current_idx) == len(self.current_metrics)
        assert len(convergence_tensor.shape) == 1
        assert updated_state.n_batches > 0

        completed_idx = torch.where(convergence_tensor)[0].tolist()
        completed_idx.sort(reverse=True)

        remaining_state, completed_states = pop_states(
            updated_state, completed_idx
        )

        self._delete_old_states(completed_idx)
        next_states = self._get_next_states()

        # there are no states left to run, return the completed states
        if not self.current_idx:
            return (
                (None, completed_states, [])
                if return_indices
                else (None, completed_states)
            )

        # concatenate remaining state with next states
        if remaining_state.n_batches > 0:
            next_states = [remaining_state] + next_states
        next_batch = concatenate_states(next_states)

        if return_indices:
            return next_batch, completed_states, self.current_idx

        return next_batch, completed_states

    def restore_original_order(
        self, completed_states: list[BaseState]
    ) -> list[BaseState]:
        """Take the list of completed states and reconstruct the original order.

        Args:
            completed_states: List of completed states to reorder.

        Returns:
            States in their original order.

        Raises:
            ValueError: If the number of completed states doesn't match
            the number of indices.
        """
        # TODO: should act on full states, not state slices

        if len(completed_states) != len(self.completed_idx_og_order):
            raise ValueError(
                f"Number of completed states ({len(completed_states)}) does not match "
                f"number of completed indices ({len(self.completed_idx_og_order)})"
            )

        # Create pairs of (original_index, state)
        indexed_states = list(
            zip(self.completed_idx_og_order, completed_states, strict=False)
        )

        # Sort by original index
        return [state for _, state in sorted(indexed_states, key=lambda x: x[0])]
