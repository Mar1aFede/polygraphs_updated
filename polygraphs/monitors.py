"""
Monitoring infrastructure
"""
import os
import abc
import torch
import h5py

from . import timer


class BasicHook(metaclass=abc.ABCMeta):
    """
    Abstract periodic monitor
    """

    def __init__(self, interval=1, atend=True):
        super().__init__()
        self._interval = interval
        self._atend = atend
        # Last processed step (to avoid duplicate runs at end)
        self._last = None

    def _isvalid(self, step):
        return step == 1 or step % self._interval == 0

    def _islast(self, step):
        return self._last and self._last == step

    def _run(self, step, polygraph):
        raise NotImplementedError

    def mayberun(self, step, polygraph, interval_label=None):
        """
        Monitors progress at given simulation step.
        """
        if not self._isvalid(step):
            return
        # Store last processed step
        self._last = step
        # User-defined run method with interval_label
        self._run(step, polygraph, interval_label=interval_label)

    def conclude(self, step, polygraph):
        """
        Concludes monitoring.
        """
        if not self._atend or self._islast(step):
            return
        self._run(step, polygraph)


class MonitorHook(BasicHook):
    """
    Periodic monitor for performance measurements
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Timer that starts after the first step
        self._clock = timer.Timer()

    def _run(self, step, polygraph, interval_label=None):
        # Compute throughput
        if not self._clock.isrunning():
            assert step == 1
            self._clock.start()
            throughput = 0.0
        else:
            dt = self._clock.lap()
            throughput = ((step - 1) / dt) / 1000.0
        
        if 'beliefs' not in polygraph.ndata:
            print(f"[ERROR] Beliefs tensor is missing at step {step}.")
            return

        beliefs = polygraph.ndata["beliefs"]
        
        if beliefs.numel() == 0:
            print(f"[ERROR] Beliefs tensor is empty at step {step}.")
            return

        a = torch.sum(torch.le(beliefs, 0.5))  # A: belief <= 0.5
        b = torch.sum(torch.gt(beliefs, 0.5))  # B: belief > 0.5
        
        # Log the beliefs with the interval label
        if interval_label:
            print(f"Beliefs at step {step} (Interval {interval_label}): A/B = {a.item()}/{b.item()}")
        else:
            print(f"Beliefs at step {step}: A/B = {a.item()}/{b.item()}")
        
        # Log progress with the interval label
        msg = f"[MON] Interval {interval_label}: step {step:04d}"
        msg = f"{msg} Ksteps/s {throughput:6.2f}"
        msg = f"{msg} A/B {a / (a + b):4.2f}/{b / (a + b):4.2f}"
        print(msg)


class SnapshotHook(BasicHook):
    """
    Periodic logger for agent beliefs
    """

    def __init__(self, messages=False, location=None, filename="data.hd5", **kwargs):
        super().__init__(**kwargs)
        # Store snapshots in user-specified directory
        assert location, "Location for saving snapshots must be specified."
        self._location = location
        # Construct HDF5 filename
        self._filename = filename
        # Whether to snapshot messages or not
        self._messages = messages

    def _run(self, step, polygraph, interval_label=None):
        # Ensure that the directory exists
        filepath = os.path.join(self._location, f"{interval_label}.hd5")
        directory = os.path.dirname(filepath)
        
        # Create the directory if it does not exist
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        try:
            # Open the HDF5 file for writing
            with h5py.File(filepath, "a") as f:
                # Store beliefs
                beliefs = polygraph.ndata["beliefs"].numpy()
                # Create or modify group
                grp = f.require_group("beliefs")

                # Use interval_label if provided to make dataset names unique for each interval
                dataset_name = f"{interval_label}_{step}" if interval_label else str(step)

                # Check if the dataset already exists, and if so, delete it
                if dataset_name in grp:
                    del grp[dataset_name]

                # Create new dataset
                grp.create_dataset(dataset_name, data=beliefs)

                # Store messages if enabled
                if self._messages:
                    payoffs = polygraph.ndata["payoffs"].numpy()
                    grp_payoffs = f.require_group("payoffs")
                    if dataset_name in grp_payoffs:
                        del grp_payoffs[dataset_name]
                    grp_payoffs.create_dataset(dataset_name, data=payoffs)

        except Exception as e:
            print(f"Error while creating or writing to HDF5 file: {e}")
            raise e

        # Close file
        f.close()
