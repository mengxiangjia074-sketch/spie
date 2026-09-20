from abc import ABC, abstractmethod


class LensControlError(Exception):
    pass


class LensControllerPort(ABC):
    @abstractmethod
    def connect(self, device_number):
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError

    @abstractmethod
    def read_ranges(self, capabilities):
        raise NotImplementedError

    @abstractmethod
    def ensure_ready(self, name, init_if_needed):
        raise NotImplementedError

    @abstractmethod
    def move_connected_motor(
        self,
        name,
        target,
        capabilities,
        settle_seconds=0,
        init_if_needed=None,
        clamp_target=False,
    ):
        raise NotImplementedError

