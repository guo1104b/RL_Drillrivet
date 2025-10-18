from collections import deque
import time

# phase_id={stepfwd:0, rotate:1, movehand:2, resethand:3, stepside:4}
class ActionRunner:
    def __init__(self, sim):
        self.sim = sim
        self.queue = deque()
        self.current = None
        self.t0 = None
        self._phase_id = None
        self._target_id = None

    def get_phase_id(self, default: int = -1) -> int:
        return self._phase_id if self._phase_id is not None else default

    def get_target_id(self, default: int = -1) -> int:
        return self._target_id if self._target_id is not None else default

    def enqueue(self, do_signal: str, done_signal: str, timeout: float = 30.0,
                ints: dict = None, strings: dict = None,
                phase_id: int  = None, target_id: int = None):
        """
        Enqueue an action：
        - do_signal/done_signal
        - ints:  {Signal Name: Value}    -> setIntegerSignal
        - strings: {Signal Name: String} -> setStringSignal
        At the moment the action is started, strings/ints will be written in batches first,
        and then do will be launched.
        """
        self.queue.append({
            "kind": "single",
            "do": do_signal,
            "done": done_signal,
            "timeout": float(timeout),
            "ints": ints or {},
            "strings": strings or {},
            "meta": {"phase_id": phase_id, "target_id": target_id},
        })

    def enqueue_wait(self, seconds: float):
        self.queue.append({"kind": "wait", "seconds": float(seconds)})

    def is_busy(self):
        return self.current is not None

    def clear(self):
        self.queue.clear()
        self.current = None
        self.t0 = None
        self._phase_id = None
        self._target_id = None

    def _apply_params_then_do(self, item):
        meta = item.get("meta", {})
        if meta.get("phase_id") is not None:
            self._phase_id = int(meta["phase_id"])
            item["_set_phase"] = True  # 记录：这个动作设置过 phase
        else:
            item["_set_phase"] = False

        if meta.get("target_id") is not None:
            self._target_id = int(meta["target_id"])
            item["_set_target"] = True  # 记录：这个动作设置过 target
        else:
            item["_set_target"] = False

        self.sim.clearIntegerSignal(item["done"])
        for k, v in item["strings"].items():
            self.sim.setStringSignal(k, str(v))
        for k, v in item["ints"].items():
            self.sim.setIntegerSignal(k, int(v))
        self.sim.setIntegerSignal(item["do"], 1)

    def tick(self):
        now = time.time()

        if self.current is None and self.queue:
            self.current = self.queue.popleft()
            self.t0 = now

            if self.current["kind"] == "single":
                self._apply_params_then_do(self.current)
                return ("started", self.current["do"])

            elif self.current["kind"] == "wait":
                return ("started", f"wait_{self.current['seconds']}s")

        if self.current is not None:
            kind = self.current["kind"]

            if kind == "single":
                if self.sim.getIntegerSignal(self.current["done"]) == 1:
                    self.sim.clearIntegerSignal(self.current["done"])
                    # if self.current.get("_set_phase", False):
                    #     self._phase_id = None
                    # if self.current.get("_set_target", False):
                    #     self._target_id = None
                    done_name = self.current["done"]
                    self.current = None; self.t0 = None
                    return ("finished", done_name)

                if now - self.t0 > self.current["timeout"]:
                    if self.current.get("_set_phase", False):
                        self._phase_id = None
                    if self.current.get("_set_target", False):
                        self._target_id = None
                    bad = self.current
                    self.current = None; self.t0 = None
                    return ("timeout", bad)
                return None

            elif kind == "wait":
                if now - self.t0 >= self.current["seconds"]:
                    secs = self.current["seconds"]
                    self.current = None; self.t0 = None
                    return ("finished", f"wait_{secs}s")
                return None

        return None