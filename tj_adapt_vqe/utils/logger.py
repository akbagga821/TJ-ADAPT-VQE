import json
import os
import tempfile

import mlflow  # type: ignore
from matplotlib.figure import Figure
from typing_extensions import Any, Self

RUN_DIR = "./runs/"

mlflow.set_tracking_uri(RUN_DIR)


class Logger:
    """
    Logger class that manages all the data wanting to be saved during training
    Will be repurposed later for checkpointing and loading / writing to files
    """

    def __init__(self: Self, run_name: str) -> None:
        self.run_name = run_name

        self.config_options: dict[str, Any] = {}
        self.logged_values: dict[str, list[Any]] = {}

    def start(self: Self) -> None:
        self.run = mlflow.start_run(run_name=self.run_name)

    def end(self: Self) -> None:
        mlflow.end_run()

    def add_config_option(self: Self, name: str, config: Any) -> None:
        """
        Add a new config option from training, example includes which Pool or which Optimizer
        """
        self.config_options[name] = config
        mlflow.log_param(name, config)

    def add_logged_value(
        self: Self, name: str, value: Any, t: int | None = None, file: bool = False
    ) -> None:
        """
        Adds a new logged value to the end of the list of the name
        Examples of logged values include observable values

        Args:
            self (Self): A reference to the current class instance.
            name (str): name of the logged value
            value (Any): actual value of the logged value
            t (int | None, optional): the timestamp of the value, defaults to next successive. Defaults to None.
            file (bool, optional). whether or not the data is more complicated than a scalar and neneds to be placed in a file. Defaults to False.
        """

        if name not in self.logged_values:
            self.logged_values[name] = []

        self.logged_values[name].append(value)

        t = t or len(self.logged_values[name])

        if file:
            file_suffix = "txt"
            if isinstance(value, (list, dict)):
                file_suffix = "json"
            elif isinstance(value, Figure):
                file_suffix = "png"

            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_file = open(os.path.join(tmp_dir, f"{t}.{file_suffix}"), "w")

                if file_suffix == "json":
                    json.dump(value, tmp_file)
                elif file_suffix == "png":
                    value.savefig(tmp_file.name)
                else:
                    tmp_file.write(str(value))

                tmp_file.flush()

                mlflow.log_artifact(tmp_file.name, artifact_path=name)  # type: ignore
        else:
            mlflow.log_metric(name, value, step=t)
