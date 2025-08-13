from pathlib import Path


class ScenarioBashWriter:
    """Class to write Bash scripts to call kedro scenarios."""

    def __init__(self, name: str, command: str) -> None:
        self.name = name
        assert command in ["salloc", "sbatch", "local"]
        self.command = command

    def build_slurm_request(self, resources: dict, reporting: dict = None) -> str:
        """Build the SLURM request in `sbatch` or `salloc` form."""
        # Set up resources
        if resources is None:
            raise RuntimeError(
                "Requested resources must be specified when the command is 'sbatch' or 'salloc'."
            )
        res_opts = resources
        if self.command == "sbatch":
            prefix = ""
            template = "#SBATCH --KEY=VALUE\n"
            res_opts.update({"job-name": self.name})
            if reporting is not None:
                res_opts.update(reporting)
        elif self.command == "salloc":
            prefix = "salloc"
            template = " --KEY=VALUE"
            res_opts.pop("array", None)  # Remove the array argument if it's given
        slurm_opts = self.build_opts(d=res_opts, template=template)
        slurm_opts = "".join([prefix, slurm_opts])
        return slurm_opts

    def build_kedro_run(
        self,
        params: dict,
        prefix: str = "",
        n_tasks: int = 1,
    ) -> str:
        """Build the bash script to dynamically select the kedro environment."""
        # If we're doing scenario runs, then find the appropriate config environment
        conf_lines = ["cd conf"]
        scen_pth = Path("scenarios") / self.name
        conf_dir_cmd = (
            f'conf_dir=$(find {scen_pth} -type d -name "task_$SLURM_ARRAY_TASK_ID")'
        )
        conf_lines.append(conf_dir_cmd)
        conf_lines.append("cd ..")
        conf_finder = " && ".join(conf_lines)

        if params is None:
            params = {}
        params.update({"env": "$conf_dir"})

        # Write kedro call
        kedro_opts = self.build_opts(d=params, template=" --KEY=VALUE")
        cmds = f"{prefix} kedro run{kedro_opts}"

        lines = []
        if self.command == "sbatch":
            lines.append(conf_finder)
            lines.append(cmds)
        else:
            lines.append(f"for SLURM_ARRAY_TASK_ID in `seq 0 {n_tasks - 1}`; do")
            lines.append(f"\t{conf_finder}")
            lines.append(f"\t{cmds}")
            lines.append("done")
        kedro_runner = "\n".join(lines)
        return kedro_runner

    def compile(
        self,
        params: dict,
        configs: dict = None,
        resources: dict = None,
        reporting: dict = None,
    ) -> str:
        """Compile the commands into a single multi-line string."""
        lines = ["#!/bin/bash"]
        if self.command in ["sbatch", "salloc"]:
            slurm_opts = self.build_slurm_request(
                resources=resources, reporting=reporting
            )
            lines += [slurm_opts]

        if configs is None:
            n_tasks = None
        else:
            n_tasks = len(configs)
        kedro_run = self.build_kedro_run(
            prefix=params["prefix"],
            params=params["kedro"],
            n_tasks=n_tasks,
        )
        lines += [kedro_run]
        sh = "\n".join(lines)
        return sh

    @staticmethod
    def build_opts(d: dict[str : str | int | float], template: str) -> str:
        """Build options based on a dictionary of key, value pairs and a string
        template."""
        opt_lines = []
        for k, v in d.items():
            line = template.replace("KEY", str(k))
            line = line.replace("VALUE", str(v))
            opt_lines.extend(line)
        out_str = "".join(opt_lines)
        return out_str


def generate_bash_script(
    command: str,
    scenario: dict,
    cmd_params: dict = None,
    configs: dict = None,
    resources: dict = None,
    reporting: dict = None,
) -> dict[Path:str]:
    """Generate Bash script for running many scenarios."""
    scen_name = scenario["name"]
    writer = ScenarioBashWriter(name=scen_name, command=command)
    if configs is not None:
        cfgs = {k: v for k, v in configs.items() if k.startswith(scen_name)}
    else:
        cfgs = None
    sh = writer.compile(
        params=cmd_params, configs=cfgs, resources=resources, reporting=reporting
    )
    return {scen_name: sh}
