"""Access to global configuration variables as defined in params.yaml"""

from typing import Optional, TypeAlias, Union, Any
from pathlib import Path
import logger
import sys
import yaml


global log
log = logger.set_logger()


PathOrStr: TypeAlias = Union[Path, str]


def assert_path(path: PathOrStr) -> Path:
    """Assert that file at `path` exists and return it as a Path object."""
    path = Path(path)
    if not path.exists():
        log.critical(f"{path} does not exist")
        sys.exit(1)
    return path


class Config:

    config: dict
    root_path: Path

    def __init__(self,
                 yaml_file: Optional[PathOrStr] = None,
                 root_node: str = 'preprocess') -> None:
        """
        Access to the configuration as defined in the `yaml_file`.

        If no argument is given, look for a YAML file called `params.yaml`
        in the project's root directory.  The root directory
        is determined by locating the directory `.git`.
        """
        if not yaml_file:
            root = self.get_root_path()
            results = list(root.glob('params.yaml'))
            if results:
                yaml_file = results[0]
            if not yaml_file:
                log.error(f"Could not locate configuration file {yaml_file}")
                sys.exit(1)
        config_path = Path(yaml_file)
        self.root_path = config_path.parent
        self.config = dict()
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        if not self.config.get(root_node):
            log.error(f"YAML configuration file {config_path}' has no root node '{root_node}'")
            sys.exit(1)
        self.config = self.config[root_node]

    def get_root_path(self) -> Path:
        """
        Find the root directory which as a `.git` directory.
        If none is found, return the current directory.
        """
        path = Path.cwd()
        for dir in path.resolve().parents:
            if (dir / ".git").exists():
                path = dir
        return path

    def get_param(self, tree_path: str, separator: str = '/',
                  log_not_found: bool = False) -> Any:
        """
        Find the configuration parameter defined by the 'tree-path'.

        Example:
                    Config().get_param('image-sources/raw')

        Args:

           tree-path: String where  a slash designats a child node.
           separator: string separating the nodes, defaults to '/'
           log_not_found: Log an error if `tree_path` yields None.

        Returns:

           Any value associated with this path, or None.
        """
        paths = tree_path.strip(separator).split(separator)
        val = None
        config = self.config
        while paths and type(config) is dict:
            val = config.get(paths[0])
            if type(val) is dict:
                config = val
            paths = paths[1:]
        if not val and log_not_found:
            log.error(f"Could not find configuration value associated with {tree_path}")
        return val

    def get_image_segmented(self) -> Path:
        """Return path to directory with segmented images."""
        return self.root_path / self.get_param("images/segmented", log_not_found=True)

    def get_image_bw(self) -> Path:
        """Return path to directory with b/w images."""
        return self.root_path / self.get_param("images/bw", log_not_found=True)

    def get_index(self) -> Path:
        """Return path to .csv file with index data."""
        return self.root_path / self.get_param("index", log_not_found=True)
