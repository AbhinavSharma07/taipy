# Copyright 2021-2024 Avaiga Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""
This script assists with dependency management for the project.
It can be used to:
- Ensure consistent package versions across multiple requirements files.
- Generate Pipfile and requirements files with the latest installable versions.
- Display a summary of dependencies that require updates.
"""

import glob
import itertools
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import argparse
import requests
import tabulate
import toml
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


@dataclass
class Release:
    """
    Information about a package release.
    """
    version: str
    upload_date: datetime.date


@dataclass
class Package:
    """
    Information about a package.
    """
    name: str
    min_version: str = ""
    max_version: str = ""
    installation_markers: str = ""
    is_taipy: bool = False
    extras_dependencies: List[str] = field(default_factory=list)
    files: List[str] = field(default_factory=list)
    releases: List[Release] = field(default_factory=list)
    min_release: Optional[Release] = None
    max_release: Optional[Release] = None
    latest_release: Optional[Release] = None

    def __eq__(self, other):
        return isinstance(other, Package) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def load_releases(self, session: requests.Session) -> None:
        """
        Retrieve all releases of the package from PyPI.
        """
        try:
            response = session.get(f"https://pypi.org/pypi/{self.name}/json", timeout=10)
            response.raise_for_status()
            releases_data = response.json().get("releases", {})
        except requests.RequestException as e:
            logger.error(f"Failed to fetch releases for package '{self.name}': {e}")
            return

        for version, info_list in releases_data.items():
            # Skip if no release info or pre/post releases
            if not info_list or any(re.search(r'[a-zA-Z]', c) for c in version):
                continue
            try:
                upload_time = info_list[0]["upload_time"]
                upload_date = datetime.strptime(upload_time, "%Y-%m-%dT%H:%M:%S").date()
                release = Release(version=version, upload_date=upload_date)
                self.releases.append(release)

                if self.min_version and self.min_version == version:
                    self.min_release = release
                if self.max_version and self.max_version == version:
                    self.max_release = release
            except (KeyError, ValueError) as e:
                logger.warning(f"Skipping malformed release data for '{self.name}' version '{version}': {e}")

        self.releases.sort(key=lambda x: x.upload_date, reverse=True)
        self.latest_release = self.releases[0] if self.releases else None

    def as_requirements_line(self, with_version: bool = True) -> str:
        """
        Return the package as a requirements line.
        """
        if self.is_taipy:
            return self.name

        name = self.name
        if self.extras_dependencies:
            name += f'[{",".join(self.extras_dependencies)}]'

        if with_version:
            version_spec = f">={self.min_version},<={self.max_version}"
            if self.installation_markers:
                return f"{name}{version_spec}; {self.installation_markers}"
            return f"{name}{version_spec}"

        if self.installation_markers:
            return f"{name}; {self.installation_markers}"
        return name

    def as_pipfile_line(self) -> str:
        """
        Return the package as a Pipfile line.
        """
        line = f'"{self.name}" = {{version="=={self.max_version}"'

        if self.installation_markers:
            line += f', markers="{self.installation_markers}"'

        if self.extras_dependencies:
            extras = ", ".join(f'"{extra}"' for extra in self.extras_dependencies)
            line += f", extras=[{extras}]"

        line += "}"
        return line

    @classmethod
    def from_requirements(cls, package_line: str, filename: str) -> 'Package':
        """
        Create a Package instance from a requirements line.
        """
        name, min_version, max_version, markers, extras = parse_requirements_line(package_line)
        is_taipy = "taipy" in name.lower()

        return cls(
            name=name.lower(),
            min_version=min_version if not is_taipy else "",
            max_version=max_version if not is_taipy else "",
            installation_markers=markers if not is_taipy else "",
            is_taipy=is_taipy,
            extras_dependencies=extras,
            files=[filename],
        )


def parse_requirements_line(package_line: str) -> (str, str, str, str, List[str]):
    """
    Parse a requirements line using regex to extract package details.

    Returns:
        Tuple containing:
        - name (str)
        - min_version (str)
        - max_version (str)
        - installation_markers (str)
        - extras_dependencies (List[str])
    """
    pattern = re.compile(
        r"""^(?P<name>[A-Za-z0-9_\-]+)              # Package name
            (?:\[(?P<extras>[A-Za-z0-9_,\-]+)\])?  # Optional extras
            \s*
            (?P<version_spec>(?:==|>=|<=|>|<)[^;]+)? # Version specifications
            \s*
            (?:;\s*(?P<markers>.+))?$               # Optional markers
        """,
        re.VERBOSE
    )
    match = pattern.match(package_line)
    if not match:
        raise ValueError(f"Invalid package format: '{package_line}'")

    name = match.group("name")
    extras = match.group("extras").split(",") if match.group("extras") else []
    version_spec = match.group("version_spec") or ""
    markers = match.group("markers") or ""

    min_version = ""
    max_version = ""

    # Extract min and max versions
    if version_spec:
        version_parts = [part.strip() for part in version_spec.split(",")]
        for part in version_parts:
            if part.startswith(">="):
                min_version = part[2:].strip()
            elif part.startswith("=="):
                min_version = max_version = part[2:].strip()
            elif part.startswith("<="):
                max_version = part[2:].strip()
            elif part.startswith("<"):
                max_version = part[1:].strip()

    return name, min_version, max_version, markers, extras


def load_dependencies(requirements_filenames: List[str], enforce_format: bool) -> Dict[str, Package]:
    """
    Load and concatenate dependencies from requirements files.

    Args:
        requirements_filenames (List[str]): List of requirements file paths.
        enforce_format (bool): Whether to enforce package format.

    Returns:
        Dict[str, Package]: Dictionary of package name to Package instance.
    """
    dependencies: Dict[str, Package] = {}

    for filename in requirements_filenames:
        try:
            content = Path(filename).read_text(encoding="UTF-8")
            package_lines = [line.strip() for line in content.splitlines() if line.strip() and not line.startswith("#")]
        except IOError as e:
            logger.error(f"Failed to read file '{filename}': {e}")
            continue

        for package_line in package_lines:
            if enforce_format:
                if not re.search(r">=|<", package_line) and "taipy" not in package_line.lower():
                    logger.error(f"Invalid package format: '{package_line}' in '{filename}'")
                    raise ValueError(f"Invalid package format: '{package_line}' in '{filename}'")

            try:
                package = Package.from_requirements(package_line, filename)
            except ValueError as e:
                logger.error(e)
                continue

            if package.name in dependencies:
                existing_package = dependencies[package.name]
                if (existing_package.min_version != package.min_version or
                        existing_package.max_version != package.max_version):
                    logger.error(
                        f"Inconsistent versions for package '{package.name}' between files: "
                        f"{existing_package.files} and '{filename}'."
                    )
                    raise ValueError(
                        f"Inconsistent versions for package '{package.name}' between files: "
                        f"{existing_package.files} and '{filename}'."
                    )
                existing_package.files.append(filename)
            else:
                dependencies[package.name] = package

    return dependencies


def display_dependencies_versions(dependencies: Dict[str, Package]) -> None:
    """
    Display dependencies information in a tabulated format.

    Args:
        dependencies (Dict[str, Package]): Dictionary of package name to Package instance.
    """
    to_print = []

    with requests.Session() as session:
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_package = {executor.submit(pkg.load_releases, session): pkg for pkg in dependencies.values() if not pkg.is_taipy}
            for future in as_completed(future_to_package):
                pkg = future_to_package[future]
                if pkg.latest_release:
                    to_print.append((
                        pkg.name,
                        f"{pkg.min_version} ({pkg.min_release.upload_date if pkg.min_release else 'N/A'})",
                        f"{pkg.max_version} ({pkg.max_release.upload_date if pkg.max_release else 'N/A'})",
                        f"{pkg.latest_release.version} ({pkg.latest_release.upload_date})",
                        len([r for r in pkg.releases if r.version != pkg.max_version])
                    ))

    to_print.sort(key=lambda x: x[0].lower())
    headers = ["Name", "Min Version", "Max Version", "Latest Version", "Releases Behind"]
    print(tabulate.tabulate(to_print, headers=headers, tablefmt="pretty"))


def update_dependencies(
    dependencies_installed: Dict[str, Package],
    dependencies_set: Dict[str, Package],
    requirements_filenames: List[str],
) -> None:
    """
    Display and update dependencies based on installed versions.

    Args:
        dependencies_installed (Dict[str, Package]): Installed dependencies.
        dependencies_set (Dict[str, Package]): Dependencies defined in requirements files.
        requirements_filenames (List[str]): List of requirements file paths to update.
    """
    to_update = []

    for name, set_pkg in dependencies_set.items():
        if set_pkg.is_taipy:
            continue

        installed_pkg = dependencies_installed.get(name) or dependencies_installed.get(name.replace("-", "_"))
        if installed_pkg and installed_pkg.max_version != set_pkg.max_version:
            to_update.append((name, installed_pkg.max_version, ", ".join(set_pkg.files)))
            set_pkg.max_version = installed_pkg.max_version

    if not to_update:
        logger.info("All dependencies are up to date.")
        return

    # Display dependencies to update
    to_update.sort(key=lambda x: x[0].lower())
    headers = ["Name", "New Version", "Files"]
    print(tabulate.tabulate(to_update, headers=headers, tablefmt="pretty"))

    # Update requirements files
    for filename in requirements_filenames:
        try:
            content = Path(filename).read_text(encoding="UTF-8")
            updated_lines = []
            for line in content.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    updated_lines.append(line)
                    continue

                pkg_name = parse_requirements_line(line)[0].lower()
                pkg = dependencies_set.get(pkg_name)
                if pkg and filename in pkg.files:
                    updated_lines.append(pkg.as_requirements_line())
                else:
                    updated_lines.append(line)
            updated_content = "\n".join(updated_lines) + "\n"
            Path(filename).write_text(updated_content, encoding="UTF-8")
            logger.info(f"Updated '{filename}' with new dependency versions.")
        except IOError as e:
            logger.error(f"Failed to update file '{filename}': {e}")


def generate_raw_requirements_txt(dependencies: Dict[str, Package]) -> None:
    """
    Print the dependencies as requirements lines without version specifications.

    Args:
        dependencies (Dict[str, Package]): Dictionary of package name to Package instance.
    """
    for package in sorted(dependencies.values(), key=lambda p: p.name):
        if not package.is_taipy:
            print(package.as_requirements_line(with_version=False))


def update_pipfile(pipfile_path: str, dependencies_version: Dict[str, Package]) -> None:
    """
    Update the Pipfile in place with the specified dependency versions.

    Args:
        pipfile_path (str): Path to the Pipfile.
        dependencies_version (Dict[str, Package]): Dependencies with updated versions.
    """
    try:
        pipfile_obj = toml.load(pipfile_path)
    except (IOError, toml.TomlDecodeError) as e:
        logger.error(f"Failed to load Pipfile '{pipfile_path}': {e}")
        return

    packages = pipfile_obj.get("packages", {})
    updated_packages = {}

    for name, dep in packages.items():
        pkg = dependencies_version.get(name) or dependencies_version.get(name.replace("-", "_"))
        if pkg:
            pkg.name = name  # Ensure correct casing
            updated_packages[name] = pkg.as_pipfile_line()
        else:
            if isinstance(dep, dict):
                dep_str = toml.dumps({"": dep}).strip().replace('"', '\\"').replace('\n', ', ')
                updated_packages[name] = f'{{version="{dep["version"]}", markers="{dep.get("markers", "")}", extras={dep.get("extras", [])}}}'
            else:
                updated_packages[name] = dep

    # Write updated packages back to Pipfile
    pipfile_obj["packages"] = {}
    toml_str = toml.dumps(pipfile_obj)
    with open(pipfile_path, "w", encoding="UTF-8") as f:
        f.write(f"{toml_str}\n\n[packages]\n")
        for line in updated_packages.values():
            f.write(f"{line}\n")

    logger.info(f"Pipfile '{pipfile_path}' has been updated successfully.")


def fetch_installed_dependencies(requirements_filename: str) -> Dict[str, Package]:
    """
    Load dependencies from an installed environment's requirements file.

    Args:
        requirements_filename (str): Path to the installed requirements file.

    Returns:
        Dict[str, Package]: Dictionary of package name to Package instance.
    """
    return load_dependencies([requirements_filename], enforce_format=False)


def main():
    parser = argparse.ArgumentParser(
        description="Dependency management helper script."
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # ensure-same-version command
    parser_ensure = subparsers.add_parser(
        "ensure-same-version",
        help="Ensure that the same version of each package is set across all requirements files."
    )
    parser_ensure.add_argument(
        "-f", "--files", nargs='+', required=True, help="List of requirements files to check."
    )

    # dependencies-summary command
    parser_summary = subparsers.add_parser(
        "dependencies-summary",
        help="Display a summary of dependencies that need to be updated."
    )
    parser_summary.add_argument(
        "installed_requirements", help="Path to the installed requirements file."
    )
    parser_summary.add_argument(
        "-f", "--files", nargs='+', required=True, help="List of requirements files to compare."
    )

    # generate-raw-requirements command
    parser_generate_raw = subparsers.add_parser(
        "generate-raw-requirements",
        help="Generate a raw requirements.txt without version specifications."
    )
    parser_generate_raw.add_argument(
        "-f", "--files", nargs='+', required=True, help="List of requirements files to process."
    )

    # generate-pipfile command
    parser_generate_pipfile = subparsers.add_parser(
        "generate-pipfile",
        help="Generate or update a Pipfile based on requirements files."
    )
    parser_generate_pipfile.add_argument(
        "pipfile", help="Path to the Pipfile to update."
    )
    parser_generate_pipfile.add_argument(
        "requirements", help="Path to the requirements file to base the Pipfile on."
    )

    args = parser.parse_args()

    if args.command == "ensure-same-version":
        logger.info("Ensuring the same version across all requirements files...")
        dependencies = load_dependencies(args.files, enforce_format=True)
        display_dependencies_versions(dependencies)

    elif args.command == "dependencies-summary":
        logger.info("Generating dependencies summary...")
        dependencies_installed = fetch_installed_dependencies(args.installed_requirements)
        dependencies_set = load_dependencies(args.files, enforce_format=False)
        update_dependencies(dependencies_installed, dependencies_set, args.files)

    elif args.command == "generate-raw-requirements":
        logger.info("Generating raw requirements.txt without version specifications...")
        dependencies = load_dependencies(args.files, enforce_format=False)
        generate_raw_requirements_txt(dependencies)

    elif args.command == "generate-pipfile":
        logger.info("Updating Pipfile based on requirements file...")
        dependencies_version = load_dependencies([args.requirements], enforce_format=False)
        update_pipfile(args.pipfile, dependencies_version)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
