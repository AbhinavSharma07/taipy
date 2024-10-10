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


import logging
import os
import sys
from pathlib import Path


def test_import_taipy_packages() -> bool:
    """
    Import taipy package and validate the presence of required attributes.
    """
    import taipy as tp

    required_attributes = ["gui", "Scenario", "rest"]
    missing_attributes = [attr for attr in required_attributes if not hasattr(tp, attr)]

    if missing_attributes:
        logging.error(f"Taipy installation is missing attributes: {', '.join(missing_attributes)}")
        return False

    return True


def is_taipy_gui_install_valid() -> bool:
    """
    Validate the existence of necessary Taipy GUI components.
    """
    import taipy

    taipy_gui_core_path = Path(taipy.__file__).absolute().parent / "gui_core" / "lib" / "taipy-gui-core.js"
    taipy_gui_webapp_path = Path(taipy.__file__).absolute().parent / "gui" / "webapp"

    if not taipy_gui_core_path.exists():
        logging.error(f"File {taipy_gui_core_path} not found in Taipy installation path.")
        return False

    if not taipy_gui_webapp_path.exists():
        logging.error(f"Taipy GUI webapp path {taipy_gui_webapp_path} not found.")
        return False

    if not any(fname.endswith(".js") for fname in os.listdir(taipy_gui_webapp_path)):
        logging.error("No JavaScript (.js) files found inside the Taipy GUI webapp folder.")
        return False

    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    logging.info("Validating Taipy installation and main attributes...")
    if not test_import_taipy_packages() or not is_taipy_gui_install_valid():
        sys.exit(1)

    logging.info("Taipy installation validated successfully.")
    sys.exit(0)
