# Copyright 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

# Apache 2.0 License Headers for different file types
APACHE_HEADER_PY = """# Copyright 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

APACHE_HEADER_CSS_JS = """/*
 * Copyright 2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
"""

APACHE_HEADER_HTML = """<!--
Copyright 2025 Intel Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
"""


def add_apache_header_to_file(file_path):
    with open(file_path, "r+") as file:
        content = file.read()
        file.seek(0, 0)

        if file_path.endswith(".py"):
            file.write(APACHE_HEADER_PY + content)
        elif file_path.endswith(".css"):
            file.write(APACHE_HEADER_CSS_JS + content)
        elif file_path.endswith(".js"):
            file.write(APACHE_HEADER_CSS_JS + content)
        elif file_path.endswith(".html"):
            file.write(APACHE_HEADER_HTML + content)


def process_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith((".py", ".css", ".js", ".html")):
                file_path = os.path.join(root, file)
                add_apache_header_to_file(file_path)
                print(f"Added Apache 2.0 header to {file_path}")


if __name__ == "__main__":
    # directory = input("Enter the directory path: ")
    process_directory("..")
