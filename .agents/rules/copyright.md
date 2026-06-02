---
trigger: glob
description: Mandatory guideline for including Copyright notices in source code files.
globs: **/*
---

# Copyright and Licensing Notice

* **Requirement**: In every source code file (new or modified), you must **always ensure the presence**, in its header, of the SPDX identifier comment and copyright notice.
* **Standard Text**: Always use the following text template (adapting as block or line comments for the corresponding file language):

  ``` text
  SPDX-License-Identifier: Apache-2.0
  Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
  ```

* **Best Practices and Placement**: Always place the notice fluidly and professionally at the top of the file. In script files containing a *shebang/hashbang* (e.g., `#!/bin/bash`), place the copyright block immediately below it. Ensure that the comment format does not break the syntax of the current source file. Note that the year 2026 should be replaced with the corresponding year (if applicable), but only in files being edited.
* **Non-applicable Files**: `Cargo.lock`, temporary files, binary files, and auto-generated image assets do not need the header.
