#!/bin/sh
# run from repository root folder
gitingest . \
  --include-pattern "*.py" \
  --include-pattern "README.md" \
  --include-pattern "requirements.txt" \
  --exclude-pattern "LICENSE" \
  --exclude-pattern "*/__pycache__/*" \
  --output out_gitingest/differ_hug_04_works.txt
