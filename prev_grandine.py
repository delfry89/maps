name: Test Meteo Script

on:
  push:
    branches: [ main, master ]
  pull_request:
    branches: [ main, master ]
  workflow_dispatch:

# Permessi di scrittura fondamentali per fare il commit dei file generati
permissions:
  contents: write

jobs:
  run-script:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout Repository
      uses: actions/checkout@v4

    - name: Setup Miniconda
      uses: conda-incubator/setup-miniconda@v3
      with:
        activate-environment: meteo
        environment-file: environment.yml
        auto-activate-base: false

    - name: Run Script
      shell: bash -el {0}
      run: |
        python prev_grandine.py

    # Commit e Push automatico del file generato
    - name: Commit & Push Changes
      uses: stefanzweifel/git-auto-commit-action@v5
      with:
        commit_message: "Auto-generate mappa riassunto 24h"
        file_pattern: "mappa_riassunto_24h.png downloaded_maps/*.png"
