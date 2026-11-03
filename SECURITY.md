# Security policy

## Dependency status

The optional `mapping` dependency set uses DGL 2.2.1, the latest Windows wheel
available for the supported LocalMapper release. DGL 2.2.1 is affected by
[GHSA-3x5x-fw77-g54c](https://github.com/advisories/GHSA-3x5x-fw77-g54c), an
unsafe-deserialization issue in DistDGL's distributed RPC service. No patched
DGL release is currently available.

This repository performs only local, single-process atom mapping. Its
`create_localmapper()` entry point replaces the unused `dgl.distributed` import
with a local-only compatibility module before DGL loads. Consequently, the
affected RPC service is not initialized by the supported workflow.

Do not import or expose DistDGL's distributed RPC components from this
environment. If distributed DGL execution is required, use a separately
isolated environment and reassess the upstream advisory first.

## Reporting a vulnerability

Please report suspected vulnerabilities privately through GitHub's security
advisory interface rather than opening a public issue.
