# Hack the Snack

This repository is for code, documentation, schemas, and metadata only.

The real cafeteria datasets are under NDA and must never be stored inside this workspace.
## Privacy Boundary

Use a private data directory outside the repository, for example:

```text
C:\Users\<user>\HtS-data
```

Configure that location through the `HTS_DATA_ROOT` environment variable in a local `.env` file.

The repository contains:

- `src/hts/`: application code
- `doc/`: notes, assumptions, and non-sensitive documentation, dataset schemas, column definitions, and metadata templates