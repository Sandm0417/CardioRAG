# Security

This repository is intended for research code release. It must not contain API keys, `.env` files, patient records, or full-text society guidelines.

## What is excluded

- `.env` and any file matching `.env.*` except `.env.example`
- LightRAG vector indexes (`lightrag_index/`)
- Raw guideline PDFs and slide decks
- Generation logs and LLM judge outputs

## How keys are handled

Scripts read keys from the process environment, optionally loading a local `.env` through `cardiorag.env.load_env()`. Missing `.env` is not an error for import or unit tests. Generation and index-building scripts exit with a message if the needed key is absent.

## If a secret is committed

1. Rotate the key at the provider immediately.
2. Remove the file from git history (`git filter-repo` or GitHub secret scanning guidance).
3. Confirm `.gitignore` still excludes `.env`.
