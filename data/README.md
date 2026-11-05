# Flight data and version pointers

This project uses `T_ONTIME_REPORTING.csv`, an on-time reporting extract provided through WGU coursework. Its columns identify US flight schedules, airports and delays. The corresponding public source system is [BTS Reporting Carrier On-Time Performance](https://www.transtats.bts.gov/ONTIME/). The extract supplied with the course is the input used here; it is not a fresh download from BTS.

`src/preprocess.py` derives `cleaned_data.csv` (22,580 Atlanta departures) and the IATA-to-BTS mapping. Both CSVs are committed directly in Git. `.dvc` pointer files record the included files’ MD5 hashes and byte sizes; no remote is configured and no `dvc pull` is needed. Hashes refer to repository bytes, so line-ending conversion can affect a local CSV hash.
