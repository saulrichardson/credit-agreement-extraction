# Manual Performance-Pricing Extraction (Task 02)

All tables were pulled directly from the raw EX-10 exhibits inside `data/daily_filings/2023/QTR1/20230103.nc.tar.gz`. I extracted the applicable documents into `scratch/manual_pricing_task02/` for traceability.

## Structured grids

| Accession | Table reference | Tier | Metric | Value (%) | Snippet/context |
| --- | --- | --- | --- | --- | --- |
| 0001140361-23-000046 | Article I – "Applicable Margin" | Level I Status | Term Benchmark/RFR/Money Market margin | 0.700% | scratch/manual_pricing_task02/0001140361-23-000046_EX-10.1.html:1459 |
| 0001140361-23-000046 | Article I – "Applicable Margin" | Level I Status | ABR & Canadian Prime margin | 0.000% | scratch/manual_pricing_task02/0001140361-23-000046_EX-10.1.html:1459 |
| 0001140361-23-000046 | Article I – "Applicable Margin" | Level II Status | Term Benchmark/RFR/Money Market margin | 0.725% | scratch/manual_pricing_task02/0001140361-23-000046_EX-10.1.html:1459 |
| 0001140361-23-000046 | Article I – "Applicable Margin" | Level II Status | ABR & Canadian Prime margin | 0.000% | scratch/manual_pricing_task02/0001140361-23-000046_EX-10.1.html:1459 |
| 0001140361-23-000046 | Article I – "Applicable Margin" | Level III Status | Term Benchmark/RFR/Money Market margin | 0.775% | scratch/manual_pricing_task02/0001140361-23-000046_EX-10.1.html:1459 |
| 0001140361-23-000046 | Article I – "Applicable Margin" | Level III Status | ABR & Canadian Prime margin | 0.000% | scratch/manual_pricing_task02/0001140361-23-000046_EX-10.1.html:1459 |
| 0001140361-23-000046 | Article I – "Applicable Margin" | Level IV Status | Term Benchmark/RFR/Money Market margin | 0.850% | scratch/manual_pricing_task02/0001140361-23-000046_EX-10.1.html:1459 |
| 0001140361-23-000046 | Article I – "Applicable Margin" | Level IV Status | ABR & Canadian Prime margin | 0.000% | scratch/manual_pricing_task02/0001140361-23-000046_EX-10.1.html:1459 |
| 0001140361-23-000046 | Article I – "Applicable Margin" | Level V Status | Term Benchmark/RFR/Money Market margin | 1.050% | scratch/manual_pricing_task02/0001140361-23-000046_EX-10.1.html:1459 |
| 0001140361-23-000046 | Article I – "Applicable Margin" | Level V Status | ABR & Canadian Prime margin | 0.050% | scratch/manual_pricing_task02/0001140361-23-000046_EX-10.1.html:1459 |
| 0001140361-23-000046 | Article I – "Applicable Margin" | Level VI Status | Term Benchmark/RFR/Money Market margin | 1.400% | scratch/manual_pricing_task02/0001140361-23-000046_EX-10.1.html:1459 |
| 0001140361-23-000046 | Article I – "Applicable Margin" | Level VI Status | ABR & Canadian Prime margin | 0.400% | scratch/manual_pricing_task02/0001140361-23-000046_EX-10.1.html:1459 |
| 0001140361-23-000046 | Article I – "Facility Fee Rate" | Level I Status | Commitment facility fee | 0.100% | scratch/manual_pricing_task02/0001140361-23-000046_EX-10.1.html:2031 |
| 0001140361-23-000046 | Article I – "Facility Fee Rate" | Level II Status | Commitment facility fee | 0.125% | scratch/manual_pricing_task02/0001140361-23-000046_EX-10.1.html:2031 |
| 0001140361-23-000046 | Article I – "Facility Fee Rate" | Level III Status | Commitment facility fee | 0.150% | scratch/manual_pricing_task02/0001140361-23-000046_EX-10.1.html:2031 |
| 0001140361-23-000046 | Article I – "Facility Fee Rate" | Level IV Status | Commitment facility fee | 0.200% | scratch/manual_pricing_task02/0001140361-23-000046_EX-10.1.html:2031 |
| 0001140361-23-000046 | Article I – "Facility Fee Rate" | Level V Status | Commitment facility fee | 0.250% | scratch/manual_pricing_task02/0001140361-23-000046_EX-10.1.html:2031 |
| 0001140361-23-000046 | Article I – "Facility Fee Rate" | Level VI Status | Commitment facility fee | 0.300% | scratch/manual_pricing_task02/0001140361-23-000046_EX-10.1.html:2031 |
| 0001140361-23-000046 | Article I – "L/C Fee Rate" | Level I Status | Standard L/C issuance fee | 0.700% | scratch/manual_pricing_task02/0001140361-23-000046_EX-10.1.html:2288 |
| 0001140361-23-000046 | Article I – "L/C Fee Rate" | Level II Status | Standard L/C issuance fee | 0.725% | scratch/manual_pricing_task02/0001140361-23-000046_EX-10.1.html:2288 |
| 0001140361-23-000046 | Article I – "L/C Fee Rate" | Level III Status | Standard L/C issuance fee | 0.775% | scratch/manual_pricing_task02/0001140361-23-000046_EX-10.1.html:2288 |
| 0001140361-23-000046 | Article I – "L/C Fee Rate" | Level IV Status | Standard L/C issuance fee | 0.850% | scratch/manual_pricing_task02/0001140361-23-000046_EX-10.1.html:2288 |
| 0001140361-23-000046 | Article I – "L/C Fee Rate" | Level V Status | Standard L/C issuance fee | 1.050% | scratch/manual_pricing_task02/0001140361-23-000046_EX-10.1.html:2288 |
| 0001140361-23-000046 | Article I – "L/C Fee Rate" | Level VI Status | Standard L/C issuance fee | 1.400% | scratch/manual_pricing_task02/0001140361-23-000046_EX-10.1.html:2288 |
| 0001193125-23-000306 | Article I – "Applicable Margin" | Level I | Base Rate margin | 0.250% | scratch/manual_pricing_task02/0001193125-23-000306_EX-10.1.html:2691 |
| 0001193125-23-000306 | Article I – "Applicable Margin" | Level I | Non-Base Rate (Term SOFR) margin | 1.250% | scratch/manual_pricing_task02/0001193125-23-000306_EX-10.1.html:2691 |
| 0001193125-23-000306 | Article I – "Applicable Margin" | Level II | Base Rate margin | 0.500% | scratch/manual_pricing_task02/0001193125-23-000306_EX-10.1.html:2691 |
| 0001193125-23-000306 | Article I – "Applicable Margin" | Level II | Non-Base Rate (Term SOFR) margin | 1.500% | scratch/manual_pricing_task02/0001193125-23-000306_EX-10.1.html:2691 |
| 0001193125-23-000306 | Article I – "Applicable Margin" | Level III | Base Rate margin | 0.750% | scratch/manual_pricing_task02/0001193125-23-000306_EX-10.1.html:2691 |
| 0001193125-23-000306 | Article I – "Applicable Margin" | Level III | Non-Base Rate (Term SOFR) margin | 1.750% | scratch/manual_pricing_task02/0001193125-23-000306_EX-10.1.html:2691 |
| 0001193125-23-000306 | Article I – "Applicable Unused Line Fee Percentage" | Level I | Unused commitment fee | 0.250% | scratch/manual_pricing_task02/0001193125-23-000306_EX-10.1.html:2765 |
| 0001193125-23-000306 | Article I – "Applicable Unused Line Fee Percentage" | Level II | Unused commitment fee | 0.375% | scratch/manual_pricing_task02/0001193125-23-000306_EX-10.1.html:2765 |
| 0001731122-23-000003 | Definition of "Unused Fee Rate" | Pricing Level I | Unused fee rate | 0.250% | scratch/manual_pricing_task02/0001731122-23-000003_EX-10.1.html:4781 |
| 0001731122-23-000003 | Definition of "Unused Fee Rate" | Pricing Level II | Unused fee rate | 0.150% | scratch/manual_pricing_task02/0001731122-23-000003_EX-10.1.html:4781 |

## Accession narratives

### 0001140361-23-000046 (Kimco Realty OP, LLC revolving credit facility)
- Files: `data/daily_filings/2023/QTR1/0001140361-23-000046.nc` → `scratch/manual_pricing_task02/0001140361-23-000046_EX-10.1.html`
- Located the pricing grid by searching for “Applicable Margin” and “Status” within Article I. Lines `scratch/manual_pricing_task02/0001140361-23-000046_EX-10.1.html:1417-1487` show the six-level ratings-based table covering Term Benchmark/RFR/Money Market spreads alongside ABR/Canadian Prime spreads. Status thresholds (Level I–VI) are defined immediately later at lines `2820-2880`, tying each level to S&P/Moody’s unsecured ratings. Facility fee (lines `2031-2080`) and L/C fee tables (`2288-2368`) share the same status grid. Late compliance certificates or Events of Default revert margins to Level VI until cured.

### 0001193125-23-000306 (NOW Inc. revolving credit facility)
- Files: `data/daily_filings/2023/QTR1/0001193125-23-000306.nc` → `scratch/manual_pricing_task02/0001193125-23-000306_EX-10.1.html`
- Searched for “Applicable Margin” and “Fixed Charge Coverage Ratio.” Lines `2671-2735` contain the FCCR-based grid (Levels I–III) with Base Rate vs Non-Base Rate margins. Per the proviso, margins reset monthly once the compliance certificate is delivered; missing certificates or defaults force Level III spreads.
- The unused fee schedule at lines `2748-2785` uses average revolver usage (>35% vs <35%) to determine the Applicable Unused Line Fee Percentage; the amendment also lets Required Lenders set Level II during defaults.

### 0001731122-23-000003 (Landsea Homes – Sixth Amendment)
- Files: `data/daily_filings/2023/QTR1/0001731122-23-000003.nc` → `scratch/manual_pricing_task02/0001731122-23-000003_EX-10.1.html`
- Searched for “Unused Fee Rate” (line `4774`). Schedule defines two pricing levels based on outstanding credit exposure as a percentage of total commitments (`4781-4796`). Amendment text shows the header change from “Unused Commitments” to “Outstanding Credit Exposure,” so the fee steps automatically adjust with actual draw levels.

### 0001193125-23-000849 (Amazon.com term loan) – *no pricing grid*
- File: `scratch/manual_pricing_task02/0001193125-23-000849_EX-10.1.html`. Article I simply sets the Applicable Rate at 0.00% for Base Rate Loans and 0.75% for Term SOFR/Daily Simple SOFR Loans (`1260-1262`), with a slight step-up only if the maturity is extended. No tiered schedule exists.

### 0001628280-23-000083 (Life Time Group) – *no EX-10*
- File: `data/daily_filings/2023/QTR1/0001628280-23-000083.nc` contains only the EX-101XBRL exhibits (see `<TYPE>EX-101.*` entries around lines `56-182`). There is no EX-10 credit agreement to review, so no pricing table is available.

### 0001861449-23-000002 (Bird Global – Amendment No. 8) – *fixed margin*
- File: `scratch/manual_pricing_task02/0001861449-23-000002_EX-10.9.html`. The Loan and Security Agreement definition of “Applicable Margin” is a flat 7.50% (line `79`). No performance-based tiers are provided—only a constant spread for the term loans.

### 0001193125-23-000686 (Stone Point Credit revolver amendment) – *fixed SOFR/ABR spreads*
- File: `scratch/manual_pricing_task02/0001193125-23-000686_EX-10.1.html`. Article I (line `1957`) redefines the Applicable Margin as 1.95% for SOFR Loans and 0.95% for Reference Rate Loans, regardless of leverage or rating. No tiered grid or schedule accompanies the amendment.

### 0001628280-23-000094 (Lifetime Brands LIBOR transition amendment) – *fixed spreads*
- File: `scratch/manual_pricing_task02/0001628280-23-000094_EX-10.01.html`. The amendment simply replaces LIBOR with SOFR and states an Applicable Rate of 3.50% for Term Benchmark/RFR Loans and 2.50% for ABR Loans (single-line paragraph containing “Applicable Rate” near the middle of the document). No leverage- or rating-based schedule appears.
