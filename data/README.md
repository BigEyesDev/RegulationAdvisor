# Data Files

Download these before running `scripts/ingest.py`.

## PDFs

| File | Source |
|------|--------|
| `eu_ai_act.pdf` | https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=OJ:L_202401689 |
| `gdpr.pdf` | https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32016R0679 |

## CSVs (create these manually — 30–45 minutes of reading the Act)

### ai_act_timeline.csv
Columns: `deadline_date, article_reference, requirement, applies_to`
Extract from Articles 113 and 85. ~20 rows.

### risk_classification.csv
Columns: `risk_tier, use_case, article_reference, obligations_summary`
Extract from Annex III. ~40 rows.

### penalty_structure.csv
Columns: `violation_type, max_fine_eur, max_fine_percent_turnover, article_reference`
Extract from Article 99. ~10 rows.
