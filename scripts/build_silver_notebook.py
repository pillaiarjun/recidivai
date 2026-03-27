"""
scripts/build_silver_notebook.py
=================================
Generates notebooks/03_silver_pipeline.ipynb for the RecidivAI project.

Run from the project root:
    python3 scripts/build_silver_notebook.py

The generated notebook is designed to run on Databricks Free Edition
(Serverless Spark 4.1.0).  It avoids spark.sparkContext (JVM_ATTRIBUTE_NOT_SUPPORTED
on Serverless) and uses only the spark session object.
"""

import json
import os

# ── helpers ──────────────────────────────────────────────────────────────────

def md(*lines):
    """Return a Markdown cell."""
    source = "\n".join(lines)
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source
    }

def code(*lines):
    """Return a Code cell."""
    source = "\n".join(lines)
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source
    }

# ── cells ────────────────────────────────────────────────────────────────────

cells = []

# ── 0  Title ─────────────────────────────────────────────────────────────────
cells.append(md(
    "# RecidivAI — Phase 3: Silver Layer Pipeline",
    "",
    "**Notebook:** `03_silver_pipeline.ipynb`  ",
    "**Stack:** PySpark · Delta Lake · Databricks Serverless  ",
    "**Reads:** `workspace.recidivism.bronze`  ",
    "**Writes:** `workspace.recidivism.silver`",
    "",
    "---",
    "",
    "### What this notebook does",
    "",
    "This notebook implements the **Silver layer** of the Medallion architecture.",
    "It reads the raw Bronze Delta table, applies a PySpark **ELT** pipeline",
    "(Extract → Load → Transform, inside the platform), and writes a clean,",
    "deduplicated, type-enforced Silver table.",
    "",
    "**Transformations applied:**",
    "1. Enforce an explicit `StructType` schema (no inferred types)",
    "2. Cast columns to their correct types",
    "3. Drop rows where `v_decile_score <= 0` (invalid COMPAS scores)",
    "4. Drop rows with null values in critical modelling columns",
    "5. Deduplicate on `id` (keep first occurrence)",
    "6. Add an `ingestion_timestamp` audit column",
    "7. Write to `workspace.recidivism.silver` as a managed Delta table",
))

# ── 1  Interview concepts ─────────────────────────────────────────────────────
cells.append(md(
    "---",
    "",
    "## 💡 Interview Concept: ELT vs ETL",
    "",
    "| | ETL | ELT |",
    "|---|---|---|",
    "| **Transform happens** | Before load, outside the DB | After load, inside the platform |",
    "| **Raw data preserved?** | No | Yes (Bronze layer) |",
    "| **Best for** | Legacy data warehouses | Cloud lakehouses (Databricks, Snowflake, BigQuery) |",
    "",
    "> **Why we use ELT here:** Cloud object storage is cheap, so we preserve the",
    "> raw Bronze table exactly as received. Spark runs the cleaning *inside* the",
    "> cluster where compute is powerful and data never leaves the platform.",
    "> That is the core ELT philosophy — load raw first, transform lazily.",
))

# ── 2  Interview concept: Lazy evaluation ────────────────────────────────────
cells.append(md(
    "---",
    "",
    "## 💡 Interview Concept: Lazy Evaluation & the Catalyst Optimizer",
    "",
    "PySpark uses **lazy evaluation**: transformations like `.filter()`, `.select()`,",
    "`.cast()`, and `.dropDuplicates()` build a **logical plan** — they do **not**",
    "touch data immediately.",
    "",
    "Only **actions** (`.write`, `.count()`, `.show()`) trigger execution.",
    "",
    "Before execution, Spark passes the plan through the **Catalyst Optimizer**,",
    "which rewrites it for maximum efficiency:",
    "",
    "- **Predicate pushdown** — moves `.filter()` as close to the data source as possible",
    "  so fewer rows are read from disk.",
    "- **Column pruning** — drops columns you never use from the scan.",
    "- **Join reordering** — picks the cheapest join order automatically.",
    "",
    "> **Practical takeaway:** The order you write PySpark transformations often",
    "> doesn't change performance — Catalyst reorders them. But explicit filters",
    "> early in your *code* make intent clearer for readers and reviewers.",
))

# ── 3  Imports ────────────────────────────────────────────────────────────────
cells.append(md(
    "---",
    "",
    "## Section 1 — Imports & Session Verification",
))

cells.append(code(
    "# ── Imports ──────────────────────────────────────────────────────────────",
    "# On Databricks Serverless, `spark` is pre-injected into every notebook.",
    "# Do NOT call spark.sparkContext — it raises JVM_ATTRIBUTE_NOT_SUPPORTED",
    "# on Serverless compute (Spark 4.x with no persistent JVM driver).",
    "",
    "from pyspark.sql import SparkSession",
    "from pyspark.sql import functions as F",
    "from pyspark.sql.types import (",
    "    StructType, StructField,",
    "    IntegerType, StringType, FloatType, LongType,",
    ")",
    "from datetime import datetime, timezone",
    "",
    "# Verify the session is live",
    "print(f\"Spark version : {spark.version}\")",
    "print(f\"App name      : {spark.conf.get('spark.app.name', 'Serverless')}\")",
))

# ── 4  Config ─────────────────────────────────────────────────────────────────
cells.append(md(
    "---",
    "",
    "## Section 2 — Pipeline Configuration",
    "",
    "All table names and paths are defined in one place so the notebook",
    "can be parameterised (e.g. via Databricks Widgets) later.",
))

cells.append(code(
    "# ── Pipeline configuration ────────────────────────────────────────────────",
    "",
    "CATALOG    = \"workspace\"",
    "SCHEMA     = \"recidivism\"",
    "BRONZE_TBL = f\"{CATALOG}.{SCHEMA}.bronze\"",
    "SILVER_TBL = f\"{CATALOG}.{SCHEMA}.silver\"",
    "",
    "# Minimum valid COMPAS decile score (scores <= 0 are sentinel / error values)",
    "MIN_VALID_DECILE = 1",
    "",
    "# Column used to deduplicate (primary key in source data)",
    "DEDUP_KEY = \"id\"",
    "",
    "# Columns that must be non-null for a row to enter the Silver layer",
    "CRITICAL_COLS = [",
    "    \"id\",",
    "    \"is_violent_recid\",",
    "    \"v_decile_score\",",
    "    \"race\",",
    "    \"sex\",",
    "    \"age\",",
    "]",
    "",
    "print(f\"Bronze source : {BRONZE_TBL}\")",
    "print(f\"Silver target : {SILVER_TBL}\")",
))

# ── 5  Explicit schema ────────────────────────────────────────────────────────
cells.append(md(
    "---",
    "",
    "## Section 3 — Explicit StructType Schema",
    "",
    "### 💡 Interview Concept: Why declare the schema explicitly?",
    "",
    "Spark can **infer** a schema automatically, but that has two hidden costs:",
    "",
    "1. **An extra scan of the data** — Spark reads the file/table once to guess",
    "   types, then again to process it. For a 4 TB table that doubles your IO.",
    "2. **Inference can be wrong** — a column with `None` in every row is inferred",
    "   as `StringType`, not the `IntegerType` you need for modelling.",
    "",
    "An explicit `StructType` is **faster, deterministic, and self-documenting**.",
    "In production pipelines it also makes schema drift detectable at ingest time.",
))

cells.append(code(
    "# ── Silver schema ─────────────────────────────────────────────────────────",
    "# We declare every column we want to carry into Silver.",
    "# Columns that exist in Bronze but are not listed here are intentionally",
    "# dropped — they are not needed for modelling (interview answer ready).",
    "",
    "SILVER_SCHEMA = StructType([",
    "    # ── identifiers ───────────────────────────────────────────────────────",
    "    StructField(\"id\",                 IntegerType(), nullable=False),",
    "    StructField(\"name\",               StringType(),  nullable=True),",
    "",
    "    # ── demographics ──────────────────────────────────────────────────────",
    "    StructField(\"sex\",                StringType(),  nullable=False),",
    "    StructField(\"race\",               StringType(),  nullable=False),",
    "    StructField(\"age\",                IntegerType(), nullable=False),",
    "    StructField(\"age_cat\",            StringType(),  nullable=True),",
    "",
    "    # ── charge details ────────────────────────────────────────────────────",
    "    StructField(\"c_charge_degree\",    StringType(),  nullable=True),",
    "    StructField(\"c_charge_desc\",      StringType(),  nullable=True),",
    "",
    "    # ── criminal history ──────────────────────────────────────────────────",
    "    StructField(\"juv_fel_count\",      IntegerType(), nullable=True),",
    "    StructField(\"juv_misd_count\",     IntegerType(), nullable=True),",
    "    StructField(\"juv_other_count\",    IntegerType(), nullable=True),",
    "    StructField(\"priors_count\",       IntegerType(), nullable=True),",
    "",
    "    # ── COMPAS scores (for fairness comparison, not used as features) ─────",
    "    StructField(\"v_decile_score\",     IntegerType(), nullable=False),",
    "    StructField(\"v_score_text\",       StringType(),  nullable=True),",
    "",
    "    # ── target variable ───────────────────────────────────────────────────",
    "    StructField(\"is_violent_recid\",   IntegerType(), nullable=False),",
    "",
    "    # ── audit ─────────────────────────────────────────────────────────────",
    "    StructField(\"ingestion_timestamp\", StringType(), nullable=True),",
    "])",
    "",
    "print(f\"Silver schema has {len(SILVER_SCHEMA.fields)} columns\")",
    "for f in SILVER_SCHEMA.fields:",
    "    print(f\"  {f.name:<28} {str(f.dataType):<16}  nullable={f.nullable}\")",
))

# ── 6  Read Bronze ────────────────────────────────────────────────────────────
cells.append(md(
    "---",
    "",
    "## Section 4 — Read Bronze Delta Table",
    "",
    "> **No action triggered yet.** `.read.table()` builds a logical plan.",
    "> Spark has not read a single row of data at this point.",
))

cells.append(code(
    "# ── Read Bronze ───────────────────────────────────────────────────────────",
    "# spark.read.table() is a *transformation* — it constructs a logical plan.",
    "# No data is scanned until an action is called (count, write, show).",
    "# This is Spark lazy evaluation in practice.",
    "",
    "bronze_df = spark.read.table(BRONZE_TBL)",
    "",
    "print(f\"Bronze table read into logical plan.\")",
    "print(f\"Schema inferred from Delta metadata:\")",
    "bronze_df.printSchema()",
))

# ── 7  Bronze audit ───────────────────────────────────────────────────────────
cells.append(md(
    "---",
    "",
    "## Section 5 — Bronze Audit (Pre-Transform Counts)",
    "",
    "We count the Bronze table **before** any transformations so we can report",
    "exactly how many rows each cleaning step removes — this is a required",
    "quality log in any production pipeline.",
))

cells.append(code(
    "# ── Bronze audit ──────────────────────────────────────────────────────────",
    "# .count() is an ACTION — this is the first point where Spark actually",
    "# reads data from the Bronze Delta table.",
    "",
    "bronze_count = bronze_df.count()",
    "print(f\"{'='*55}\")",
    "print(f\"  BRONZE ROW COUNT (pre-transform): {bronze_count:,}\")",
    "print(f\"{'='*55}\")",
))

# ── 8  Cast types ─────────────────────────────────────────────────────────────
cells.append(md(
    "---",
    "",
    "## Section 6 — Type Casting",
    "",
    "Bronze stores all columns as `StringType` (schema-on-read).  We cast",
    "every column to its correct type here so that downstream Spark SQL queries",
    "and Scikit-learn feature matrices receive the right dtypes.",
    "",
    "> **Interview note:** Type casting in PySpark is a **transformation** —",
    "> lazy, zero cost until an action fires.",
))

cells.append(code(
    "# ── Type casting ──────────────────────────────────────────────────────────",
    "# We select only the columns we want (implicit column pruning) and cast",
    "# each to its target type.  Catalyst will push this selection down to the",
    "# Bronze scan — rows and columns not selected are never read into memory.",
    "",
    "now_utc = datetime.now(timezone.utc).isoformat()",
    "",
    "cast_df = bronze_df.select(",
    "    # identifiers",
    "    F.col(\"id\").cast(IntegerType()),",
    "    F.col(\"name\").cast(StringType()),",
    "",
    "    # demographics",
    "    F.col(\"sex\").cast(StringType()),",
    "    F.col(\"race\").cast(StringType()),",
    "    F.col(\"age\").cast(IntegerType()),",
    "    F.col(\"age_cat\").cast(StringType()),",
    "",
    "    # charge",
    "    F.col(\"c_charge_degree\").cast(StringType()),",
    "    F.col(\"c_charge_desc\").cast(StringType()),",
    "",
    "    # criminal history",
    "    F.col(\"juv_fel_count\").cast(IntegerType()),",
    "    F.col(\"juv_misd_count\").cast(IntegerType()),",
    "    F.col(\"juv_other_count\").cast(IntegerType()),",
    "    F.col(\"priors_count\").cast(IntegerType()),",
    "",
    "    # COMPAS scores",
    "    F.col(\"v_decile_score\").cast(IntegerType()),",
    "    F.col(\"v_score_text\").cast(StringType()),",
    "",
    "    # target",
    "    F.col(\"is_violent_recid\").cast(IntegerType()),",
    "",
    "    # audit column — stamp when this Silver row was created",
    "    F.lit(now_utc).alias(\"ingestion_timestamp\"),",
    ")",
    "",
    "print(\"Type casting transformation registered (lazy — no data read yet).\")",
    "cast_df.printSchema()",
))

# ── 9  Filter invalid scores ──────────────────────────────────────────────────
cells.append(md(
    "---",
    "",
    "## Section 7 — Filter Invalid COMPAS Scores",
    "",
    "The COMPAS `v_decile_score` field should be an integer in [1, 10].",
    "Values `<= 0` indicate missing or invalid scores (sentinel values from",
    "the source system).  Including them would corrupt any fairness analysis",
    "that compares our model's output against COMPAS scores.",
    "",
    "> **Interview note:** This is a **business rule filter** — it encodes domain",
    "> knowledge about what a valid COMPAS score is.  Filters like this belong",
    "> in the **Silver layer** (not Gold) because they correct data quality issues,",
    "> not feature semantics.",
))

cells.append(code(
    "# ── Filter invalid COMPAS scores ──────────────────────────────────────────",
    "# Predicate pushdown: Catalyst will push this filter to the Bronze scan,",
    "# meaning rows with v_decile_score <= 0 are never loaded into memory.",
    "",
    "filtered_df = cast_df.filter(F.col(\"v_decile_score\") >= MIN_VALID_DECILE)",
    "",
    "print(f\"Filter registered: v_decile_score >= {MIN_VALID_DECILE} (lazy).\")",
))

# ── 10  Drop critical nulls ───────────────────────────────────────────────────
cells.append(md(
    "---",
    "",
    "## Section 8 — Drop Rows with Nulls in Critical Columns",
    "",
    "Some source rows may have null values in columns that are essential for",
    "modelling or fairness analysis.  We drop those rows here rather than",
    "imputing (Silver is about validity, Gold is about features).",
))

cells.append(code(
    "# ── Drop critical nulls ───────────────────────────────────────────────────",
    "# dropna(subset=...) only removes rows where ANY of the listed columns is null.",
    "# Rows with nulls in non-critical columns (e.g. c_charge_desc) are kept.",
    "",
    "no_null_df = filtered_df.dropna(subset=CRITICAL_COLS)",
    "",
    "print(f\"Null-drop filter registered for columns: {CRITICAL_COLS} (lazy).\")",
))

# ── 11  Deduplicate ───────────────────────────────────────────────────────────
cells.append(md(
    "---",
    "",
    "## Section 9 — Deduplicate on `id`",
    "",
    "The ProPublica COMPAS dataset should already be unique on `id`, but",
    "deduplication is a standard Silver-layer step for any pipeline that",
    "may be re-run with overlapping data (incremental loads, retries, etc.).",
    "",
    "> **Interview note:** `.dropDuplicates([key])` is deterministic in the sense",
    "> that *a* single row is kept — but which row depends on partition ordering.",
    "> For a keyed dedup with a guaranteed 'keep latest' semantic, you would use",
    "> a window function ranked by a timestamp — something we add in Gold.",
))

cells.append(code(
    "# ── Deduplicate on id ─────────────────────────────────────────────────────",
    "# dropDuplicates is a transformation — still lazy at this point.",
    "# The full logical plan so far:",
    "#   Bronze scan → select + cast → filter score → drop nulls → dedup",
    "# Catalyst will optimise the entire plan before execution.",
    "",
    "deduped_df = no_null_df.dropDuplicates([DEDUP_KEY])",
    "",
    "print(\"Deduplication transformation registered (lazy).\")",
))

# ── 12  Plan inspection ───────────────────────────────────────────────────────
cells.append(md(
    "---",
    "",
    "## Section 10 — Inspect the Catalyst Query Plan",
    "",
    "> **Interview talking point:** `.explain()` lets you see exactly what Catalyst",
    "> built.  Look for `Project` (column selection), `Filter` (pushed predicates),",
    "> and `Scan` nodes.  The order in the plan often differs from the order you",
    "> wrote the transformations — that is Catalyst at work.",
))

cells.append(code(
    "# ── Inspect the optimised plan ────────────────────────────────────────────",
    "# This prints the logical → optimised → physical plan without triggering",
    "# a full data scan.  Safe to run on large datasets.",
    "",
    "print(\"=\" * 60)",
    "print(\"CATALYST OPTIMISED QUERY PLAN\")",
    "print(\"=\" * 60)",
    "deduped_df.explain(mode=\"formatted\")",
))

# ── 13  Silver write ──────────────────────────────────────────────────────────
cells.append(md(
    "---",
    "",
    "## Section 11 — Write Silver Delta Table",
    "",
    "This is the **first and only ACTION** that triggers full pipeline execution.",
    "Spark will:",
    "",
    "1. Scan the Bronze Delta table",
    "2. Apply the optimised plan (cast, filter, drop nulls, dedup)",
    "3. Write the result as a managed Delta table in Unity Catalog",
    "",
    "We use `overwrite` mode so the notebook is idempotent — safe to re-run.",
    "In production, this would typically be `append` with a watermark.",
))

cells.append(code(
    "# ── Write Silver Delta table ──────────────────────────────────────────────",
    "# saveAsTable triggers the FIRST full data scan and execution of the",
    "# entire logical plan.  This is where lazy evaluation meets the real world.",
    "",
    "print(\"Triggering Silver write (first full scan)...\")",
    "",
    "(",
    "    deduped_df",
    "    .write",
    "    .format(\"delta\")",
    "    .mode(\"overwrite\")",
    "    .option(\"overwriteSchema\", \"true\")  # allow schema evolution on re-runs",
    "    .saveAsTable(SILVER_TBL)",
    ")",
    "",
    "print(f\"\\n✅  Silver table written: {SILVER_TBL}\")",
))

# ── 14  Verify ────────────────────────────────────────────────────────────────
cells.append(md(
    "---",
    "",
    "## Section 12 — Verify: Row Counts & Removals",
))

cells.append(code(
    "# ── Read back and verify ──────────────────────────────────────────────────",
    "silver_df    = spark.read.table(SILVER_TBL)",
    "silver_count = silver_df.count()",
    "removed      = bronze_count - silver_count",
    "",
    "print(f\"{'='*55}\")",
    "print(f\"  BRONZE row count (raw)           : {bronze_count:>6,}\")",
    "print(f\"  SILVER row count (clean)         : {silver_count:>6,}\")",
    "print(f\"  Rows removed by ELT pipeline     : {removed:>6,}\")",
    "print(f\"  Retention rate                   : {silver_count/bronze_count:.1%}\")",
    "print(f\"{'='*55}\")",
))

# ── 15  Schema check ──────────────────────────────────────────────────────────
cells.append(md(
    "---",
    "",
    "## Section 13 — Schema Verification",
))

cells.append(code(
    "# ── Schema verification ───────────────────────────────────────────────────",
    "print(\"Silver table schema:\")",
    "silver_df.printSchema()",
    "",
    "# Spot check: v_decile_score distribution — should have no values <= 0",
    "print(\"\\nv_decile_score distribution in Silver:\")",
    "(",
    "    silver_df",
    "    .groupBy(\"v_decile_score\")",
    "    .count()",
    "    .orderBy(\"v_decile_score\")",
    "    .show(15)",
    ")",
))

# ── 16  Target distribution ───────────────────────────────────────────────────
cells.append(md(
    "---",
    "",
    "## Section 14 — Target Distribution in Silver",
))

cells.append(code(
    "# ── Target distribution ───────────────────────────────────────────────────",
    "total = silver_count",
    "print(\"Target (is_violent_recid) distribution in Silver:\")",
    "(",
    "    silver_df",
    "    .groupBy(\"is_violent_recid\")",
    "    .agg(",
    "        F.count(\"*\").alias(\"count\"),",
    "        F.round(F.count(\"*\") / F.lit(total) * 100, 2).alias(\"pct\"),",
    "    )",
    "    .orderBy(\"is_violent_recid\")",
    "    .show()",
    ")",
))

# ── 17  Demographic check ─────────────────────────────────────────────────────
cells.append(md(
    "---",
    "",
    "## Section 15 — Demographic Breakdown (Silver Sanity Check)",
    "",
    "We verify that racial group proportions are preserved after cleaning.",
    "A large shift from Bronze → Silver would signal that our filters are",
    "accidentally removing rows from one demographic group disproportionately",
    "— a critical fairness concern.",
))

cells.append(code(
    "# ── Demographic breakdown ─────────────────────────────────────────────────",
    "print(\"Race distribution in Silver vs Bronze:\")",
    "",
    "silver_race = (",
    "    silver_df",
    "    .groupBy(\"race\")",
    "    .agg(",
    "        F.count(\"*\").alias(\"silver_n\"),",
    "        F.round(F.count(\"*\") / F.lit(silver_count) * 100, 2).alias(\"silver_pct\"),",
    "    )",
    "    .orderBy(F.desc(\"silver_n\"))",
    ")",
    "",
    "bronze_race = (",
    "    bronze_df",
    "    .groupBy(\"race\")",
    "    .agg(",
    "        F.count(\"*\").alias(\"bronze_n\"),",
    "        F.round(F.count(\"*\") / F.lit(bronze_count) * 100, 2).alias(\"bronze_pct\"),",
    "    )",
    ")",
    "",
    "silver_race.join(bronze_race, on=\"race\", how=\"left\").orderBy(F.desc(\"silver_n\")).show()",
))

# ── 18  Describe History ──────────────────────────────────────────────────────
cells.append(md(
    "---",
    "",
    "## Section 16 — Delta `DESCRIBE HISTORY` (Audit Trail)",
    "",
    "Delta Lake maintains a full transaction log.  `DESCRIBE HISTORY` is how",
    "you prove to an interviewer (or a data auditor) exactly when the table",
    "was written, by what process, and with what row count.",
))

cells.append(code(
    "# ── Delta transaction history ─────────────────────────────────────────────",
    "print(\"Delta transaction history for Silver table:\")",
    "spark.sql(f\"DESCRIBE HISTORY {SILVER_TBL}\").select(",
    "    \"version\", \"timestamp\", \"operation\", \"operationParameters\", \"userEmail\"",
    ").show(5, truncate=80)",
))

# ── 19  Summary ───────────────────────────────────────────────────────────────
cells.append(md(
    "---",
    "",
    "## Section 17 — Pipeline Summary & Interview Talking Points",
    "",
    "### What this notebook demonstrates",
    "",
    "| Concept | Where in this notebook |",
    "|---|---|",
    "| **ELT pattern** | Raw Bronze preserved; transform inside Spark |",
    "| **Lazy evaluation** | Sections 4–9 build a plan; Section 11 executes it |",
    "| **Catalyst Optimizer** | Section 10: predicate pushdown, column pruning |",
    "| **Explicit StructType** | Section 3: declared schema, not inferred |",
    "| **Delta ACID** | Section 16: full transaction history |",
    "| **Fairness awareness** | Section 15: demographic parity check post-clean |",
    "",
    "### Key numbers to quote in your interview",
    "",
    "- Bronze → Silver row retention rate (see Section 12)",
    "- Number of rows removed by invalid score filter",
    "- Racial group proportions unchanged after cleaning (Section 15)",
    "",
    "**Next step:** Phase 4 — Gold layer feature engineering",
    "(`notebooks/04_gold_features.ipynb`)",
))

# ── Notebook assembly ─────────────────────────────────────────────────────────

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0",
        },
    },
    "cells": cells,
}

# ── Write ─────────────────────────────────────────────────────────────────────

OUTPUT_PATH = os.path.join("notebooks", "03_silver_pipeline.ipynb")
os.makedirs("notebooks", exist_ok=True)

with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
    json.dump(notebook, fh, indent=2, ensure_ascii=False)

print(f"\n✅  Notebook written → {OUTPUT_PATH}")
print(f"   Cells  : {len(cells)}")
print(f"   Sections covered:")
print(f"     1  Imports & session verification")
print(f"     2  Pipeline configuration")
print(f"     3  Explicit StructType schema")
print(f"     4  Read Bronze Delta table")
print(f"     5  Bronze audit (pre-transform counts)")
print(f"     6  Type casting")
print(f"     7  Filter invalid COMPAS scores (v_decile_score <= 0)")
print(f"     8  Drop critical nulls")
print(f"     9  Deduplicate on id")
print(f"    10  Catalyst query plan inspection")
print(f"    11  Write Silver Delta table")
print(f"    12  Row count verification")
print(f"    13  Schema verification")
print(f"    14  Target distribution")
print(f"    15  Demographic fairness sanity check")
print(f"    16  Delta DESCRIBE HISTORY")
print(f"    17  Interview talking points summary")
print(f"\nNext step:")
print(f"  1. Copy scripts/build_silver_notebook.py to your project root")
print(f"  2. cd ~/Desktop/recidivism-prediction")
print(f"  3. python3 scripts/build_silver_notebook.py")
print(f"  4. Upload notebooks/03_silver_pipeline.ipynb to Databricks")
