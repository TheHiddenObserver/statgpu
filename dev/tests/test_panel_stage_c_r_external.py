"""R plm/sandwich external-alignment gate for Panel Stage C.

The permanent workflow opts into this test explicitly.  Ordinary local/unit
runs skip it so absence of an R installation is not silently treated as
external evidence.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from statgpu.panel import PanelOLS
from statgpu.panel._covariance import ols_covariance


pytestmark = pytest.mark.skipif(
    os.environ.get("STATGPU_RUN_R_PANEL_EXTERNAL") != "1",
    reason="set STATGPU_RUN_R_PANEL_EXTERNAL=1 in the dedicated R external gate",
)


def _dataset(seed=12720):
    rng = np.random.default_rng(seed)
    n_entities, n_times = 12, 6
    entity = np.repeat(np.arange(n_entities), n_times)
    time = np.tile(np.arange(n_times), n_entities)
    x1 = rng.normal(size=entity.size)
    x2 = rng.normal(size=entity.size)
    alpha = np.repeat(rng.normal(scale=0.45, size=n_entities), n_times)
    y = 0.6 + 0.8 * x1 - 0.35 * x2 + alpha
    y += rng.normal(scale=0.25, size=entity.size)
    return y, x1, x2, entity, time


def _write_csv(path: Path, y, x1, x2, entity, time):
    matrix = np.column_stack([y, x1, x2, entity, time])
    np.savetxt(
        path,
        matrix,
        delimiter=",",
        header="y,x1,x2,entity,time",
        comments="",
    )


def test_r_plm_and_sandwich_alignment(tmp_path):
    rscript = shutil.which("Rscript")
    if rscript is None:
        pytest.fail("dedicated R external gate requires Rscript")

    y, x1, x2, entity, time = _dataset()
    data_path = tmp_path / "panel.csv"
    _write_csv(data_path, y, x1, x2, entity, time)

    script_path = tmp_path / "reference.R"
    script_path.write_text(
        r'''
args <- commandArgs(trailingOnly=TRUE)
d <- read.csv(args[1])
prefix <- args[2]

stopifnot(as.character(utils::packageVersion("plm")) == "2.6.7")
stopifnot(as.character(utils::packageVersion("sandwich")) == "3.1.3")

fit_lm <- lm(y ~ x1 + x2, data=d)
for (kind in c("HC0", "HC2", "HC3")) {
  value <- sandwich::vcovHC(fit_lm, type=kind)
  write.table(
    value,
    file=paste0(prefix, "_", tolower(kind), ".csv"),
    sep=",",
    row.names=FALSE,
    col.names=FALSE
  )
}

fit_fe <- plm::plm(
  y ~ x1 + x2,
  data=d,
  index=c("entity", "time"),
  model="within",
  effect="individual"
)
write.table(
  matrix(stats::coef(fit_fe), nrow=1),
  file=paste0(prefix, "_plm_fe_coef.csv"),
  sep=",",
  row.names=FALSE,
  col.names=FALSE
)
writeLines(
  c(
    paste0("R=", R.version.string),
    paste0("plm=", as.character(utils::packageVersion("plm"))),
    paste0("sandwich=", as.character(utils::packageVersion("sandwich")))
  ),
  con=paste0(prefix, "_versions.txt")
)
'''.strip()
        + "\n",
        encoding="utf-8",
    )

    prefix = tmp_path / "r_reference"
    subprocess.run(
        [rscript, str(script_path), str(data_path), str(prefix)],
        check=True,
        text=True,
    )

    X = np.column_stack([np.ones(len(y)), x1, x2])
    params = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ params
    for kind in ("hc0", "hc2", "hc3"):
        actual = ols_covariance(X, resid, cov_type=kind)
        expected = np.loadtxt(
            f"{prefix}_{kind}.csv",
            delimiter=",",
            ndmin=2,
        )
        assert_allclose(actual, expected, rtol=5e-9, atol=5e-11)

    fe = PanelOLS(entity_effects=True, cov_type="nonrobust").fit(
        np.column_stack([x1, x2]),
        y,
        entity_ids=entity,
    )
    expected_fe = np.loadtxt(
        f"{prefix}_plm_fe_coef.csv",
        delimiter=",",
        ndmin=1,
    ).ravel()
    assert_allclose(fe.coef_, expected_fe, rtol=5e-10, atol=5e-11)

    versions = Path(f"{prefix}_versions.txt").read_text(encoding="utf-8")
    assert "plm=2.6.7" in versions
    assert "sandwich=3.1.3" in versions
