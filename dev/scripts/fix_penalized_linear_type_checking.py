from pathlib import Path

path = Path("statgpu/linear_model/penalized/_penalized_linear.py")
text = path.read_text(encoding="utf-8")
text = text.replace(
    "from typing import Optional, Union\n",
    "from typing import TYPE_CHECKING, Optional, Union\n",
    1,
)
anchor = "from statgpu.linear_model.penalized._base import PenalizedGeneralizedLinearModel\n"
insert = anchor + "\nif TYPE_CHECKING:\n    from statgpu.penalties._base import Penalty\n"
if insert not in text:
    if anchor not in text:
        raise RuntimeError("penalized linear import anchor not found")
    text = text.replace(anchor, insert, 1)
path.write_text(text, encoding="utf-8")
