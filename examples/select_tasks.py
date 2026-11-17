"""Freeze a per-horizon choice of ONNX and reference outputs in a new model file."""
import argparse
import hashlib
from pathlib import Path
import onnx


def select(source,output,masks):
    if output.exists():raise FileExistsError(output)
    graph=onnx.load(source)
    shape=graph.graph.output[0].type.tensor_type.shape.dim
    if len(shape)!=3 or shape[2].dim_value!=6 or len(masks)!=shape[1].dim_value:
        raise ValueError("Supply one mask per forecast horizon")
    if any(m<0 or m>15 for m in masks):raise ValueError("Task masks must be 0..15")
    meta={p.key:p.value for p in graph.metadata_props}
    if not {"dn3.contract","dn3.zero_probe"}<=meta.keys():raise ValueError("Missing DN3 contract")
    meta.update({"dn3.tasks":" ".join(map(str,masks)),
                 "dn3.parent_sha256":hashlib.sha256(source.read_bytes()).hexdigest()})
    onnx.helper.set_model_props(graph,meta)
    onnx.checker.check_model(graph)
    with output.open("xb") as f:f.write(graph.SerializeToString())


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("masks",type=int,nargs="+",help="Bit sum: 1 point, 2 quantiles, 4 direction, 8 variance; zero uses reference")
    a=p.parse_args()
    select(a.source,a.output,a.masks)
