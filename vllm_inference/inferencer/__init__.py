from .base_inferencer import FoldingInferencer, VissimvaInferencer, VissimTextinstInferencer, MvideoInferencer, NperspectiveInferencer

inferencer_dict = {
    "folding": FoldingInferencer,
    "va": VissimvaInferencer,
    "text_instruct": VissimTextinstInferencer,
    "mvideo": MvideoInferencer,
    "nperspective": NperspectiveInferencer
}