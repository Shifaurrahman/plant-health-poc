import numpy as np
import os

from app.services import image_retrieval
from app.services.embedder import embed_multimodal

feats, labels, paths = image_retrieval._reference_library()

q = embed_multimodal([
    {
        "image": r"data\uploads\e16cf918e334\e1ac93d4762346b3b38324a779429629_bb_003.jpg",
        "text": "Water-soaked spots on the leaves are spreading and turning brown."
    }
])[0]

sims = feats @ q

for i in np.argsort(-sims)[:10]:
    print(f"{sims[i]:.3f}  {labels[i]:<28} {os.path.basename(paths[i])}")