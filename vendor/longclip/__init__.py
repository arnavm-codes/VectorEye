"""Vendored Long-CLIP (github.com/beichenzbc/Long-CLIP, Apache-2.0) -- no pip
package exists upstream, so this is copied in directly rather than
reimplemented. Used by app/pipeline/embedder.py when EMBEDDING_BACKEND is
"longclip". Not this project's own code -- see longclip.py/model_longclip.py
for the one documented compatibility fix made to it (pkg_resources ->
packaging), otherwise left as upstream.
"""

from .longclip import *
