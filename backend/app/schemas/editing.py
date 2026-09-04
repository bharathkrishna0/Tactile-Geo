from typing import Literal

from pydantic import BaseModel, Field


class ElementEditRequest(BaseModel):
    """A single teacher correction to one semantic element.

    The edit scope is inferred from ``action`` and the targeted element's type:
      - ``set_type``        : change a geometry element's semantic type.
      - ``set_label``       : change a text label's text (Braille re-translated).
      - ``set_association`` : re-bind a label to a geometry element (either the
                              label's ``association_target_id`` or a geometry
                              element's ``association_label_id``).
      - ``set_confidence``  : accept (high confidence) or flag an element as
                              uncertain (low confidence) for review.
      - ``nudge``           : translate coordinates by ``offset``.
      - ``delete``          : remove the element entirely.
    """
    action: Literal["set_type", "set_label", "set_association", "set_confidence", "nudge", "delete"]
    type: str | None = None
    text: str | None = None
    confidence: float | None = None
    association_label_id: str | None = None
    association_target_id: str | None = None
    offset: tuple[int, int] | None = None


class BatchEditItem(ElementEditRequest):
    element_id: str


class BatchElementEditRequest(BaseModel):
    """Apply several corrections in one call, each carrying ``element_id``."""
    edits: list[BatchEditItem] = Field(default_factory=list)