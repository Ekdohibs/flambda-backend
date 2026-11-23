(** The shape of a block from the kinds of its fields, for the text of a term,
    which writes the kinds and not the shape: of the fields of a constructor in
    the kind of a value, and of a static block that has unboxed fields.

    The fields that are values come first. If all are values, the shape is
    [Value_only]. If not, it is that of a mixed block whose flat suffix begins
    at the first field that is not a value, each field of the suffix being the
    element that goes with its kind. [Error] with the reason if a value stands
    after a field that is not one, or if a field is of a kind that a block has
    not got.

    For a [Mixed_record] shape returned here, passing its
    [Flambda_kind.Mixed_block_shape.field_kinds] back to [scannable] gives the
    same shape again. *)
val scannable :
  Flambda_kind.t list -> (Flambda_kind.Scannable_block_shape.t, string) result
