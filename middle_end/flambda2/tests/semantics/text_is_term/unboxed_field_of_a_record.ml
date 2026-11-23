(* A read and a write of a field of a record that is an unboxed number, and a
   read of a field of the same record that is a value. The text says of the
   first two that the field is of the flat suffix of the block, with its kind
   ([flat[`float`]], [flat[`imm`]]). It wrote the kind alone, which was not
   read back if it was not [imm], and for [imm], the kind of an unboxed
   integer, was read as that of a tagged integer of the value prefix. So the
   text is the term. *)

type t =
  { a : int;
    mutable b : float#;
    mutable c : int#
  }

let read_float (r : t) = r.b

let write_float (r : t) (x : float#) = r.b <- x

let read_int (r : t) = r.c

let write_int (r : t) (x : int#) = r.c <- x

let read_value (r : t) = r.a
