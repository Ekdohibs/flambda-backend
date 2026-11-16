(* flags: -w -58 *)

(* The twin of maker_kept_unread.ml. Keeps the function of the first unit, which
   the second unit gives again, so the tool's Simplify reads the .cmx of the
   first unit too (the compiler is given that of the second unit only, and warns
   of it, which the flag silences). The prefix binds the same module block and
   the same closure, and the code of that closure from the file of its own unit:
   the check of static names passes. *)

let made = Maker_kept.make
