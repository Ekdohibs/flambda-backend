(* flags: -w -58 *)

(* Calls, at toplevel, a function of another unit that calls a function of a
   third, whose code has no body in its .cmx. The compiler is given the .cmx of
   the second unit and not that of the third (it warns of that, which the flag
   silences). The tool finds both: the unit of the third is read, and the prefix
   declares its code deleted. *)

let three = Never_called.add_two 1
