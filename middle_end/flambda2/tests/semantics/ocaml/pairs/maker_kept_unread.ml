(* A control. Reads a field of the second unit that is a constant. The tool's
   Simplify reads the .cmx of the second unit and not that of the first, which
   is on its load path all the same: nothing here reaches the first unit. The
   prefix binds the module block of the second unit, and with it the closure
   that is a field of it, whose code is of the first unit. It declares nothing
   of that code, as the unit of the code was not read, and what the .cmx of the
   second unit says of the code id is not taken: the check of static names fails
   of both terms with that prefix in front of them, and the guard of the tool
   stops the unit ("not checked"). *)

let one = Maker_kept.one
