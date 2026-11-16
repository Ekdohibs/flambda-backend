(******************************************************************************
 *                                  OxCaml                                    *
 * -------------------------------------------------------------------------- *
 *                               MIT License                                  *
 *                                                                            *
 * Copyright (c) 2026 Jane Street Group LLC                                   *
 * opensource-contacts@janestreet.com                                         *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included    *
 * in all copies or substantial portions of the Software.                     *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER *
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING    *
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER        *
 * DEALINGS IN THE SOFTWARE.                                                  *
 ******************************************************************************)

(* What [Compare.flambda_units] says of a pair of units that are read from
   texts: for the NAME given, of the units of NAME_a.fl and NAME_b.fl. Where it
   says that the two are different it gives a unit, the first with the names of
   the second where they agree; that unit is printed here with the two printers
   of units, to nowhere, and the line says whether both printed it. The line
   also has the names that the comparison gives to the differences that it
   found.

   names: the two differ in the names of code, of symbols, of variables and of
   continuations only. They are the same.

   four_arguments: the same, with a primitive of four arguments.

   arity: the two differ in the kind of the parameter of code. They are
   different, and the unit given is printed: its code has the code ID that its
   binding has.

   params: code with one parameter and code with two. They are different; the
   comparison does not stop on them.

   value_slot, function_slot: the two project different slots of the same
   closure. They are different: a slot of the first is one slot of the second
   and no other.

   handler_kind, invariant_kind: the two differ in the kind of a parameter of a
   continuation, and of an invariant parameter of a recursive continuation. They
   are different.

   tupled: the code of the second takes its parameters as one tuple. They are
   different.

   code_variant_same, code_variant: the parameter and the result of code have
   the kind of a variant; the same variant in the first pair, a variant with one
   constant constructor fewer in the second of the second pair.

   four_arguments_other: the primitive of four arguments, with another fourth
   argument. They are different.

   alloc_stack: the code of the second may allocate on the stack.
   invariant_count: the recursive continuation of the second has one invariant
   parameter more. They are different; the comparison does not stop on them.

   result_arity: the two differ in the kind of the result of code.
   four_arguments_operator: the primitive of four arguments is for another kind
   of field. They are different.

   value_slots_consistent, function_slots_consistent: the two project the same
   slots of a closure three times, the first and the third time the same one;
   the slots have other names in the second. They are the same.
   value_slots_collision, function_slots_collision: the second projects one slot
   where the first projects two. They are different: two slots of the first are
   not one slot of the second. *)

module P = Flambda2_parser

let parse file =
  match P.Parse_flambda.parse file with
  | Ok unit -> unit
  | Error _ -> Misc.fatal_errorf "Cannot parse %s" file

let is_printed print x =
  let ppf = Format.make_formatter (fun _ _ _ -> ()) (fun () -> ()) in
  match
    print ppf x;
    Format.pp_print_flush ppf ()
  with
  | () -> true
  | exception _ -> false

let () =
  let name = Sys.argv.(1) in
  let file1 = name ^ "_a.fl" in
  let unit1 = parse file1 in
  let unit2 = parse (name ^ "_b.fl") in
  (* As ../tools/fldiff.ml does: the symbols of both are of this unit. *)
  Env.set_current_unit (P.Parse_flambda.make_unit_info ~filename:file1);
  let answer =
    match
      Flambda2_compare.Compare.flambda_units_and_differences unit1 unit2
    with
    | Equivalent, _ -> "the same"
    | Different { approximant }, differences ->
      let raw = is_printed Flambda2_terms.Flambda_unit.print approximant in
      let text =
        is_printed
          (fun ppf unit ->
            P.Print_fexpr.flambda_unit ppf (P.Flambda_to_fexpr.conv unit))
          approximant
      in
      Printf.sprintf "different (%s); the unit given is %s"
        (String.concat "; " differences)
        (if raw && text then "printed" else "NOT PRINTED")
    | exception exn -> "STOPPED: " ^ Printexc.to_string exn
  in
  Printf.printf "%s: %s\n" (Filename.basename name) answer
