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
   of units, to nowhere, and the line says whether both printed it.

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
   and no other. *)

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
    match Flambda2_compare.Compare.flambda_units unit1 unit2 with
    | Equivalent -> "the same"
    | Different { approximant } ->
      let raw = is_printed Flambda2_terms.Flambda_unit.print approximant in
      let text =
        is_printed
          (fun ppf unit ->
            P.Print_fexpr.flambda_unit ppf (P.Flambda_to_fexpr.conv unit))
          approximant
      in
      Printf.sprintf "different; the unit given is %s"
        (if raw && text then "printed" else "NOT PRINTED")
    | exception exn -> "STOPPED: " ^ Printexc.to_string exn
  in
  Printf.printf "%s: %s\n" (Filename.basename name) answer
