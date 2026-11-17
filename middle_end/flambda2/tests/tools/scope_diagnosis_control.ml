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

(* A control of [Flsemantics.not_in_scope], which finds the words that the test
   tool has for what a term mentions out of scope, on terms made by hand. What
   this prints is compared with ../semantics/guard/scope_diagnosis.expected.

   A prefix that is made from a [.cmx] file gives a symbol with a number and no
   name to a field of which the file says nothing, and the term as it is run has
   that symbol among its imports. [not_in_scope] finds its words by asking the
   check again with every symbol in scope, and with every code id. The symbols
   that it is given must be among "every symbol": otherwise a term that mentions
   such a symbol, and a named symbol that is out of scope, is said to mention "a
   symbol and a code id" out of scope where it mentions no code id at all. No
   text of the tests reaches that: they give the tool a term after that is out
   of scope with [-after], and then no [.cmx] is read and there is no prefix.
   That Simplify makes such a term of a unit that has a prefix is not ruled out;
   no test has one. *)

module Model = Flambda2_rocq_semantics.FlambdaModel

let compilation_unit =
  Compilation_unit.create Compilation_unit.Prefix.empty
    (Compilation_unit.Name.of_string "Scope_diagnosis")

(* A unit that gives the simples to its return continuation, and does nothing
   else. *)
let unit_of (args : Model.simple list) : Flsemantics.converted =
  let return_continuation = Flsemantics.positive_of_int 1 in
  { body =
      Model.Apply_cont
        { Model.ac_continuation = return_continuation;
          ac_args = args;
          ac_trap_action = None
        };
    return_continuation;
    exn_continuation = Flsemantics.positive_of_int 2
  }

let show what ~symbols unit =
  Printf.printf "%s: %s\n" what
    (match Flsemantics.not_in_scope ~symbols ~code_ids:[] unit with
    | None -> "nothing out of scope"
    | Some names -> names ^ " out of scope")

let () =
  Flsemantics.reset_conversion ();
  let hidden = Flsemantics.Symbols.fresh () in
  let named =
    Flsemantics.Symbols.find
      (Import.Symbol.create compilation_unit (Linkage_name.of_string "named"))
  in
  show "a symbol with no name, given as in scope" ~symbols:[hidden]
    (unit_of [Model.Symbol hidden]);
  show "that symbol, and a named symbol that is not in scope" ~symbols:[hidden]
    (unit_of [Model.Symbol hidden; Model.Symbol named]);
  show "a named symbol that is not in scope" ~symbols:[]
    (unit_of [Model.Symbol named])
