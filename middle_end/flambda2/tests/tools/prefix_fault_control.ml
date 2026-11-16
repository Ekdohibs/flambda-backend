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

(* Two controls of a prefix that declares code deleted, on prefixes and units
   made by hand. What this prints is compared with
   ../semantics/prefix_fault/control.expected.

   The first is of [Flsemantics.prefix_fault], the test that the tool makes of a
   prefix before it uses one. A prefix declares deleted the code of another unit
   that it has from a file and does not bind, and no source and no text makes a
   unit that binds, or declares deleted, a code id of another unit: so nothing
   that the tool is given reaches the refusals shown.

   The second is of the steps of a run. A declaration binds nothing, and the
   group that holds it is one step of a run all the same. So a unit that has a
   prefix of declarations alone, where it had no prefix, ends as it did after
   one step more. The tool counts the steps of a unit with its prefix in front
   of it against the steps that it allows ([Flsemantics.fuel]), and a run that
   has not ended at that count is "still running": the one run that such a
   prefix changes in more than its count is a run which ended, without it, at
   the last step allowed. It is shown last, on a unit of that many groups that
   bind nothing. *)

module Model = Flambda2_rocq_semantics.FlambdaModel

let code_id = Flsemantics.positive_of_int 1

let other_code_id = Flsemantics.positive_of_int 2

let return_continuation = Flsemantics.positive_of_int 1

let exn_continuation = Flsemantics.positive_of_int 2

(* A unit whose static bindings are one group, or none. *)
let unit_of (bindings : Model.static_binding list) : Flsemantics.converted =
  { body =
      (match bindings with
      | [] -> Invalid
      | _ :: _ -> Let_static (bindings, Invalid));
    return_continuation;
    exn_continuation
  }

let code : Model.code =
  Code
    ( [],
      Flsemantics.positive_of_int 1,
      return_continuation,
      exn_continuation,
      Model.make_code_metadata [] [],
      Invalid )

let prefix groups = { Flsemantics.no_prefix with prefix_groups = groups }

let show what prefix before after =
  Printf.printf "%s: %s\n" what
    (match Flsemantics.prefix_fault prefix ~imports:[] before after with
    | None -> "no fault"
    | Some fault -> "the prefix " ^ fault)

let () =
  let declares = prefix [[Model.Bind_deleted_code code_id]] in
  let nothing = unit_of [] in
  show "a prefix declares a code id deleted, and the two units bind nothing"
    declares nothing nothing;
  show "the same prefix, and the unit before binds that code id" declares
    (unit_of [Bind_code (code_id, code)])
    nothing;
  show "the same prefix, and the unit after declares that code id deleted"
    declares nothing
    (unit_of [Bind_deleted_code code_id]);
  show "the same prefix, and the units bind and declare deleted another code id"
    declares
    (unit_of [Bind_code (other_code_id, code)])
    (unit_of [Bind_deleted_code other_code_id]);
  show "a prefix declares a code id deleted twice"
    (prefix [[Bind_deleted_code code_id]; [Bind_deleted_code code_id]])
    nothing nothing;
  show "a prefix binds a code id and declares it deleted"
    (prefix [[Bind_deleted_code code_id; Bind_code (code_id, code)]])
    nothing nothing

(* The steps. *)

let unit_that_returns : Flsemantics.converted =
  { body =
      Apply_cont
        { ac_continuation = return_continuation;
          ac_args = [Const (Tagged_immediate (Flsemantics.z_of_int 7))];
          ac_trap_action = None
        };
    return_continuation;
    exn_continuation
  }

let unit_that_binds : Flsemantics.converted =
  let body = unit_that_returns.body in
  { unit_that_returns with
    body = Let_static ([Bind_code (other_code_id, code)], body)
  }

(* A unit of [groups] groups that bind nothing, which then returns: a run of
   that many steps. *)
let unit_of_steps groups : Flsemantics.converted =
  let rec nest groups (body : Model.expr) : Model.expr =
    if groups = 0 then body else nest (groups - 1) (Model.Let_static ([], body))
  in
  { unit_that_returns with body = nest groups unit_that_returns.body }

let show_run what (unit : Flsemantics.converted) =
  let outcome, steps =
    Flsemantics.run ~statics:(Flsemantics.Imports.imports_statics []) unit
  in
  Printf.printf "%s (%d steps): %s\n" what steps
    (Format.asprintf "%a" Flsemantics.print_outcome outcome)

let () =
  let declares = prefix [[Model.Bind_deleted_code code_id]] in
  let nothing = unit_of [] in
  show_run "a unit that returns, with no prefix" unit_that_returns;
  show_run "the same unit after a prefix that declares a code id deleted"
    (Flsemantics.with_prefix declares unit_that_returns);
  show_run "a unit that binds code and returns, with no prefix" unit_that_binds;
  show_run "the same unit after the same prefix"
    (Flsemantics.with_prefix declares unit_that_binds);
  show_run "a unit that is undefined, with no prefix" nothing;
  show_run "the same unit after the same prefix"
    (Flsemantics.with_prefix declares nothing);
  show_run "the first unit after a prefix that binds and declares nothing"
    (Flsemantics.with_prefix (prefix []) unit_that_returns);
  let last = unit_of_steps (Flsemantics.fuel - 1) in
  show_run "a unit that returns at the last step allowed, with no prefix" last;
  show_run "the same unit after the same prefix"
    (Flsemantics.with_prefix declares last)
