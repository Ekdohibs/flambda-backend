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

(* Inlining, in the body of code, the direct calls of the other code of
   its group: a normalisation that refines.

   This file is to be reviewed: [inline_sibling_calls] and
   [inline_in_group], which are extracted and run by the validation of
   middle_end/flambda2/tests/semantics/ (FlambdaValidateSpec.v), and
   the statements.

   FlambdaNormaliseSpec.inline_calls inlines a direct call of code that
   it knows. Down a term it knows the code that the groups of static
   bindings around it have bound; in the body of code, that is the code
   of the groups around the group that binds it, and not the code of its
   own group. A round of the validation ends by making one group of
   the groups of static bindings that follow one another
   (FlambdaNormaliseSpec.merge_statics), as Simplify does, so after a
   round much of the code of a unit is in one group. So a call in the
   body of code that becomes direct during one round, as that of a
   local function given to a function that is inlined does, finds its
   code in its own group from the next round on, and is not inlined,
   where Simplify inlines it.

   [inline_in_group] goes through the bindings of a group in order. The
   body of each piece of code is rewritten by
   FlambdaNormaliseSpec.inline_calls, with the other code of the group
   known: that of the bindings before it as they have just been
   rewritten, and that of the bindings after it as they are. The
   binding itself is not among what is known. So in a group whose code
   ids are distinct, as those of a well-formed term are, a call of code
   to itself is left. (If another binding of the group has the same
   code id, the code of that one is known under the id. A group that
   binds a code id twice has no meaning, and the statements are of
   every term.) [inline_sibling_calls] does this at every group of a
   term that is not in the body of code or in the handler of a
   recursive continuation.

   Why one binding at a time. When the body of code runs, the static
   environment binds the code of its group as the group now has it.
   What is inlined must be that code. With one binding rewritten and
   the others as they are, the others are the same on the two sides,
   and they are what the rewritten body has inlined.

   The code whose calls are left ([kept]) is an argument, as for
   FlambdaNormaliseSpec.inline_calls, and the statements hold whatever
   it is. The validation gives the code that calls itself through
   other code, so that a round does not unroll a recursion:
   that is the policy of the test tool, and no statement supposes it.

   The proofs are in FlambdaInlineSiblingsProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaSemanticsSpec
  FlambdaNominalSpec FlambdaNormaliseSpec.
Import ListNotations.

(* The bindings of a group, those of [done] as they are and each of
   [todo] with the calls of the other code of the group inlined in the
   body of its code: the code of [done], and of the bindings of [todo]
   that come after it. *)
Fixpoint inline_in_group (kept : list code_id)
    (done todo : list static_binding) : list static_binding :=
  match todo with
  | [] => done
  | binding :: todo =>
    let others := codes_of (done ++ todo) [] in
    inline_in_group kept
      (done ++ [inline_calls_in_binding kept others binding]) todo
  end.

(* The same at every group of an expression, but for those in the body
   of code and in the handlers of recursive continuations. *)
Fixpoint inline_sibling_calls (kept : list code_id) (e : expr) : expr :=
  match e with
  | Let x kind d body => Let x kind d (inline_sibling_calls kept body)
  | Let_static bindings body =>
    Let_static (inline_in_group kept [] bindings)
      (inline_sibling_calls kept body)
  | Let_cont_non_recursive k (Handler params metadata handler) body =>
    Let_cont_non_recursive k
      (Handler params metadata (inline_sibling_calls kept handler))
      (inline_sibling_calls kept body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive handlers (inline_sibling_calls kept body)
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end.

(* The result refines the term, under all environments. *)
Definition inline_sibling_calls_refines_statement : Prop :=
  ∀ (kept : list code_id) (e : expr),
    expr_refines e (inline_sibling_calls kept e).

(* The rewrite commutes with renaming. Code ids are not renamed. *)
Definition inline_sibling_calls_equivariant_statement : Prop :=
  ∀ (kept : list code_id) (r : renaming) (e : expr),
    inline_sibling_calls kept (rename_expr r e)
    = rename_expr r (inline_sibling_calls kept e).
