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
   its group whose callee is a symbol that the group binds to a closure
   of the code called: a normalisation that is an equivalence.

   This file is to be reviewed: [inline_known_sibling_calls] and
   [inline_known_in_group], which are extracted and run by the
   validation of middle_end/flambda2/tests/semantics/ on the term after
   Simplify (FlambdaValidateSpec.normalise_after_round), and the
   statements.

   What it is for. A round of the term before inlines a call in the
   body of code by two passes: FlambdaNormaliseSpec.inline_calls, which
   knows the code of the groups of static bindings around the group
   that binds the code, and
   FlambdaInlineSiblingsSpec.inline_sibling_calls, which knows the
   other code of the group itself. A round of the term after had the
   first of the two only, as an equivalence
   (FlambdaNormaliseSpec.inline_known_calls). Where a group binds two
   functions of which one calls the other, and the one called is not
   recursive (the functions of a [let rec ... and ...] of which one
   calls no function of the group), Simplify keeps the call: while it
   simplifies the code of a group, the code of the group is not there
   yet for it to inline. The term before then had the call inlined in
   the body of the caller, the term after kept it, and no rewrite
   brought the two together.

   [inline_known_in_group] is
   FlambdaInlineSiblingsSpec.inline_in_group with [inline_known_calls]
   in the place of [inline_calls]. It goes through the bindings of a
   group in order. The body of each piece of code is rewritten by
   [inline_known_calls] with what the other bindings of the group make
   known, those before it as they have just been rewritten and those
   after it as they are: their code (FlambdaNormaliseSpec.codes_of),
   and of each symbol that they bind whether it is a closure, and of
   which code ([symbols_of]). So a call is inlined where
   [inline_known_calls] would inline it had the other bindings of the
   group been bound around the code: its callee is a symbol that the
   group binds to a closure of the code called, that code is the code
   of another binding of the group and is not among the code whose
   calls are left ([kept]), and the call passes the tests of
   FlambdaNormaliseSpec.inline_known_apply ([inlinable_b],
   [call_agrees_b]). A binding of code binds no symbol, so the symbols
   known are those of the whole group. The code of the binding itself
   is not among what is known, so in a group whose code ids are
   distinct, as those of a well-formed term are, a call of code to
   itself is left. (If another binding of the group has the same code
   id, the code of that one is known under the id. A group that binds
   a code id twice has no meaning, and the statements are of every
   term.) [inline_known_sibling_calls] does this at every group of a
   term that is not in the body of code or in the handler of a
   recursive continuation, as [inline_sibling_calls] does.

   Inside the body of code of a group the rewrite is
   [inline_known_calls] itself, given what the other bindings of the
   group make known. It goes into the handlers of the continuations
   of the body, those of recursive ones too, and into the groups that
   the body binds: the code of such an inner group is rewritten with
   what is known around that group, and what follows the group with
   its code and its symbols known as well. The code of an inner group
   does not know the other code of that group: [inline_known_in_group]
   is applied to no group that is in the body of code.

   Why one binding at a time. When the body of code runs, the static
   environment binds the code and the symbols of its group as the
   group now has them. What is inlined must be that code. With one
   binding rewritten and the others as they are, the others are the
   same on the two sides, and they are what the rewritten body has
   inlined.

   Why it is an equivalence, where [inline_sibling_calls] refines.
   [inline_calls] inlines a direct call whatever its callee is. The
   call is undefined where the callee is not a closure of the code
   called, or the arities of the call are not those of the code
   (FlambdaSemanticsSpec.call_function), and the copy of the body is
   not. [inline_known_calls] inlines only where neither can be: the
   callee is a symbol known to be bound to a closure of the code
   called, and the arities agree. Here that is known of the static
   environment in which the body of the code runs, which is one that
   the group has extended: it binds each symbol of the group to what
   its binding says, and the code of each other binding to that code.
   The rewrites of the term after are equivalences
   (FlambdaValidateSpec.v), and this is one of them.

   The code whose calls are left ([kept]) is an argument, as for
   [inline_known_calls], and the statements hold whatever it is. The
   validation gives the list that it gives [inline_known_calls] in the
   same round: the code that calls itself through other code, so that
   a round does not unroll a recursion. That is the policy of the test
   tool, and no statement supposes it.

   Renaming. The pass compares code ids and symbols, which the
   renamings of FlambdaNominalSpec.v leave as they are, and makes no
   name: the copy of a body has the names that the body binds. It
   commutes with renaming
   ([inline_known_sibling_calls_renamed_statement]).

   The proofs are in FlambdaKnownSiblingsProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaSemanticsSpec
  FlambdaNominalSpec FlambdaNormaliseSpec.
Import ListNotations.

(* The bindings of a group, those of [done] as they are and each of
   [todo] with the calls of the other code of the group inlined in the
   body of its code, where the callee is a symbol that the group binds
   to a closure of the code called. What is known is the code and the
   symbols of [done] and of the bindings of [todo] that come after
   it. *)
Fixpoint inline_known_in_group (kept : list code_id)
    (done todo : list static_binding) : list static_binding :=
  match todo with
  | [] => done
  | binding :: todo =>
    let others := done ++ todo in
    inline_known_in_group kept
      (done
       ++ [inline_known_calls_in_binding kept (codes_of others [])
             (symbols_of others []) binding])
      todo
  end.

(* The same at every group of an expression, but for those in the body
   of code and in the handlers of recursive continuations. *)
Fixpoint inline_known_sibling_calls (kept : list code_id) (e : expr)
    : expr :=
  match e with
  | Let x kind d body =>
    Let x kind d (inline_known_sibling_calls kept body)
  | Let_static bindings body =>
    Let_static (inline_known_in_group kept [] bindings)
      (inline_known_sibling_calls kept body)
  | Let_cont_non_recursive k (Handler params metadata handler) body =>
    Let_cont_non_recursive k
      (Handler params metadata (inline_known_sibling_calls kept handler))
      (inline_known_sibling_calls kept body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive handlers (inline_known_sibling_calls kept body)
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end.

(* The result is equivalent to the term, under all environments. *)
Definition inline_known_sibling_calls_equivalent_statement : Prop :=
  ∀ (kept : list code_id) (e : expr),
    expr_equivalent e (inline_known_sibling_calls kept e).

(* The rewrite commutes with renaming. Code ids are not renamed. *)
Definition inline_known_sibling_calls_renamed_statement : Prop :=
  ∀ (kept : list code_id) (r : renaming) (e : expr),
    inline_known_sibling_calls kept (rename_expr r e)
    = rename_expr r (inline_known_sibling_calls kept e).
