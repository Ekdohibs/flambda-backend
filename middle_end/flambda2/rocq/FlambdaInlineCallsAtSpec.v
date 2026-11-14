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


(* Inlining the direct calls that stand at chosen places of a term: a
   normalisation that refines.

   This file is to be reviewed: [inline_calls_at_paths] and
   [inline_calls_at], which are extracted and run by the validation
   of middle_end/flambda2/tests/semantics/
   (FlambdaValidateInliningSpec.v gives the second the term before),
   and the statements.

   FlambdaNormaliseSpec.inline_calls is told the code whose calls it
   leaves, and inlines every other direct call of code that it knows:
   the choice is by code. Simplify chooses by call. It inlines one
   call of a function and leaves another call of the same function in
   the same body; it unrolls a recursive function at a call that is
   marked for it, and leaves the other calls of that function; it
   inlines a function everywhere but in the body of a stub. The
   validation normalises both terms, the term after as the term
   before. Of such a unit it can still be that neither choice
   matches: that the term before with every call of that code
   inlined does not come to what the term after comes to, nor the
   term before with every one left.

   Here the choice is by place. A place is a FlambdaLiftSpec.path:
   where a call is in the term, or in the body of the code that it
   stands in, counted as that file counts and as
   FlambdaLiftFromCodeSpec.lift_from_code walks. A call that stands
   at a chosen place is given to FlambdaNormaliseSpec.inline_apply
   with no code to leave. So it is rewritten as [inline_calls] would
   rewrite it there if told to leave nothing, with the code that
   [inline_calls] knows there: inlined, or split if it has more
   arguments than its code has parameters, or left if it passes
   neither test. A call at any other place is left as it is, and a
   chosen place where no call stands changes nothing. The places are
   a guess of the caller, which is not trusted: the statement holds
   whatever they are.

   As for [inline_calls], what is known in the body of code is the
   code of the groups of static bindings around the group that binds
   it, and not the code of its own group. So a call of code to
   itself, or to code bound with it, is not inlined here
   (FlambdaInlineSiblingsSpec.inline_sibling_calls is for the
   second).

   One application rewrites the calls of the term as it is given. A
   call that an inlining brings into the term is at a place of the
   term that results, and is for another application.

   Why this is not in FlambdaNormaliseSpec.v beside [inline_calls]: a
   place is a FlambdaLiftSpec.path, and FlambdaLiftSpec.v comes after
   that file in the order of the files.

   The proofs are in FlambdaInlineCallsAtProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaNominalSpec FlambdaNormaliseSpec
  FlambdaLiftSpec.
Import ListNotations.

(* The guess for the code of a term: for a code id, the places of the
   calls to rewrite in the body of its code, by their paths in the
   body. *)
Definition calls_in_code : Type := list (code_id * list path).

Definition calls_of (K : calls_in_code) (id : code_id) : list path :=
  match lookup id K with
  | Some sites => sites
  | None => []
  end.

(* The expression with the calls at the paths [sites] rewritten, and
   with the code that its groups of static bindings bind rewritten:
   in the body of the code that [K] names, the calls at the paths
   that [K] gives for it. [at_path] is the path of the expression, and
   [known] the code that the groups around it have bound, as for
   FlambdaNormaliseSpec.inline_calls. *)
Fixpoint inline_calls_at_paths (K : calls_in_code) (sites : list path)
    (known : list (code_id * code)) (at_path : path) (e : expr) : expr :=
  match e with
  | Let x kind d body =>
    Let x kind d (inline_calls_at_paths K sites known (xO at_path) body)
  | Let_static bindings body =>
    let bindings := map (binding_inline_calls_at_paths K known) bindings in
    Let_static bindings
      (inline_calls_at_paths K sites (codes_of bindings known) (xO at_path)
         body)
  | Let_cont_non_recursive k h body =>
    Let_cont_non_recursive k
      (handler_inline_calls_at_paths K sites known (xI at_path) h)
      (inline_calls_at_paths K sites known (xO at_path) body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive
      (rec_handlers_map (handler_inline_calls_at_paths K sites known)
         (xI at_path) handlers)
      (inline_calls_at_paths K sites known (xO at_path) body)
  | Apply ap => if among at_path sites then inline_apply [] known ap else e
  | Apply_cont _ | Switch _ _ | Invalid => e
  end
with handler_inline_calls_at_paths (K : calls_in_code) (sites : list path)
    (known : list (code_id * code)) (at_path : path)
    (h : continuation_handler) : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata
      (inline_calls_at_paths K sites known at_path handler)
  end
with binding_inline_calls_at_paths (K : calls_in_code)
    (known : list (code_id * code)) (binding : static_binding)
    : static_binding :=
  match binding with
  | Bind_code id (Code params my_closure ret exn metadata body) =>
    Bind_code id
      (Code params my_closure ret exn metadata
         (inline_calls_at_paths K (calls_of K id) known root body))
  | Bind_deleted_code _ | Bind_symbol _ _ => binding
  end.

(* The same of a whole term: [sites] are the places chosen outside
   code, and [K] those chosen in code. Nothing is known at its head. *)
Definition inline_calls_at (sites : list path) (K : calls_in_code)
    (e : expr) : expr :=
  inline_calls_at_paths K sites [] root e.

(* The result refines the term, under all environments. *)
Definition inline_calls_at_refines_statement : Prop :=
  ∀ (sites : list path) (K : calls_in_code) (e : expr),
    expr_refines e (inline_calls_at sites K e).

(* The rewrite commutes with renaming: a call is chosen by where it
   is. The renaming is of variables and continuations, in code too:
   code ids are not renamed. *)
Definition inline_calls_at_equivariant_statement : Prop :=
  ∀ (r : renaming) (sites : list path) (K : calls_in_code) (e : expr),
    inline_calls_at sites K (rename_expr r e)
    = rename_expr r (inline_calls_at sites K e).
