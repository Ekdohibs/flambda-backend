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

(* Not projecting again a value slot that has been projected: a
   normalisation that refines.

   This file is to be reviewed: [share_known_projections] and what it
   is made of, which are extracted and run by the validation of
   middle_end/flambda2/tests/semantics/; the action of a renaming on
   what the pass knows ([rename_projected]); and the statements.

   The body of a function reads a value of its closure by projecting a
   value slot. Where a function is inlined twice in one piece of code,
   the term before Simplify projects the slot twice from the same
   closure, and the term after projects it once: Simplify meets the
   type of the closure with the shape that names the variable of the
   projection ([Simplify_common.simplify_projection]), and a second
   projection of the slot finds that variable there. It is how
   Simplify has a second load of an immutable field stand for the
   first, too (FlambdaCseSpec.v, which shares those loads). The
   projections are not primitives in the model, so that file does not
   reach them.

   [share_known_projections] does the same with what can be read off
   the term. It goes through a term keeping, for a variable bound by a
   [Let] to the projection of a value slot ([is_projection]: either
   form of the projection, the one that names no function slot and
   the one that names the function slot it projects from), that
   projection ([projected]). A later [Let] of the same projection, in
   the same form, of the same slot from the same closure, becomes a
   [Let] of that variable ([share_projection]). Nothing else is
   rewritten, and the [Let] stays. As in FlambdaCseSpec.v, which this
   file follows: what is known of a variable, and what projects from
   it, is forgotten under a binder of it; a handler of a non-recursive
   continuation is gone into, with what is known where it is defined;
   the handlers of recursive ones and the bodies of code are left as
   they are, and the validation takes the pass into them as it does
   the other rewrites of a round.

   The closure projected from is any [Simple]: a variable, or the
   symbol of a static closure, whose slot may hold a variable that is
   not in scope where the projection is, so that the projection cannot
   be replaced by what the slot holds.

   Why it is sound. A projection has the value that the slot has in
   the closure (FlambdaSemanticsSpec.eval_named), and closures are not
   written to: the same projection from the same closure, while
   neither the variable bound to the first nor the closure is bound
   again, has the same value. A static environment that binds more
   gives a symbol that was bound the closure it had.

   The result refines the term
   ([share_known_projections_refines_statement]). It is not claimed to
   be equivalent to it, though it most likely is; a refinement is what
   the term before Simplify needs.

   The pass commutes with the renamings of FlambdaNominalSpec.v, what
   it knows being renamed with the term
   ([share_projections_knowing_equivariant_statement]): it compares
   variables for equality and makes no name.

   The proofs are in FlambdaShareProjectionsProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaNormaliseSpec FlambdaNominalSpec
  FlambdaCseSpec.
Import ListNotations.

(* ======================================================================== *)
(* What is known                                                            *)
(* ======================================================================== *)

(* The projections of a value slot. *)
Definition is_projection (d : named) : bool :=
  match d with
  | Project_value_slot _ _ | Project_value_slot_from _ _ _ => true
  | _ => false
  end.

(* The closure that a projection is from, as a list of one; nothing for
   what is not a projection. *)
Definition projected_from (d : named) : list simple :=
  match d with
  | Project_value_slot _ closure | Project_value_slot_from _ _ closure =>
    [closure]
  | _ => []
  end.

(* The variables known to hold a projection, each with the
   projection. *)
Definition projected : Type := list (variable * named).

(* What is known that does not mention [x]. *)
Definition forget_projected (x : variable) (known : projected) : projected :=
  filter
    (fun entry =>
       negb (Pos.eqb x (fst entry))
       && negb (var_in_simples x (projected_from (snd entry))))
    known.

(* The same of parameters, bound one after the other. *)
Fixpoint forget_projected_vars (xs : list variable) (known : projected)
    : projected :=
  match xs with
  | [] => known
  | x :: xs => forget_projected_vars xs (forget_projected x known)
  end.

(* What is known in the body of a [Let] of [x] to [d]: nothing that
   mentions the old [x], and that the new one holds [d], if that is a
   projection from a closure that is not the old [x]. *)
Definition projected_under_let (known : projected) (x : variable)
    (d : named) : projected :=
  let known := forget_projected x known in
  if is_projection d && negb (var_in_simples x (projected_from d))
  then (x, d) :: known
  else known.

(* ======================================================================== *)
(* Sharing                                                                  *)
(* ======================================================================== *)

(* Two projections of the same value slot from the same closure, at
   the same function slot where they name one. *)
Definition same_projection (d d' : named) : bool :=
  match d, d' with
  | Project_value_slot slot closure, Project_value_slot slot' closure' =>
    Pos.eqb slot slot' && same_simples [closure] [closure']
  | Project_value_slot_from project_from slot closure,
    Project_value_slot_from project_from' slot' closure' =>
    Pos.eqb project_from project_from'
    && Pos.eqb slot slot'
    && same_simples [closure] [closure']
  | _, _ => false
  end.

(* The first variable known to hold the projection [d]. *)
Fixpoint projected_by (known : projected) (d : named) : option variable :=
  match known with
  | [] => None
  | (x, d') :: known =>
    if same_projection d d' then Some x else projected_by known d
  end.

Definition share_projection (known : projected) (d : named) : named :=
  match projected_by known d with
  | Some x => Simple (Var x)
  | None => d
  end.

Fixpoint share_projections_knowing (known : projected) (e : expr) : expr :=
  match e with
  | Let x kind d body =>
    Let x kind (share_projection known d)
      (share_projections_knowing (projected_under_let known x d) body)
  | Let_static bindings body =>
    Let_static bindings (share_projections_knowing known body)
  | Let_cont_non_recursive k h body =>
    Let_cont_non_recursive k (share_projections_in_handler known h)
      (share_projections_knowing known body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive handlers (share_projections_knowing known body)
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end
with share_projections_in_handler (known : projected)
    (h : continuation_handler) : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata
      (share_projections_knowing (forget_projected_vars params known)
         handler)
  end.

Definition share_known_projections (e : expr) : expr :=
  share_projections_knowing [] e.

Definition share_known_projections_refines_statement : Prop :=
  ∀ e : expr, expr_refines e (share_known_projections e).

(* ======================================================================== *)
(* The pass commutes with renaming                                          *)
(* ======================================================================== *)

(* What is known holds variables: the one bound to each projection,
   and the closure projected from where it is one. A renaming acts on
   both. A slot is not a name of the term. *)
Definition rename_projected (r : renaming) (known : projected) : projected :=
  map
    (fun entry => (rename_variable r (fst entry), rename_named r (snd entry)))
    known.

(* What is known is renamed with the term. *)
Definition share_projections_knowing_equivariant_statement : Prop :=
  ∀ (r : renaming) (known : projected) (e : expr),
    share_projections_knowing (rename_projected r known) (rename_expr r e)
    = rename_expr r (share_projections_knowing known e).

(* The pass as the validation runs it, with nothing known. *)
Definition share_known_projections_equivariant_statement : Prop :=
  ∀ (r : renaming) (e : expr),
    share_known_projections (rename_expr r e)
    = rename_expr r (share_known_projections e).
