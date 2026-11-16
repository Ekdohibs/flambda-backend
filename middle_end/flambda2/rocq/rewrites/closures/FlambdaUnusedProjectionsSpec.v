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

(* Removing the projection of a function slot that nothing uses, from a
   closure that a [Let] around it has made: a normalisation that is an
   equivalence.

   This file is to be reviewed: [remove_unused_projections] and what it
   is made of, which are extracted and run by the validation of
   middle_end/flambda2/tests/semantics/ on the term after Simplify
   (FlambdaValidateSpec.v); the action of a renaming on what the pass
   knows ([rename_known_sets]); and the statement.

   The compiler's [Let] of a set of closures binds one variable to each
   closure of the set. The model's [Let] binds one variable, and such a
   [Let] is here a [Let] of the first closure and, for each other
   closure, a [Let] of its projection from the first
   (FlambdaModel.let_closures):

     let g = closure_of_set g_slot {g_slot: code_g; h_slot: code_h} ...
     let h = project_function_slot (g_slot -> h_slot) g

   Functions defined together (let rec g ... and h ...) are bound so.
   Where the term uses g and not h, the compiler's term still binds h,
   as a set is bound whole; so the model's term has a [Let] of a
   projection that nothing uses, after Simplify as before.

   In the term before Simplify the validation removes that [Let] by
   FlambdaNormaliseSpec.remove_unused_lets. That is a refinement
   whether or not the projection has a value. The term after may be
   rewritten by equivalences only, and the projection has no value on
   a closure that is not at g_slot of a set that has h_slot
   (FlambdaSemanticsSpec.project_function_slot_defined_statement). So
   FlambdaNormaliseSpec.remove_unused_defined_lets, which removes a
   [Let] whose defining expression has a value whatever the variables
   are bound to, leaves it, and the two terms differ by it. It has a
   value here because of what g is bound to.

   [remove_unused_projections] goes through a term keeping, for a
   variable bound by a [Let] to a closure made from a set, the slot of
   the closure and the functions of the set ([known_sets]). A [Let] of
   the projection of a function slot from such a variable, whose own
   variable the body does not use, is removed, if the closure is at the
   slot that the projection moves from and the set has that slot and
   the slot it moves to ([projection_defined]). The first of the two
   slots being in the set is what makes the closure
   (FlambdaSemanticsSpec.eval_named), so the [Let] around has already
   said it; it is tested again here so that this file can be read
   alone. The [Let] removed must bind its variable at the kind
   [Value], the kind of a closure, as those of
   FlambdaModel.let_closures do: a [Let] has no step unless the value
   of its defining expression is of its kind
   (FlambdaSemanticsSpec.step), so a [Let] of a projection at another
   kind is undefined, and removing it would not be an equivalence.
   Nothing else is rewritten.

   As in FlambdaCseSpec.v, which this file follows: what is known of a
   variable is forgotten under a binder of it; a handler of a
   non-recursive continuation is gone into, with what is known where
   it is defined; the handlers of recursive ones and the bodies of
   code are left as they are (FlambdaNormaliseSpec.rewrite_code takes
   a rewrite into the bodies of code).

   The projection of a function slot from a symbol is not removed: the
   model gives it a value only where the set names a symbol for the
   slot moved to, and no unit of the corpus has needed it.

   The result is equivalent to the term
   ([remove_unused_projections_equivalent_statement]).

   The pass commutes with the renamings of FlambdaNominalSpec.v, what it
   knows being renamed with the term ([rename_known_sets]): it compares
   variables for equality and makes no name. Function slots and code
   ids are not renamed. That statement is with those of the other
   rewrites that remove a [Let],
   FlambdaValidatorEquivarianceSpec
   .remove_projections_knowing_equivariant_statement, because its proof
   uses what is proved there of the test of a free variable.

   The proof is in FlambdaUnusedProjectionsProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaNormaliseSpec FlambdaNominalSpec.
Import ListNotations.

(* ======================================================================== *)
(* What is known                                                            *)
(* ======================================================================== *)

(* The variables known to be bound to a closure made from a set, each
   with the slot of the closure and the functions of the set. *)
Definition known_sets : Type :=
  list (variable * (function_slot * list (function_slot * code_id))).

(* What is known of the variables other than [x]. *)
Definition forget_set (x : variable) (known : known_sets) : known_sets :=
  filter (fun entry => negb (Pos.eqb x (fst entry))) known.

(* The same of parameters. *)
Fixpoint forget_sets (xs : list variable) (known : known_sets)
    : known_sets :=
  match xs with
  | [] => known
  | x :: xs => forget_sets xs (forget_set x known)
  end.

(* What is known in the body of a [Let] of [x] to [d]: nothing of the
   old [x], and that the new one is a closure of a set, if [d] makes
   one. *)
Definition sets_under_let (known : known_sets) (x : variable) (d : named)
    : known_sets :=
  let known := forget_set x known in
  match d with
  | Closure_of_set slot function_decls _ =>
    (x, (slot, function_decls)) :: known
  | _ => known
  end.

(* ======================================================================== *)
(* The projections that have a value                                        *)
(* ======================================================================== *)

(* The defining expression is the projection of a function slot from a
   variable known to be a closure at the slot moved from, of a set that
   has that slot and the slot moved to. *)
Definition projection_defined (known : known_sets) (d : named) : bool :=
  match d with
  | Project_function_slot move_from move_to (Var closure) =>
    match lookup closure known with
    | Some (slot, function_decls) =>
      Pos.eqb slot move_from
      && is_bound move_from function_decls
      && is_bound move_to function_decls
    | None => false
    end
  | _ => false
  end.

Fixpoint remove_projections_knowing (known : known_sets) (e : expr) : expr :=
  match e with
  | Let x kind d body =>
    let body' := remove_projections_knowing (sets_under_let known x d) body in
    if kind_eqb kind Value
       && projection_defined known d
       && negb (var_free_b x body')
    then body'
    else Let x kind d body'
  | Let_static bindings body =>
    Let_static bindings (remove_projections_knowing known body)
  | Let_cont_non_recursive k (Handler params metadata handler) body =>
    Let_cont_non_recursive k
      (Handler params metadata
         (remove_projections_knowing (forget_sets params known) handler))
      (remove_projections_knowing known body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive handlers (remove_projections_knowing known body)
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end.

Definition remove_unused_projections (e : expr) : expr :=
  remove_projections_knowing [] e.

Definition remove_unused_projections_equivalent_statement : Prop :=
  ∀ e : expr, expr_equivalent e (remove_unused_projections e).

(* ======================================================================== *)
(* Renaming what is known                                                   *)
(* ======================================================================== *)

(* What is known holds the variable bound to each closure. A renaming
   acts on it. The slots and the code ids are not renamed. *)
Definition rename_known_sets (r : renaming) (known : known_sets)
    : known_sets :=
  map (fun entry => (rename_variable r (fst entry), snd entry)) known.
