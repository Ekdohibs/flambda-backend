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


(* The unboxing of variants commutes with renaming, and keeps scope.

   This file is to be reviewed: [rename_read] and the statements.

   FlambdaUnboxVariantsSpec.unbox_variant_params, like the pass of
   FlambdaUnboxParamsSpec.v, goes through a term carrying what is known
   of its variables, compares names by equality only, and binds no name
   that the term has not got: the variable that becomes a parameter was
   bound by the [Let] that goes, and what an application is given is a
   constant or something in scope. So it commutes with the renamings of
   FlambdaNominalSpec.v, in the forms of
   FlambdaUnboxParamsRenameSpec.v, which this file follows:
   - an expression with its applications extended by a function of
     their arguments ([extend_applications_by]) is renamed when what is
     known, the continuation and the expression are, if the function
     commutes with renaming;
   - an expression with the arms taken that known numbers lead to
     ([take_arms]) is renamed when the variables of which the numbers
     are known ([rename_immediates]) and the expression are;
   - the arguments that [extend_args_certain] gives are renamed when
     the tests, the parameters, the two handlers, what is known and
     the arguments are, whichever constant an untagging is given where
     there is nothing to read; the test that an application does not
     come to a read answers
     the same, as the removal of unused continuations and the
     comparison commute with renaming;
   - the first read of a parameter by a primitive that is wanted
     ([first_read]) is the first read of the renamed expression,
     renamed ([rename_read]): the variable and the expression without
     the [Let] are renamed, and the kind, the primitive and the
     position are as they were;
   - the pass given what is known ([unbox_variants_knowing]) gives the
     renamed result when what is known is renamed with the term;
   - the pass with nothing known is a rewrite that takes a term and
     nothing else ([unbox_variant_params_equivariant_statement], which
     is [rewrite_equivariant unbox_variant_params]).

   Each of these statements is an equation and holds of every term.

   The pass also keeps scope
   (FlambdaValidatorEquivarianceSpec.keeps_scope): on a term whose code
   is closed, the code of the result is closed and every variable or
   continuation that is free in the result is free in the term. The
   variable that becomes a parameter is bound in all of the handler,
   where it was bound below the [Let] only; the test that it is not
   free in the handler is what makes the pass refine the term, and is
   not needed for this.

   With these two and [unbox_variant_params_refines] the pass is in a
   round of the validation (FlambdaValidateSpec.normalise_expr_params):
   the validation is still sound, answers the same under a renaming
   and keeps scope with it.

   The unboxing of the reads of a loop
   (FlambdaUnboxLoopsSpec.unbox_loops_knowing), which the same round
   applies next, is made of the same things: the first read of a
   parameter, the applications extended, and what is known. It makes
   no name either, and given what is known, renamed with the term, it
   gives the term that it gave, renamed
   ([unbox_loops_knowing_equivariant_statement]). That it keeps scope
   is stated with what is known of symbols alone, in
   FlambdaUnboxInScopeRenameSpec.v.

   The unboxing of boxed numbers that continuations pass to one
   another (FlambdaUnboxTogetherSpec.unbox_together_knowing) takes
   what is known too, for the arguments that the term shows to be
   boxes. It compares names by equality and makes none: the positions
   that it unboxes are computed from the term, and those of the term
   renamed are those positions renamed. Given what is known, renamed
   with the term, it gives the term that it gave, renamed
   ([unbox_together_knowing_equivariant_statement]). That it keeps
   scope is stated in FlambdaUnboxInScopeRenameSpec.v too.

   The proofs are in FlambdaUnboxVariantsRenameProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaPrimitiveModel FlambdaNominalSpec FlambdaBlockLoadSpec
  FlambdaUnboxSpec FlambdaValidatorEquivarianceSpec FlambdaUnboxParamsSpec
  FlambdaUnboxParamsRenameSpec FlambdaUnboxVariantsSpec
  FlambdaUnboxLoopsSpec FlambdaUnboxTogetherSpec.
Import ListNotations.

(* A read that was found, renamed. *)
Definition rename_read (r : renaming) (found : read) : read :=
  Read (rename_variable r (read_variable found)) (read_kind found)
    (read_primitive found) (read_position found)
    (rename_expr r (read_without found)).

Definition extend_applications_by_equivariant_statement : Prop :=
  ∀ (r : renaming)
    (extend extend' : facts -> list simple -> option (list simple)),
    (∀ (known : facts) (args : list simple),
      extend' (rename_facts r known) (map (rename_simple r) args)
      = option_map (map (rename_simple r)) (extend known args)) ->
    ∀ (known : facts) (k : continuation) (e : expr),
      extend_applications_by extend' (rename_facts r known)
        (rename_continuation r k) (rename_expr r e)
      = option_map (rename_expr r) (extend_applications_by extend known k e).

(* The numbers that are known of variables, renamed. *)
Definition rename_immediates (r : renaming) (consts : list (variable * Z))
    : list (variable * Z) :=
  map (fun entry => (rename_variable r (fst entry), snd entry)) consts.

Definition take_arms_equivariant_statement : Prop :=
  ∀ (r : renaming) (consts : list (variable * Z)) (e : expr),
    take_arms (rename_immediates r consts) (rename_expr r e)
    = rename_expr r (take_arms consts e).

Definition extend_args_certain_equivariant_statement : Prop :=
  ∀ (r : renaming) (variant : bool) (tests params : list variable)
    (handler without : expr) (known : facts) (i : nat) (p : unary_primitive)
    (kind : flambda_kind) (args : list simple),
    extend_args_certain variant (map (rename_variable r) tests)
      (map (rename_variable r) params) (rename_expr r handler)
      (rename_expr r without) (rename_facts r known) i p kind
      (map (rename_simple r) args)
    = option_map (map (rename_simple r))
        (extend_args_certain variant tests params handler without known i p
           kind args).

Definition first_read_equivariant_statement : Prop :=
  ∀ (r : renaming) (wanted : unary_primitive -> bool)
    (params bound : list variable) (e : expr),
    first_read wanted (map (rename_variable r) params)
      (map (rename_variable r) bound) (rename_expr r e)
    = option_map (rename_read r) (first_read wanted params bound e).

Definition unbox_variants_knowing_equivariant_statement : Prop :=
  ∀ (r : renaming) (known : facts) (e : expr),
    unbox_variants_knowing (rename_facts r known) (rename_expr r e)
    = rename_expr r (unbox_variants_knowing known e).

Definition unbox_variant_params_equivariant_statement : Prop :=
  rewrite_equivariant unbox_variant_params.

Definition unbox_variant_params_keeps_scope_statement : Prop :=
  keeps_scope unbox_variant_params.

Definition unbox_loops_knowing_equivariant_statement : Prop :=
  ∀ (r : renaming) (known : facts) (e : expr),
    unbox_loops_knowing (rename_facts r known) (rename_expr r e)
    = rename_expr r (unbox_loops_knowing known e).

Definition unbox_together_knowing_equivariant_statement : Prop :=
  ∀ (r : renaming) (known : facts) (e : expr),
    unbox_together_knowing (rename_facts r known) (rename_expr r e)
    = rename_expr r (unbox_together_knowing known e).
