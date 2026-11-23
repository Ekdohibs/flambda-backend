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

(* Proofs of the statement of FlambdaCseEquivalentSpec.v. Nothing here
   needs review.

   FlambdaCseProofs.v proves that the result of the pass refines the
   term, by recursion on the term, with what the pass knows holding
   where the term runs. The converse is by the same recursion, with
   the lemmas of that file that keep what is known: where what is
   known holds, a defining expression and what the pass makes of it
   run the same on the machine: to the same value, heap and read
   record, or neither runs (FlambdaCseProofs.share_named_same). *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaKindModel FlambdaValueSpec FlambdaPrimitiveModel
  FlambdaPrimitiveSpec FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaContextProofs FlambdaContextMemoryProofs FlambdaWellFormedSpec
  FlambdaWellFormedProofs
  FlambdaNormaliseSpec FlambdaNormaliseProofs FlambdaCseSpec
  FlambdaCseProofs FlambdaCseEquivalentSpec.
Import ListNotations.

(* The term refines the result: FlambdaCseProofs.v has the converse. *)
Fixpoint share_primitives_knowing_refined (e : expr) :
  ∀ (known : computed) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record),
    all_computed_or_made known vars statics hp rd ->
    refines (State (share_primitives_knowing known e) vars conts statics hp rd)
      (State e vars conts statics hp rd)
with share_primitives_in_handler_refined (h : continuation_handler) :
  ∀ (known : computed) (k : continuation) (body : expr) (vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap) (rd : read_record),
    all_computed_or_made known vars statics hp rd ->
    refines
      (State
         (Let_cont_non_recursive k (share_primitives_in_handler known h)
            body)
         vars conts statics hp rd)
      (State (Let_cont_non_recursive k h body) vars conts statics hp rd).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap
         | ac | scrutinee arms | ];
      intros known vars conts statics hp rd Hhold; simpl.
    + (* Let *)
      pose proof (share_named_same known vars statics hp rd d Hhold)
        as Hsame.
      apply let_refines_step_exec; [ symmetry; exact Hsame | ].
      intros v hp' rd' Hd _. rewrite Hsame in Hd.
      apply share_primitives_knowing_refined.
      exact
        (all_computed_or_made_under_let known x d vars statics hp hp' rd rd'
           v Hhold Hd).
    + (* Let_static *)
      apply let_static_refines_step.
      destruct (eval_static_bindings vars bindings statics)
        as [extended | ] eqn:Hbindings; [ | exact I ].
      apply share_primitives_knowing_refined.
      exact
        (all_computed_or_made_extend known vars statics extended hp rd Hhold
           (eval_static_bindings_extend vars bindings _ _ Hbindings)).
    + (* Let_cont_non_recursive *)
      apply (refines_trans _
        (State
           (Let_cont_non_recursive k (share_primitives_in_handler known h)
              body)
           vars conts statics hp rd)).
      * apply let_cont_non_recursive_refines_step,
          share_primitives_knowing_refined, Hhold.
      * apply share_primitives_in_handler_refined, Hhold.
    + (* Let_cont_recursive: the body; then the handlers, all at once,
         each under what is known less what mentions its parameters. *)
      apply (refines_trans _
        (State
           (Let_cont_recursive
              (map (fun '(k, h) => (k, share_primitives_in_handler known h))
                 handlers)
              body)
           vars conts statics hp rd)).
      * apply let_cont_recursive_refines_step; [ reflexivity | intros _ ].
        apply share_primitives_knowing_refined, Hhold.
      * apply
          (let_cont_recursive_handlers_refine_where_memory
             (all_computed_or_made known vars)
             (all_computed_or_made_extend known vars)
             (all_computed_or_made_evolves known vars));
          [ exact Hhold
          | apply share_primitives_in_handlers_bind_statics
          | | ];
          (induction handlers as [ | [k [params metadata handler]] handlers IH];
             cbn [map share_primitives_in_handler];
             [ apply Forall2_nil | ];
           apply Forall2_cons; [ | exact IH ];
           split; [ reflexivity | ]; cbn [snd]; apply Handler_related;
           intros vs handler_vars conts' statics' hp' rd' Hhold' _ Hbind;
           apply share_primitives_knowing_refined;
           exact
             (all_computed_or_made_bind_params statics' hp' rd' params known
                vs vars handler_vars Hhold' Hbind)).
    + apply refines_refl.
    + apply refines_refl.
    + apply refines_refl.
    + apply refines_refl.
  - destruct h as [params metadata handler].
    intros known k body vars conts statics hp rd Hhold. simpl.
    apply
      (let_cont_handler_refines_where_memory
         (all_computed_or_made known vars)
         (all_computed_or_made_extend known vars)
         (all_computed_or_made_evolves known vars));
      [ exact Hhold | ].
    intros vs handler_vars conts' statics' hp' rd' Hhold' _ Hbind.
    apply share_primitives_knowing_refined.
    exact
      (all_computed_or_made_bind_params statics' hp' rd' params known vs
         vars handler_vars Hhold' Hbind).
Qed.

Theorem share_known_primitives_equivalent :
  share_known_primitives_equivalent_statement.
Proof.
  intros e vars conts statics hp rd. apply refines_antisymmetric.
  - apply share_primitives_knowing_refines. intros entry [].
  - apply share_primitives_knowing_refined. intros entry [].
Qed.
