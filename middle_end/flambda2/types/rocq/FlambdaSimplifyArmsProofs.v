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


(* Proofs of the statements of FlambdaSimplifyArmsSpec.v. Nothing here
   needs review. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec FlambdaPrimitiveModel
  FlambdaPrimitiveSpec FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaNominalSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaTypingEnv FlambdaTypingEnvSpec
  FlambdaTypingEnvExamplesSpec FlambdaTypingEnvRefinementSpec
  FlambdaTypingEnvRefinementProofs FlambdaEnvRenameSpec
  FlambdaSimplifyPrimitivesSpec FlambdaSimplifyPassSpec
  FlambdaSimplifyPassProofs FlambdaSimplifyPassRenamingSpec
  FlambdaSimplifyPassRenamingProofs FlambdaSimplifyUsesSpec
  FlambdaSimplifyUsesProofs FlambdaSimplifyArmsSpec.
Import ListNotations.

Local Open Scope Z_scope.

(* ======================================================================== *)
(* One arm                                                                  *)
(* ======================================================================== *)

Theorem simplify_arm_args_equivalent : simplify_arm_args_equivalent_statement.
Proof.
  intros tenv s [i ac] vars conts statics hp rd Hsat Heval.
  cbn [fst snd simplify_arm_args] in *.
  exact
    (simplify_apply_cont_equivalent (arm_env tenv s i) ac vars conts statics
       hp rd_ (arm_env_satisfied tenv s i statics vars Hsat Heval)).
Qed.

(* ======================================================================== *)
(* Arms replaced by arms that refine them                                   *)
(* ======================================================================== *)

(* A naked immediate read off a value is that value. *)
Lemma naked_immediate_read (v : value) (i : Z) :
  naked_immediate_of_value v = Some i -> v = Value_const (Naked_immediate i).
Proof.
  destruct v; try discriminate.
  match goal with c : const |- _ => destruct c; try discriminate end.
  intros [= ->]. reflexivity.
Qed.

(* The arm of a number among arms that correspond one to one. *)
Lemma find_arm_corresponding (R : Z -> apply_cont -> apply_cont -> Prop)
    (i : Z) :
  ∀ arms arms' : list (Z * apply_cont),
    Forall2
      (fun arm arm' => fst arm' = fst arm ∧ R (fst arm) (snd arm) (snd arm'))
      arms arms' ->
    match find_arm i arms with
    | Some ac => ∃ ac' : apply_cont, find_arm i arms' = Some ac' ∧ R i ac ac'
    | None => find_arm i arms' = None
    end.
Proof.
  intros arms arms' Hall.
  induction Hall as [ | [j ac] [j' ac'] arms arms' [Hfst Hsnd] _ IH]; simpl.
  - reflexivity.
  - simpl in Hfst, Hsnd. subst j'.
    destruct (Z.eqb_spec i j) as [-> | _]; [ | exact IH ].
    exists ac'. split; [ reflexivity | exact Hsnd ].
Qed.

Theorem switch_arms_refined_under : switch_arms_refined_under_statement.
Proof.
  intros tenv s arms arms' Hall vars conts statics hp rd Hsat.
  destruct (step_switch_cases s arms vars conts statics hp rd)
    as [Hundefined | [v [i [ac [Hv [Hi [Hfind Hstep]]]]]]].
  { apply step_undefined_refines, Hundefined. }
  pose proof (find_arm_corresponding
                (fun i ac ac' =>
                   arm_refines_where_taken_under tenv s i ac ac')
                i arms arms' Hall) as Hfind'.
  rewrite Hfind in Hfind'. destruct Hfind' as [ac' [Hfind' Hrefines]].
  apply
    (refines_step _ _ _ (State (Apply_cont ac') vars conts statics hp rd)
       Hstep).
  - rewrite step_switch, Hv, Hi, Hfind'. reflexivity.
  - apply Hrefines; [ exact Hsat | ].
    rewrite Hv, (naked_immediate_read v i Hi). reflexivity.
Qed.

(* ======================================================================== *)
(* The function                                                             *)
(* ======================================================================== *)

(* A [Switch] with each of its arms rewritten. *)
Lemma switch_rewritten_arms_refines (tenv : Typing_env.t) (s : simple)
    (arms : list (Z * apply_cont)) :
  expr_refines_under tenv (Switch s arms)
    (Switch s (map (simplify_arm_args tenv s) arms)).
Proof.
  apply switch_arms_refined_under.
  induction arms as [ | arm arms IH]; cbn [map]; constructor;
    [ | exact IH ].
  split; [ reflexivity | ].
  intros vars conts statics hp rd Hsat Heval.
  apply equivalent_refines.
  exact (simplify_arm_args_equivalent tenv s arm vars conts statics hp rd Hsat
           Heval).
Qed.

(* What [simplify_switch] answers: the [Switch] as it was, or an
   [Apply_cont]. *)
Lemma simplify_switch_cases (tenv : Typing_env.t) (s : simple)
    (arms : list (Z * apply_cont)) :
  simplify_switch tenv s arms = Switch s arms
  ∨ ∃ ac : apply_cont, simplify_switch tenv s arms = Apply_cont ac.
Proof.
  unfold simplify_switch.
  destruct (simplify_simple tenv s) as [[ty [x | sym | c]] | ];
    try (left; reflexivity).
  destruct c; try (left; reflexivity).
  destruct (find_arm _ arms) as [ac | ]; [ | left; reflexivity ].
  right. eexists. reflexivity.
Qed.

Theorem simplify_switch_and_arms_refines :
  simplify_switch_and_arms_refines_statement.
Proof.
  intros tenv s arms. unfold simplify_switch_and_arms.
  pose proof (simplify_switch_refines tenv s arms) as Hswitch.
  destruct (simplify_switch_cases tenv s arms) as [Hsame | [ac Hone]].
  - rewrite Hsame. apply switch_rewritten_arms_refines.
  - rewrite Hone in *. exact Hswitch.
Qed.

(* ======================================================================== *)
(* Renaming                                                                 *)
(* ======================================================================== *)

Theorem simplify_arm_args_equivariant : simplify_arm_args_equivariant_statement.
Proof.
  intros r tenv s [i ac]. unfold simplify_arm_args. cbn [fst snd].
  rewrite arm_env_equivariant, simplify_apply_cont_equivariant. reflexivity.
Qed.

Theorem simplify_switch_and_arms_equivariant :
  simplify_switch_and_arms_equivariant_statement.
Proof.
  intros r tenv s arms. unfold simplify_switch_and_arms.
  rewrite simplify_switch_equivariant.
  destruct (simplify_switch_cases tenv s arms) as [Hsame | [ac Hone]].
  - rewrite Hsame. cbn [rename_expr]. f_equal. rewrite !map_map.
    apply map_ext. intros arm.
    rewrite (simplify_arm_args_equivariant r tenv s arm). reflexivity.
  - rewrite Hone. reflexivity.
Qed.

(* ======================================================================== *)
(* Examples                                                                 *)
(* ======================================================================== *)

Theorem example_scrutinee_under_arms : example_scrutinee_under_arms_statement.
Proof. intros k. vm_compute. reflexivity. Qed.

Theorem example_tagging_under_arms : example_tagging_under_arms_statement.
Proof. intros k. vm_compute. reflexivity. Qed.
