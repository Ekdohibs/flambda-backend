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


(* Proofs of the statements of FlambdaSimplifySwitchArmsSpec.v. Nothing
   here needs review. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaPrimitiveModel FlambdaPrimitiveSpec
  FlambdaPrimitiveProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaControlFlowSpec FlambdaControlFlowProofs FlambdaClosuresSpec
  FlambdaNominalSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeGrammarProofs FlambdaTypeMeaningSpec FlambdaTypeMeaningProofs
  FlambdaTypeWellFormedSpec FlambdaTypeWellFormedProofs FlambdaBindingTime
  FlambdaEnvAliases FlambdaEnvAliasesSpec FlambdaEnvAliasesProofs
  FlambdaTypingEnv FlambdaTypingEnvSpec FlambdaTypingEnvProofs
  FlambdaTypingEnvInvariantSpec FlambdaTypingEnvInvariantProofs
  FlambdaTypingEnvOperationsSpec FlambdaTypingEnvOperationsProofs
  FlambdaTypingEnvKindedSpec FlambdaTypingEnvKindedProofs
  FlambdaTypingEnvExamplesSpec FlambdaTypingEnvRefinementSpec
  FlambdaTypingEnvRefinementProofs FlambdaMeetGrammar FlambdaExpandHead
  FlambdaMeetEnv FlambdaMeet FlambdaMeetSpec FlambdaMeetProofs
  FlambdaMeetEnvProofs FlambdaTypeCreatorsSpec FlambdaTypeCreatorsProofs
  FlambdaTypeRenameSpec FlambdaTypeRenameProofs FlambdaEnvRenameSpec
  FlambdaEnvRenameProofs FlambdaMeetTermsSpec FlambdaMeetRenameSpec
  FlambdaMeetRenameProofs FlambdaSimplifySwitchSpec
  FlambdaSimplifySwitchProofs FlambdaSimplifySwitchArmsSpec.
From OxCamlFlambda2Types Require FlambdaMeetExamplesProofs
  FlambdaMeetClosuresProofs FlambdaMeetTermsProofs FlambdaSimplifyPassSpec
  FlambdaSimplifyPassProofs FlambdaSimplifyPassRenamingProofs.
Import ListNotations.

Local Open Scope Z_scope.

(* ======================================================================== *)
(* What an arm knows                                                        *)
(* ======================================================================== *)

(* A variable that may occur in terms is known. *)
Lemma in_scope_in_terms_known (tenv : Typing_env.t) (x : variable) :
  in_scope_in_terms tenv x -> Typing_env.mem None tenv (Name_var x) = true.
Proof.
  unfold in_scope_in_terms, Typing_env.mem.
  destruct (Keyed.lookup name_eqb (Name_var x) (Typing_env.names_to_types tenv))
    as [[ty btm] | ]; [ reflexivity | discriminate ].
Qed.

(* The two types that Simplify meets for an arm, in a state in which
   the scrutinee is the number of the arm. *)
Lemma arm_meet_inputs (statics : static_env) (valuation : env)
    (tenv : Typing_env.t) (s canonical : simple) (ty : flambda_type) (i : Z) :
  typing_env_fits tenv ->
  Typing_env.type_simple_in_term_exn tenv (Some Name_mode.normal) s
  = Some (Some (ty, canonical)) ->
  typing_env_satisfied_strictly statics valuation tenv ->
  eval_simple valuation s = Some (Value_const (Naked_immediate i)) ->
  meet_inputs_well_formed tenv ty (this_naked_immediate i)
  ∧ type_satisfied statics valuation (Value_const (Naked_immediate i)) ty
  ∧ type_satisfied statics valuation (Value_const (Naked_immediate i))
      (this_naked_immediate i).
Proof.
  intros Hfits Htype Hsatisfied Hvalue.
  pose proof (proj2 (type_simple_in_term_satisfied (fun _ => False) statics
                       valuation tenv (Some Name_mode.normal) s canonical ty
                       Hsatisfied Htype) _ Hvalue) as Hty.
  split; [ | split; [ exact Hty | apply this_const_satisfied; reflexivity ] ].
  split; [ exact Hfits | ].
  split; [ exact (type_simple_in_term_fits tenv _ s canonical ty Hfits Htype)
         | ].
  split.
  - apply (type_fits_alias tenv (Naked_number Kind_naked_immediate)
             (Const (Naked_immediate i))); reflexivity.
  - rewrite <- (type_satisfied_kind _ _ _ _ Hty). reflexivity.
Qed.

Theorem arm_env_satisfied : arm_env_satisfied_statement.
Proof.
  intros meet tenv tenv' s canonical ty t i statics vars Hmeet Hkeeps Hfits
    Hmem Htype Hanswer Hvalid [Hbound [valuation [Hsatisfied Hagrees]]] Heval.
  assert (Hvalue :
    eval_simple valuation s = Some (Value_const (Naked_immediate i))).
  { rewrite (agrees_eval_simple tenv valuation vars s Hagrees);
      [ exact Heval | ].
    intros x ->. exact Hmem. }
  destruct (arm_meet_inputs statics valuation tenv s canonical ty i Hfits
              Htype Hsatisfied Hvalue) as [Hinputs [Hty Hnumber]].
  destruct (Hmeet statics valuation tenv ty (this_naked_immediate i)
              (Value_const (Naked_immediate i)) _ Hinputs Hvalid Hsatisfied
              Hty Hnumber Hanswer)
    as [t0 [tenv0 [valuation' [Hresult [Hextended [Hsatisfied' _]]]]]].
  injection Hresult as <- <-.
  pose proof (Hkeeps tenv tenv' ty (this_naked_immediate i) t Hinputs Hanswer)
    as Hterms.
  split.
  - intros x Hx. exact (Hbound x (Hterms x Hx)).
  - exists valuation'. split; [ exact Hsatisfied' | ].
    intros x Hx.
    rewrite (Hextended x (in_scope_in_terms_known tenv x (Hterms x Hx))).
    exact (Hagrees x (Hterms x Hx)).
Qed.

(* ======================================================================== *)
(* Arms replaced by arms that refine them                                   *)
(* ======================================================================== *)

(* The arm of a number among arms that correspond one to one. *)
Lemma find_arm_forall2 (R : Z -> apply_cont -> apply_cont -> Prop) (i : Z) :
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

Theorem switch_arms_refined : switch_arms_refined_statement.
Proof.
  intros tenv s arms arms' Hall vars conts statics hp rd Hvalid Hsat.
  destruct (step_switch_cases s arms vars conts statics hp rd)
    as [Hundefined | [v [i [ac [Hv [Hi [Hfind Hstep]]]]]]].
  { apply step_undefined_refines, Hundefined. }
  pose proof (find_arm_forall2
                (fun i ac ac' => arm_refines_where_taken tenv s i ac ac') i
                arms arms' Hall) as Hfind'.
  rewrite Hfind in Hfind'. destruct Hfind' as [ac' [Hfind' Hrefines]].
  apply
    (refines_step _ _ _ (State (Apply_cont ac') vars conts statics hp rd)
       Hstep).
  - rewrite step_switch, Hv, Hi, Hfind'. reflexivity.
  - apply Hrefines; [ exact Hvalid | exact Hsat | ].
    rewrite Hv, (naked_immediate_of_value_some v i Hi). reflexivity.
Qed.

(* ======================================================================== *)
(* Simplify's function                                                      *)
(* ======================================================================== *)

Theorem simplify_arm_refines : simplify_arm_refines_statement.
Proof.
  intros meet rewrite tenv s canonical ty i ac ac' Hmeet Hkeeps Hrewrite Hfits
    Hmem Htype Harm vars conts statics hp rd Hvalid Hsat Heval.
  unfold simplify_arm in Harm.
  destruct (meet tenv ty (this_naked_immediate i)) as [[[t tenv'] | ] | ]
    eqn:Hanswer; try discriminate.
  injection Harm as <-.
  apply Hrewrite; [ exact Hvalid | ].
  exact (arm_env_satisfied meet tenv tenv' s canonical ty t i statics vars
           Hmeet Hkeeps Hfits Hmem Htype Hanswer Hvalid Hsat Heval).
Qed.

(* Whether an arm is kept is whether [simplify_arm] answers an arm. *)
Lemma simplify_arm_kept (meet : meet_function) (rewrite : apply_cont_rewrite)
    (tenv : Typing_env.t) (ty : flambda_type) (i : Z) (ac : apply_cont) :
  arm_kept meet tenv ty i
  = option_map
      (fun arm => match arm with Some _ => true | None => false end)
      (simplify_arm meet rewrite tenv ty i ac).
Proof.
  unfold arm_kept, simplify_arm.
  destruct (meet tenv ty (this_naked_immediate i)) as [[[t tenv'] | ] | ];
    reflexivity.
Qed.

(* The arms that are answered are the arms that are kept, one for one,
   each as [simplify_arm] answers it. *)
Lemma simplify_arms_pruned (meet : meet_function)
    (rewrite : apply_cont_rewrite) (tenv : Typing_env.t) (ty : flambda_type) :
  ∀ arms arms' : list (Z * apply_cont),
    simplify_arms meet rewrite tenv ty arms = Some arms' ->
    ∃ kept : list (Z * apply_cont),
      prune_arms meet tenv ty arms = Some kept
      ∧ Forall2
          (fun arm arm' =>
             fst arm' = fst arm
             ∧ simplify_arm meet rewrite tenv ty (fst arm) (snd arm)
               = Some (Some (snd arm')))
          kept arms'.
Proof.
  induction arms as [ | [i ac] arms IH]; intros arms'; simpl.
  - intros [= <-]. exists []. split; [ reflexivity | constructor ].
  - rewrite (simplify_arm_kept meet rewrite tenv ty i ac).
    destruct (simplify_arm meet rewrite tenv ty i ac) as [arm | ] eqn:Harm;
      [ | discriminate ].
    destruct (simplify_arms meet rewrite tenv ty arms) as [rest | ];
      [ | discriminate ].
    destruct (IH rest eq_refl) as [kept [-> Hall]].
    cbn. intros [= <-].
    destruct arm as [ac' | ].
    + exists ((i, ac) :: kept). split; [ reflexivity | ].
      constructor; [ | exact Hall ]. split; [ reflexivity | exact Harm ].
    + exists kept. split; [ reflexivity | exact Hall ].
Qed.

Theorem simplify_arms_refines : simplify_arms_refines_statement.
Proof.
  intros meet rewrite tenv s canonical ty arms arms' Hmeet Hkeeps Hrewrite
    Hfits Hmem Htype Harms.
  destruct (simplify_arms_pruned meet rewrite tenv ty arms arms' Harms)
    as [kept [Hprune Hall]].
  apply (expr_refines_under_valid_trans tenv _ (Switch s kept)).
  { apply equivalent_under_valid_refines.
    exact (prune_arms_equivalent meet tenv s canonical ty arms kept
             (meet_answers_bottom meet Hmeet) Hfits Hmem Htype Hprune). }
  apply switch_arms_refined.
  eapply Forall2_impl; [ | exact Hall ].
  intros [i ac] [i' ac'] [Hfst Harm]. simpl in *.
  split; [ exact Hfst | ].
  exact (simplify_arm_refines meet rewrite tenv s canonical ty i ac ac' Hmeet
           Hkeeps Hrewrite Hfits Hmem Htype Harm).
Qed.

Theorem simplify_switch_arms_refines : simplify_switch_arms_refines_statement.
Proof.
  intros meet rewrite tenv s arms e Hmeet Hkeeps Hrewrite Hfits Hmem.
  unfold simplify_switch_arms.
  destruct (Typing_env.type_simple_in_term_exn tenv (Some Name_mode.normal) s)
    as [[[ty canonical] | ] | ] eqn:Htype; try discriminate.
  cbn.
  destruct (simplify_arms meet rewrite tenv ty arms) as [arms' | ] eqn:Harms;
    [ | discriminate ].
  intros [= <-].
  apply (expr_refines_under_valid_trans tenv _ (Switch s arms')).
  { exact (simplify_arms_refines meet rewrite tenv s canonical ty arms arms'
             Hmeet Hkeeps Hrewrite Hfits Hmem Htype Harms). }
  apply (expr_refines_under_valid_trans tenv _ (Switch canonical arms')).
  { apply equivalent_under_valid_refines, equivalent_under_valid.
    apply switch_canonical.
    split; [ exact Hmem | exists ty; exact Htype ]. }
  apply refines_under_valid, expr_refines_under_weaken, switch_of_arms_refines.
Qed.

Theorem simplify_apply_cont_sound : simplify_apply_cont_sound_statement.
Proof.
  intros tenv ac.
  apply equivalent_under_valid_refines, equivalent_under_valid.
  apply FlambdaSimplifyPassProofs.simplify_apply_cont_equivalent.
Qed.

Theorem simplify_arms_identity : simplify_arms_identity_statement.
Proof.
  intros meet tenv s ty arms.
  assert (Harms :
    ∀ (ty : flambda_type) (arms : list (Z * apply_cont)),
      simplify_arms meet (fun _ ac => ac) tenv ty arms
      = prune_arms meet tenv ty arms).
  { clear ty arms. intros ty arms.
    induction arms as [ | [i ac] arms IH]; [ reflexivity | ].
    cbn [simplify_arms prune_arms]. rewrite IH.
    unfold simplify_arm, arm_kept.
    destruct (meet tenv ty (this_naked_immediate i)) as [[[t tenv'] | ] | ];
      cbn; try reflexivity;
      destruct (prune_arms meet tenv ty arms); reflexivity. }
  split; [ apply Harms | ].
  unfold simplify_switch_arms, simplify_switch.
  destruct (Typing_env.type_simple_in_term_exn tenv (Some Name_mode.normal) s)
    as [[[ty' canonical] | ] | ]; try reflexivity.
  cbn. rewrite Harms. reflexivity.
Qed.

(* ======================================================================== *)
(* With the meet that the contract is proved of                             *)
(* ======================================================================== *)

Theorem simplify_switch_arms_of_meet_to_prove_refines :
  simplify_switch_arms_of_meet_to_prove_refines_statement.
Proof.
  intros join fuel rewrite tenv scrutinee arms e Hjoin.
  apply simplify_switch_arms_refines.
  - exact (proj1
             (FlambdaMeetClosuresProofs.meet_to_prove_sound join fuel Hjoin)).
  - exact (FlambdaMeetTermsProofs.meet_to_prove_keeps_terms join fuel Hjoin).
Qed.

(* ======================================================================== *)
(* Renaming                                                                 *)
(* ======================================================================== *)

Theorem simplify_arm_rename : simplify_arm_rename_statement.
Proof.
  intros meet rewrite r tenv ty i ac Hmeet Hrewrite. unfold simplify_arm.
  rewrite <- (rename_type_this_const r (Naked_immediate i)) at 1.
  unfold this_naked_immediate. rewrite Hmeet.
  destruct (meet tenv ty (this_const (Naked_immediate i)))
    as [[[t tenv'] | ] | ]; cbn; try reflexivity.
  rewrite Hrewrite. reflexivity.
Qed.

Theorem simplify_arms_rename : simplify_arms_rename_statement.
Proof.
  intros meet rewrite r tenv ty arms Hmeet Hrewrite.
  induction arms as [ | [i ac] arms IH]; [ reflexivity | ].
  cbn [rename_arms map fst snd simplify_arms].
  rewrite (simplify_arm_rename meet rewrite r tenv ty i ac Hmeet Hrewrite).
  destruct (simplify_arm meet rewrite tenv ty i ac) as [arm | ];
    [ | reflexivity ].
  change (map (fun arm => (fst arm, rename_apply_cont r (snd arm))) arms)
    with (rename_arms r arms).
  rewrite IH.
  destruct (simplify_arms meet rewrite tenv ty arms) as [rest | ];
    [ | reflexivity ].
  destruct arm; reflexivity.
Qed.

Theorem simplify_switch_arms_rename : simplify_switch_arms_rename_statement.
Proof.
  intros meet rewrite r tenv s arms Hmeet Hrewrite.
  unfold simplify_switch_arms.
  rewrite type_simple_in_term_exn_equivariant.
  destruct (Typing_env.type_simple_in_term_exn tenv (Some Name_mode.normal) s)
    as [[[ty canonical] | ] | ]; try reflexivity.
  cbn [option_map fst snd].
  rewrite (simplify_arms_rename meet rewrite r tenv ty arms Hmeet Hrewrite).
  destruct (simplify_arms meet rewrite tenv ty arms) as [arms' | ];
    [ | reflexivity ].
  cbn [option_map]. rewrite switch_of_arms_rename. reflexivity.
Qed.

Theorem simplify_switch_arms_of_meet_to_prove_rename :
  simplify_switch_arms_of_meet_to_prove_rename_statement.
Proof.
  intros join fuel rewrite r tenv s arms Hjoin Hrewrite.
  apply simplify_switch_arms_rename; [ | exact Hrewrite ].
  apply meet_to_prove_equivariant, Hjoin.
Qed.

Theorem simplify_apply_cont_rewrite_equivariant :
  simplify_apply_cont_rewrite_equivariant_statement.
Proof.
  exact FlambdaSimplifyPassRenamingProofs.simplify_apply_cont_equivariant.
Qed.

(* ======================================================================== *)
(* An example                                                               *)
(* ======================================================================== *)

Theorem example_switch_arms : example_switch_arms_statement.
Proof.
  intros k0 k1 k2.
  apply (let_typed_refines_under_valid _ example_i _ _ _ _ any_naked_bool).
  - vm_compute. reflexivity.
  - reflexivity.
  - intros y [].
  - intros statics hp hp' rd rd' vars v _ Hv. apply any_bool_satisfied.
    apply exec_named_pure in Hv as (Hv & _ & _); [ | reflexivity ].
    cbn [eval_named] in Hv.
    destruct (eval_simples vars _) as [vs | ]; [ | discriminate ].
    exact (int_comp_bool statics _ _ vs v Hv).
  - apply (simplify_switch_arms_refines example_meet
             FlambdaSimplifyPassSpec.simplify_apply_cont).
    + exact (proj1 (FlambdaMeetClosuresProofs.meet_to_prove_sound _ _
                      FlambdaMeetExamplesProofs.forgetful_join_sound)).
    + exact (FlambdaMeetTermsProofs.meet_to_prove_keeps_terms _ _
               FlambdaMeetExamplesProofs.forgetful_join_sound).
    + exact simplify_apply_cont_sound.
    + exact example_comparison_env_fits.
    + reflexivity.
    + vm_compute. reflexivity.
Qed.
